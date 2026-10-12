# Quantum Canary

Qubit health checks before, during and after a job on IBM superconducting
hardware.

1. **Before the job**: check your qubits. Each one comes back GREEN, YELLOW or
   RED. RED qubits are swapped for healthy spares, and you are told whether
   to switch on dynamical decoupling.
2. **During the job**: a short Canary round is placed on your qubits after
   every N of your circuits, in the same job. Qiskit Runtime does not
   guarantee that a job's circuits run in the order submitted, so Canary
   reads IBM's execution timestamps after the job and reports whether that
   order actually held (see below).
3. **After the job**: check again and compare. If a qubit drifted, results
   that depend on it may not hold.

Each check runs 17 short circuits per qubit (11,900 shots) and measures T1,
T2 (echo), T2\* (Ramsey), the qubit's frequency offset Δω and its
single-qubit gate error. Qubits are checked in parallel, never two
neighbours in the same circuit, so checking more qubits costs little extra
QPU time (`run_ibm(..., parallel=False)` checks one qubit at a time).

## Install

```bash
pip install quantum-canary
```

Python 3.10 or newer. From a clone of this repository: `pip install -e .`.

## Use

```python
import quantum_canary as canary
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService

backend = QiskitRuntimeService().backend("ibm_fez")

# 1. Before
before = canary.check(backend, qubits=[13, 14], spares=[12])
before.print()

# 2. During: your circuits, transpiled onto before.layout
circuits = [transpile(c, backend, initial_layout=before.layout) for c in my_circuits]
run = canary.run_monitored(backend, circuits, before, interval=20)
run.print()
results = run.results            # your results, untouched and in order
run.execution_order              # "held", "violated", "unresolved" or "unavailable"

# 3. After
after = canary.check(backend, qubits=before.layout)
canary.compare(before, after).print()
```

Import the package as `canary` (Qiskit code usually uses `qc` for a circuit).

No IBM account yet? Use a fake backend with real device calibration, which
runs locally:

```python
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
backend = FakeProviderForBackendV2().backend("fake_fez")
```

From the terminal:

```bash
quantum-canary check ibm_kingston 13,14                  # IBM hardware
quantum-canary check fake_fez 13,14                      # local, no account
quantum-canary check ibm_kingston 13,14 --json before.json
quantum-canary compare before.json after.json            # what changed
```

`check` prints each qubit's T1, T2, T2\*, frequency offset and gate error with
error bars, a **fit** column (each probe's χ²/dof against its limit: whether
the numbers can be trusted) and the health verdict with its reason.

To use IBM hardware, save your account once:
`QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="...")`.

## What the answers mean

- **GREEN**: every probe's fit converged and passed its χ²/dof limit, and T2\*
  is within 20% of T2.
- **YELLOW**: a probe's χ²/dof is over its limit (that probe's numbers may be
  wrong), or T2\* and T2 differ by more than 20% (slow frequency noise). The
  report says which.
- **RED**: a fit did not converge. Re-run the check.
- The verdict describes the measurement, not how good the qubit is: a qubit
  with a short T1 that is measured cleanly is GREEN. Read the numbers for that.
- **Dynamical decoupling** is recommended for a qubit when T2\*/T2 < 0.8
  (slow frequency noise that decoupling refocuses) or |Δω|·T2\* > 1 rad (its
  frequency offset turns a large phase while it sits idle). Turn it on in
  your sampler with `dynamical_decoupling.enable = True` for jobs without
  Canary rounds. Inside `run_monitored` leave it off: it would also act on
  Canary's own probe delays.
- **Drift** is a two-sample z-test on each parameter, Holm-corrected across
  the qubit's four parameters, at |z| > 2.
- **Execution order** (in `run_monitored`): IBM reports time windows that
  are guaranteed to contain each circuit's shots, but may be wider than
  the true run time. Each block of your circuits and each Canary round gets
  the window covering all of its circuits. Canary then reports **held**
  (every block's window ends before the next one starts), **violated**
  (a later block's window ends before an earlier one starts), or
  **unresolved** (windows overlap, so the timestamps cannot tell). If the
  backend gives no timestamps, as with local simulators, it reports
  **unavailable**. A round shows the qubits' state between the circuits
  around it only when the order **held**.

## Also included

- `canary.run_ibm(backend, qubits)`: the full per-qubit report behind `check`.
- `canary.run_simulation(...)` and `canary.simulation`: the device simulator
  Canary was validated on, with every noise model and its literature source.
- `canary.CanaryInterleaver`: the lower-level monitoring class behind
  `run_monitored`.

## License

MIT. See [LICENSE](LICENSE).
