# Quantum Canary

Qubit health checks before, during and after a job on IBM superconducting
hardware.

1. **Before the job**: check your qubits. Each one comes back GREEN, YELLOW or
   RED. RED qubits are swapped for healthy spares, and you are told whether
   to switch on dynamical decoupling.
2. **During the job**: a short Canary round runs on your qubits after every N
   of your circuits, in the same job.
3. **After the job**: check again and compare. If a qubit drifted, results
   that depend on it may not hold.

Each check runs 17 short circuits per qubit (11,900 shots) and measures T1,
T2 (echo), T2\* (Ramsey), the qubit's frequency offset Δω and its
single-qubit gate error.

## Install

```bash
pip install "quantum-canary[ibm]"
```

Python 3.10 or newer. From a clone of this repository: `pip install -e ".[ibm]"`.

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
quantum-canary check --backend ibm_fez --qubits 13,14 --spares 12
quantum-canary check --fake fake_fez --qubits 13,14
```

To use IBM hardware, save your account once:
`QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="...")`.

## What the answers mean

- **GREEN**: every probe fits its model, and T2\* agrees with T2.
- **YELLOW**: a probe fits poorly, or T2\* and T2 differ by more than 20%. The
  qubit has slow frequency noise, coherent errors, or has changed since its
  last calibration. Usable, but watch it.
- **RED**: a measurement failed. Swap the qubit out.
- **Dynamical decoupling** is recommended for a qubit when T2\*/T2 < 0.8
  (slow frequency noise that decoupling refocuses) or |Δω|·T2\* > 1 rad (its
  frequency offset turns a large phase while it sits idle). Turn it on in
  your sampler with `dynamical_decoupling.enable = True` for jobs without
  Canary rounds. Inside `run_monitored` leave it off: it would also act on
  Canary's own probe delays.
- **Drift** is a two-sample z-test on each parameter, Holm-corrected across
  the qubit's four parameters, at |z| > 2.

## Also included

- `canary.run_ibm(backend, qubits)`: the full per-qubit report behind `check`.
- `canary.run_simulation(...)` and `canary.simulation`: the device simulator
  Canary was validated on, with every noise model and its literature source.
- `canary.CanaryInterleaver`: the lower-level monitoring class behind
  `run_monitored`.

## License

MIT. See [LICENSE](LICENSE).
