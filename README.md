# Quantum Canary

Qubit health checks for IBM superconducting hardware.

1. **Before your job**: check the qubits you will use.
2. **Your jobs**: run as normal. Canary never touches them.
3. **Between jobs**, for workflows longer than about 30 minutes: check again
   every 20-30 minutes.
4. **After**: check again and compare. If a qubit changed, results that
   depend on it may not hold.

Each check is its own short job: 17 circuits per qubit (8,700 shots) that
measure T1, T2 (echo), T2\* (Ramsey), the qubit's frequency offset Δω and
its single-qubit gate error, with error bars and a check that the fits can be
trusted. Qubits are probed in at most two parallel groups, never two
neighbours in the same circuit, so checking more qubits costs little extra
QPU time.

## Install

```bash
pip install quantum-canary
```

Python 3.10 or newer. From a clone of this repository: `pip install -e .`.

## Use

```python
import quantum_canary as canary
from qiskit_ibm_runtime import QiskitRuntimeService

backend = QiskitRuntimeService().backend("ibm_kingston")

before = canary.check(backend, [13, 14])     # one short job
before.print()                               # numbers, fit, health, DD advice
before[13].t1, before[13].t2_star, before[13].health

# ... run your jobs ...

after = canary.check(backend, [13, 14])
d = canary.compare(before, after)
d.print()
d.changed                                    # qubits that moved beyond their error bars
```

For workflows longer than about 30 minutes, check between jobs:

```python
mon = canary.Monitor(backend, [13, 14])
mon.check()                    # baseline
for batch in my_batches:
    sampler.run(batch).result()            # your job, untouched
    mon.check()                            # a few seconds of QPU time
    if mon.changed:                        # moved since the baseline?
        ...                                # re-check, re-learn mitigation or move qubits
mon.print()                    # every check against the baseline
mon.save("checks/")
```

Checking many times raises the chance of a false alarm, so `Monitor` (and
`compare` with more than two files on the command line) raises the |z| limit
with the number of checks: the chance of any false alarm over the whole
workflow stays that of a single comparison.

Results can be saved and reloaded: `before.save("before.json")`,
`canary.load("before.json")`. No hardware at all: `canary.simulate(3)`.

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
quantum-canary compare c0.json c1.json c2.json           # every check vs the first
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
  your sampler with `dynamical_decoupling.enable = True`.
- **Drift** is a two-sample z-test on each parameter, Holm-corrected across
  the qubit's four parameters, at |z| > 2 (raised for repeated checks).
- Canary measures each qubit with its neighbours idle in |0⟩, the usual
  characterisation condition.

## IonQ

`canary.run_ionq(backend, ions, eps_prior=..., spam=...)` measures the
single-qubit gate error of chosen ions on IonQ hardware (T1 and T2 of trapped
ions are seconds long, so only the gate probe runs), one ion per circuit, with
IonQ's debiasing off. Pass `max_cost` to refuse a run whose IonQ estimate is
higher; Quantum Canary never prices runs itself.

## Advanced

- `canary.runner.run_ibm(backend, qubits)`: the full report behind `check`.
- `canary.runner.run_simulation(...)` and `canary.simulation`: the device
  simulator Canary was validated on, with every noise model and its source.
- `canary.workflow.check(backend, qubits, spares=...)` and
  `canary.workflow.run_monitored(backend, circuits, check_report)`: Canary
  rounds placed inside a job, after every N of your circuits. Use it with care:
  Qiskit Runtime does not guarantee that a job's circuits run in the order
  submitted (`run.execution_order` reports **held**, **violated**,
  **unresolved** or **unavailable** from IBM's timestamps, and a round shows
  the qubits between the circuits around it only when it **held**); rounds
  probe one qubit at a time, so they can cost as much as your own circuits;
  and dynamical decoupling must stay off in that job, since it would act on
  Canary's probe delays.

## License

MIT. See [LICENSE](LICENSE).
