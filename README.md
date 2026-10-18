# Quantum Canary

Qubit health checks for IBM superconducting hardware, and gate-error checks for IonQ trapped ions.

1. **Before your job**: check the qubits you will use.
2. **Your jobs**: run as normal. Canary never touches them.
3. **Between jobs**, if you like: check again as often as you want (most
   useful in long workflows).
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

To check between jobs, as often as you like:

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

## Credentials

Quantum Canary stores no secrets and searches for none. It uses the same lookup as
the vendors' libraries:

- **IBM:** the account saved once with
  `QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="...", instance="...")`
  (kept in `~/.qiskit/qiskit-ibm.json`, outside your project), or the
  `QISKIT_IBM_TOKEN` and `QISKIT_IBM_INSTANCE` environment variables.
- **IonQ** (an API key only; there is no instance or CRN): the `IONQ_API_KEY`
  environment variable.
- **A secret kept in a file** is used only when you name the file, in Python,
  `canary.credentials.ibm_service("ibm_api_token.txt", "ibm_crn_instance.txt")` and
  `canary.credentials.ionq_provider("ionq_api_token.txt")`, or on the command line,
  `--ibm-token-file`, `--ibm-instance-file` and `--ionq-token-file`. The file is read
  once and handed to the vendor's constructor; it is never printed, logged, exported
  or saved in a Report. Keep such files out of version control (add them to `.gitignore`).

## What the answers mean

- **GREEN**: every probe's fit converged and passed its χ²/dof limit, and T2\*
  is within 20% of T2.
- **YELLOW**: a probe's χ²/dof is over its limit (that probe's numbers may be
  wrong), or T2\* and T2 differ by more than 20% (slow frequency noise). The
  report says which.
- **RED**: a fit did not converge. Re-run the check.
- On simulated qubits that follow the model exactly, each probe goes over its
  limit about 1% of the time and a qubit comes out YELLOW 7-9% of the time:
  about half from a fit, half from the T2\*/T2 rule, which estimation noise
  alone can trip.
- The verdict describes the measurement, not how good the qubit is: a qubit
  with a short T1 that is measured cleanly is GREEN. Read the numbers for that.
- **Dynamical decoupling** is recommended for a qubit when T2\*/T2 < 0.8
  (slow frequency noise that decoupling refocuses). Turn it on in your
  sampler with `dynamical_decoupling.enable = True`.
- **Frequency offset**: when |Δω|·T2\* > 1 rad the qubit's offset turns a
  large phase while it sits idle. A steady offset is best removed with a
  virtual-Z frame update or a frequency recalibration; DD helps only if it
  drifts.
- **at bound** in the fit column means a fit stopped at its allowed limit
  (for example T2 = 2·T1), so that value's error bar is not reliable.
- **Drift** is a two-sample z-test on each parameter, Holm-corrected across
  the qubit's four parameters, at |z| > 2 (raised for repeated checks).
- Canary measures each qubit with its neighbours idle in |0⟩, the usual
  characterisation condition.

## IonQ

`canary.check` works on IonQ too: pass a qiskit-ionq backend
(`IonQProvider(key).get_backend("qpu.forte-1", gateset="native")`) and a list of
ions, or on the command line `quantum-canary check qpu.forte-1 0,3`
(see Credentials above). T1 and T2 of trapped ions are seconds long, so only the
gate error ε is measured, one ion per circuit, with IonQ's debiasing off. The
Report, `save`/`load`, `compare`, `Monitor` and the CLI work as on IBM; health is
RED if the gate fit failed and YELLOW if its χ²/dof is over the gate probe's
limit (there is no T2\*/T2 rule).

For a cost limit use `canary.run_ionq(backend, ions, max_cost=...)`, which
refuses a run whose IonQ estimate is higher, and
`canary.Report(canary.ionq.to_canary_report(...))` for the same Report.
Quantum Canary never prices runs itself.

## Advanced

- `canary.runner.run_ibm(backend, qubits)`: the full report behind `check`.
- `canary.runner.run_simulation(...)` and `canary.simulation`: the device
  simulator Canary was validated on, with every noise model and its source.

## License

MIT. See [LICENSE](LICENSE).
