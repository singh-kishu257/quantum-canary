# Quantum Canary

Low-shot recovery of the parameters that decide whether a qubit is fit for
computation, from a fixed set of 17 short probe circuits per qubit:

| Probe | Circuits | Recovers |
|---|---|---|
| T1 decay | 3 | **T1**: energy relaxation time |
| Ramsey X/Y | 8 (4 delays × 2 quadratures) | **Δω**: qubit detuning (phase unwrapped across all delays) and **T2\***: Ramsey dephasing time |
| Hahn echo | 3 | **T2**: echo dephasing time (the T2 IBM reports) |
| Gate repetition | 3 (N frame-twirled native G·G⁻¹ pairs) | **ε_sx**: incoherent single-qubit gate error |

Every estimate comes with a 1σ uncertainty and the reduced χ² of its fit, and
each qubit gets a health flag (GREEN / YELLOW / RED).

**Hardware:** superconducting qubits on IBM Quantum (Qiskit Runtime).
**Simulation:** superconducting and trapped-ion qubits (trapped ions are
supported in simulation only).

## Install

```bash
pip install quantum-canary            # engine, simulator, online monitoring
pip install "quantum-canary[ibm]"     # + Qiskit Runtime, to run on IBM hardware
```

Python 3.10 or newer.

## Quick start (command line)

```bash
# In silico: 3 simulated superconducting qubits with every simulator noise
# mechanism on (T1 drift and switching, 1/f frequency drift, quasi-static
# dephasing, detuning, coherent gate error, varying readout error) and a
# 1-hour-old calibration.
quantum-canary simulate --architecture superconducting --qubits 3

# Local dry run of the hardware path on a fake IBM backend (no account needed).
quantum-canary ibm --fake fake_fez --qubits 0,1

# Real hardware (uses your saved Qiskit Runtime account).
quantum-canary ibm --backend ibm_fez --qubits 13,14,2

# Validation tests of the simulation layer.
quantum-canary selftest
```

Add `--json report.json` to save the full result, `--no-color` for plain
text, and `--verbose` to also print the engine's own log lines.

To run on IBM hardware, save your account once:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="<your token>")
```

## Quick start (Python)

Import the package as `canary`. Avoid `qc`, which Qiskit code conventionally
uses for a `QuantumCircuit`. `canary` does not clash with the usual names in
Qiskit, qiskit-ionq, Cirq (`cirq`), QuTiP (`qt`) or PennyLane (`qml`). Avoid
`from quantum_canary import *`, which would bring the generic module names
`simulation`, `online` and `inversion` into your namespace; import the names
you need instead (`from quantum_canary import run_ibm, run_simulation`).

```python
import quantum_canary as canary

# Simulation: true values are kept for comparison.
report = canary.run_simulation("superconducting", n_qubits=2, regime="nisq")
report.print()                        # or: text = report.render()

# IBM hardware.
from qiskit_ibm_runtime import QiskitRuntimeService
backend = QiskitRuntimeService().backend("ibm_fez")
report = canary.run_ibm(backend, qubits=[13, 14, 2])
report.print()                        # or: text = report.render()

for q in report.qubits:
    r = q.result                      # quantum_canary.InversionResult
    print(q.qubit, q.health, r.T1_s, r.T2_s, r.T2_ramsey_s, r.delta_omega, r.epsilon_sx)

data = report.to_dict()               # JSON-serialisable
```

### Priors

The probe delays are chosen from a prior for each qubit's T1 and T2:

- **LIVE**: T1, T2, gate error and readout error are read from the backend's
  calibration (`BackendProfile.from_ibm_backend`), and the delays are placed around them.
- **ARCHITECTURE DEFAULTS**: used when the backend's T1/T2 are unavailable;
  the T1 delays are then log-spaced over the plausible range.

The report states which one was used for every qubit, and where each
calibration value came from.

### Health flags

- **GREEN**: every probe fits its model, and T2* agrees with T2.
- **YELLOW**: a probe's χ²/dof exceeds its limit, or T2* and T2 differ by
  more than 20%. This indicates non-Markovian or coherent noise, or stale
  priors.
- **RED**: a probe fit failed to converge.

### Lower-level API

```python
profile = canary.BackendProfile.from_ibm_backend(backend, qubit=13)   # or canary.BackendProfile.from_architecture("superconducting")
circuits, meta = canary.build_probe_circuits(profile)           # 17 one-qubit circuits
# ... run them, collect counts (list of {"0": n0, "1": n1}) ...
result = canary.lindblad_inversion(counts, meta, profile,
                                   shots_t1=300, shots_ramsey=1000, shots_gate=500, shots_echo=500)
```

### Online monitoring inside a running workflow

`CanaryInterleaver` weaves Canary rounds into any list of algorithm circuits
(VQE, QAOA, ...), then reports per-round health and drift against a baseline:

```python
monitor = canary.CanaryInterleaver(qubit_ids=[0, 1], profiles={0: p0, 1: p1}, interval_n=10)
circuits, round_map = monitor.prepare(algo_circuits)
# ... run `circuits`, collect counts in order ...
algo_results, rounds = monitor.process(all_counts, round_map)
monitor.drift_report(0)
```

### Simulator

`quantum_canary.simulation` is the device simulator Canary is validated on:
literature-sourced T1/T2/gate-error populations, T1 drift and two-state T1
switching, 1/f and OU frequency drift, quasi-static dephasing, detuning
(including during gate pulses), coherent gate error, per-qubit readout
heterogeneity, motional heating (trapped ions), stale calibration priors and
time-resolved execution. `run_simulation(regime="nisq")` switches all of them
on (crosstalk needs two qubits in a circuit and does not act there). Every
numerical default is registered with its source in
`quantum_canary.simulation.MODEL_PROVENANCE`.

## Shot budget

Per qubit: T1 3 × 300, Ramsey 8 × 1000, gate repetition 3 × 500 and echo
3 × 500, i.e. **11,900 shots**, run by default as 10 interleaved blocks so
every probe samples the same time window.

## Repository layout

```
src/quantum_canary/   the installable package
  inversion.py        Canary engine (identical to research/1_inversion.py)
  online.py           online monitoring (research/1b_online.py)
  simulation.py       device simulator (research/1c_simulation_models.py)
  runner.py           run_ibm(), run_simulation()
  display.py, cli.py  terminal report and `quantum-canary` command
research/             reference sources, benchmarks and hardware scripts
tests/                checks that the package equals the research sources
```

## License

MIT. See [LICENSE](LICENSE).
