# quantum-canary

Online ion-health learning and qubit-to-ion assignment for trapped-ion quantum computers (IonQ Forte).

On an all-to-all ion chain, any logical-to-ion assignment is free: same gates, same depth, same ideal output. Canary runs each batch of your circuit under a few deliberately chosen assignments. Because the ideal expectation values are identical across assignments, their differences localise errors to specific ions and ion pairs. Canary learns ion health from the shots you already pay for, adds short echo probes only for ambiguous pairs, and steers subsequent batches onto the healthiest ions.

## Install

```bash
pip install quantum-canary            # IonQ cloud (QPU and simulator)
pip install "quantum-canary[aer]"     # plus qiskit-aer for local execution
```

## Quick start

```python
from qiskit import QuantumCircuit
from quantum_canary import Canary, CanaryConfig, IonQExecutor

qc = QuantumCircuit(4)
for _ in range(6):
    for q in range(3):
        qc.rzz(0.6, q, q + 1)
    qc.rx(0.5, range(4))
qc.measure_all()

executor = IonQExecutor.qpu("forte-1")                 # or IonQExecutor.simulator("forte-1", ions=range(20))
canary = Canary(executor, CanaryConfig(seed=7))

for _ in range(5):
    batch = canary.run(qc, shots=2000)
    counts = batch.counts()                             # trusted runs only
    print(batch.recommended, batch.degraded)

print(canary.report()[:5])
canary.save("canary_state.json")
```

The IonQ API key is read from `IONQ_API_KEY` or passed as `token=`. Circuits are submitted in IonQ native gates (GPi, GPi2, ZZ) with debiasing disabled, so each circuit runs on exactly the ions Canary chose.

Useful calls:

| Call | Purpose |
|---|---|
| `canary.run(circuit, shots, symmetry="z2")` | One learning batch; `symmetry="z2"` enables the parity sentinel for circuits invariant under global X (e.g. MaxCut QAOA from \|+⟩) |
| `canary.run_many([c1, c2], shots)` | Several circuits co-scheduled in one job |
| `canary.recommend(circuit)` | Current best assignment |
| `canary.run_fixed(circuit, assignment, shots, twirls=0)` | Baseline on a fixed assignment, no learning |
| `canary.probe([(i, j), ...], shots)` | Explicit echo probes |
| `canary.report()` | Per-ion and per-pair error estimates and verdicts |
| `compile_native(circuit, assignment, rng)` | Standalone compiler to native gates on chosen ions |
| `BackendExecutor(AerSimulator(...), ions=range(n))` | Run on any Qiskit backend |

## How it works

1. **Program.** The circuit is transpiled to `rz, sx, x, rzz` with terminal measurements. Backward lightcones give, for every observable, how many two-qubit gates on each logical pair can affect it.
2. **Design.** Each batch uses one pessimistic assignment (posterior mean plus `pessimism` standard deviations) and Thompson-sampled assignments drawn from the joint posterior (combinatorial Thompson sampling), each optimised by simulated annealing over the quadratic assignment objective.
3. **Compile.** Two-qubit gates are Pauli-twirled, ZZ angles are normalised into (0, 0.25] turns, single-qubit runs are merged, and `rz` is realised as a virtual frame on subsequent GPi/GPi2 phases.
4. **Co-schedule.** Algorithm circuits and echo probes (`ZZ(π/2)^4k` sandwiched by Hadamards, ideal outcome `|00⟩`) are sent as one multi-circuit job.
5. **Learn.** For each observable, `log|⟨O⟩| = log|ideal| − Σ c_r x_r`. Differences across assignments cancel the unknown ideal value; echo probes add known-ideal observations. A Gaussian posterior over per-ion and per-pair damping rates `x_r` is updated with exponential forgetting (`half_life_s`).
6. **Decide.** A Wald sequential probability ratio test labels each resource healthy, degraded or undecided. Degraded resources are penalised in future assignments, and runs that touched them are marked untrusted.

Damping rates convert to errors assuming depolarizing two-qubit gates, `r = 3/4 (1 − e^{−x})`, and symmetric readout flips, `p = (1 − e^{−x}) / 2`. Estimates from algorithm data are effective, lightcone-averaged rates; echo probes calibrate them on an absolute scale.

## Limits

- Terminal measurements only; no mid-circuit measurement, reset or classical control.
- Circuits must fit the available ions. The IonQ cloud simulator is limited in width; pass `ions=` accordingly.
- Canary improves results only when ion pairs differ in quality. Uniform noise is not removable by assignment.

## References

- J.-S. Chen et al., "Benchmarking a trapped-ion quantum computer with 30 qubits," Quantum (2024), arXiv:2308.05071.
- A. Maksymov, J. Nguyen, Y. Nam, I. Markov, "Enhancing quantum computer performance via symmetrization," arXiv:2301.07233.
- J. J. Wallman and J. Emerson, "Noise tailoring for scalable quantum computation via randomized compiling," Phys. Rev. A 94, 052325 (2016), arXiv:1512.01098.
- S. Wang and W. Chen, "Thompson Sampling for Combinatorial Semi-Bandits," arXiv:1803.04623.
- A. Wald, "Sequential tests of statistical hypotheses," Ann. Math. Statist. 16, 117–186 (1945).
