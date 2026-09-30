# quantum-canary

Online ion-health learning and qubit-to-ion assignment for trapped-ion quantum computers (IonQ Forte).

On an all-to-all ion chain, any logical-to-ion assignment is free: same gates, same depth, same ideal output. Quantum Canary runs each batch of your circuit under a few deliberately chosen assignments. Because the ideal expectation values are identical across assignments, their differences localise errors to specific ions and ion pairs. Canary learns from the shots you already pay for, adds short ZZ return probes only for pairs it cannot judge, and steers subsequent batches onto the healthiest ions.

## Install

```bash
pip install -e ".[aer]"        # from a clone of this repository
pip install -e ".[test]"       # adds pytest; run: pytest
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

executor = IonQExecutor.qpu("forte-1")                  # or IonQExecutor.simulator("forte-1", ions=range(20))
canary = Canary(executor, CanaryConfig(seed=7))

print(canary.plan(qc, shots=2000))                      # preview circuits and gate-shot cost; nothing is sent

for _ in range(5):
    batch = canary.run(qc, shots=2000)
    counts = batch.counts()                              # trusted runs only
    print(batch.recommended, batch.degraded)

print(canary.report()[:5])
canary.save("canary_state.json")
```

The IonQ API key is read from `IONQ_API_KEY` or passed as `token=`. Circuits are submitted in IonQ native gates (GPi, GPi2, ZZ) with debiasing disabled, so each circuit runs on exactly the ions Canary chose. No delay or idle instructions are ever used.

| Call | Purpose |
|---|---|
| `canary.run(circuit, shots, symmetry="z2")` | One learning batch; `symmetry="z2"` enables the parity sentinel for circuits invariant under global X (e.g. MaxCut QAOA from \|+⟩) |
| `canary.run_many([c1, c2], shots)` | Several circuits co-scheduled in one job |
| `canary.plan(circuit, shots)` | Preview the next batch and its single- and two-qubit gate-shots without submitting |
| `canary.recommend(circuit)` | Current best assignment |
| `canary.run_fixed(circuit, assignment, shots, twirls=0)` | Baseline on a fixed assignment, no learning |
| `canary.probe([(i, j), ...], shots)` | Explicit cycle-benchmarking probes (packed up to `probe_pack` disjoint pairs per circuit) |
| `canary.sweep(shots)` | Optional warm start: every pair among the available ions probed once (12 ions: 36 circuits, 9,000 shots at `shots=250`) |
| `canary.report()` | Per-ion and per-pair error estimates, verdicts and whether each is probe-calibrated |
| `compile_native(circuit, assignment, rng)` | Standalone compiler to native gates on chosen ions |
| `BackendExecutor(AerSimulator(...), ions=range(n))` | Run on any Qiskit backend |

## How it works

1. **Program.** The circuit is transpiled to `rz, sx, x, rzz` with terminal measurements. Backward lightcones give, for every observable, the two-qubit gates on each logical pair that can affect it.
2. **Design.** Each batch uses one pessimistic (exploit) assignment (posterior mean plus `pessimism` standard deviations), solved by simulated annealing over the quadratic assignment objective, which gets `exploit_fraction` of the algorithm circuits. Exploratory assignments share the rest. With `explore_mode="local"` (default) each is the exploit with one ion moved or two swapped (at most three used pairs changed), chosen to test uncalibrated or suspect pairs, so its contrast with the exploit isolates a few pairs; `"thompson"` optimises independent posterior draws (uncalibrated resources drawn from the typical population component with `explore_typical`).
3. **Compile.** Two-qubit gates are Pauli-twirled (`twirl=True`), ZZ angles are normalised into (0, 0.25] turns, single-qubit runs are merged, and `rz` is realised as a virtual frame on subsequent GPi/GPi2 phases.
4. **Co-schedule.** Algorithm circuits and Tier-1 calibration circuits are sent as one multi-circuit job. Cycle-benchmarking probes pack up to `probe_pack` disjoint ion pairs into one circuit; each pair is estimated from the marginal of its own two clbits. Probe targets are chosen over all pairs among the available ions: first suspects (screening suspicion ≥ `trust_suspicion_z`, not yet probed) in the exploit seating, then other suspects, then by value of information with a coverage term (`coverage_weight`) that favours pairs never or not recently calibrated.
5. **Learn.** Only Tier-1 (cycle-benchmarking and SPAM) probes update the health model: a Kalman filter over per-resource damping rates under a typical-plus-outlier population prior, with forgetting (`half_life_s`). Algorithm results update a separate *screening* model (`screening=True`): for each observable, `log|⟨O⟩| = log|ideal| − Σ c_r x_r`, and centering within each batch cancels the unknown ideal value. Screening estimates are diluted (lightcone exposure is an upper bound) and are used only relative to their population, as a suspicion z-score, and after a per-program dilution calibration against Tier-1 estimates (`screen_calibration`). They guide placement among uncalibrated pairs, probe targeting and trust, and never enter the health model or the verdicts.
6. **Decide.** e-values with e-BH false-discovery-rate control label resources healthy or degraded from Tier-1 evidence only. Degraded resources are penalised in future assignments. A run is trusted if it touches no degraded resource and (with `trust_requires_vetting`) every pair it uses is calibrated or has suspicion below `trust_suspicion_z`; `batch.counts()` merges trusted runs, or returns the exploit run alone if none is trusted.

Error conventions: pair errors are **process (entanglement) infidelities**, the quantity reported by direct randomized benchmarking and by IonQ; for a two-qubit depolarizing channel with parameter λ, `e = 15λ/16` and every non-identity Pauli expectation is damped by `1 − λ`. Ion errors are symmetric readout-flip probabilities, damping `⟨Z⟩` by `1 − 2p`.

## Assumptions and limits

- **Exposure model.** Every two-qubit gate in an observable's backward lightcone is counted as one full exposure. The true sensitivity of an observable to a gate can be smaller, so rates learned from algorithm data alone are effective values that tend to understate damping. This is why a *healthy* verdict requires return-probe evidence, and why `report()` marks estimates that are not probe-calibrated.
- **Noise model.** The multiplicative model assumes stochastic (Pauli-like) noise, which twirling produces for Markovian noise (Wallman & Emerson 2016). Non-Markovian or device-wide noise is not attributable to specific ions and cannot be removed by reassignment.
- **Angle independence.** The damping of a ZZ gate on a pair is assumed independent of its rotation angle; return probes use the maximal angle (0.25 turns).
- **Statistics.** Observables computed from the same shots are treated as independent; `model_sd` adds a variance floor against the resulting overconfidence. SPRT error rates (`alpha`, `beta`) hold only under the Gaussian marginal approximation. Exploration is heuristic and carries no regret guarantee (the annealing solver is not an exact oracle).
- **Cost.** Twirling can add physical single-qubit gates (about 25% more in a 20-qubit Ising circuit); set `twirl=False` to disable. Return probes use `probe_fraction` of the job's circuits (0 disables them) and at most `max_probe_shots_per_ion` shots per ion.
- **Circuits.** Terminal measurements only; no mid-circuit measurement, reset or classical control. Circuits must fit the available ions; the IonQ cloud simulator is limited in width, so pass `ions=` accordingly.
- **Packed probes.** On hardware the pairs of a packed probe idle while the other pairs are gated, so packed estimates may include idle and crosstalk error that the emulator does not model. Compare packed and single-pair (`probe_pack=1`) estimates for a few pairs before relying on packing.
- **Screening.** Screening sees only resources that exploratory seatings exercise, is weakest for pairs whose contrasts are confounded with others, and its dilution factor is poorly determined when the Tier-1-calibrated pairs span a narrow error range. It is guidance: it can mis-rank pairs, but it never produces a verdict.
- **When it helps.** Canary improves results only when ions or pairs differ in quality. Uniform noise is not removable by assignment.

## References

- Chen, J.-S., et al. "Benchmarking a trapped-ion quantum computer with 30 qubits." Quantum 8, 1516 (2024). arXiv:2308.05071.
- Maksymov, A., et al. "Enhancing quantum computer performance via symmetrization." arXiv:2301.07233 (2023).
- Ravi, G. S., et al. "Boosting quantum fidelity with an ordered diverse ensemble of Clifford canary circuits." arXiv:2209.13732 (2022).
- Samanta, A., et al. "BAHAMAS: A control plane for optimization and execution of variational quantum circuits." SC '26, arXiv:2608.28811.
- Srivastava, S., et al. "Lightweight targeted estimation of layout noise in a quantum computer using quality indicator circuits." Quantum Inf. Process. (2026), arXiv:2509.18679.
- Wallman, J. J., and Emerson, J. "Noise tailoring for scalable quantum computation via randomized compiling." Phys. Rev. A 94, 052325 (2016).
- Wang, S., and Chen, W. "Thompson sampling for combinatorial semi-bandits." ICML 2018, arXiv:1803.04623.
- Wald, A. "Sequential tests of statistical hypotheses." Ann. Math. Statist. 16, 117–186 (1945).
