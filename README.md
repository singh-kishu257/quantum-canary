# ionpack

Pack several small, independent quantum circuits onto disjoint blocks of ions in **one** IonQ circuit, run them strictly one block after another, and get correct per-circuit results, statistics and cost savings back.

## What it does

`ionpack` places *k* small circuits on disjoint blocks of qubits (optionally separated by unused "gap" ions), concatenates them so that block 1 runs entirely before block 2, and so on, with all measurements at the end. It splits the joint histogram back into one histogram per circuit and computes statistics that account for correlations between blocks within a shot. It also computes the dollar saving of one packed circuit versus *k* separate circuits under a given price model.

## Why

IonQ bills **per circuit**: used directly or through Azure Quantum, a circuit is charged by gate-shots with a per-circuit minimum; on AWS Braket it is charged a per-task fee plus a per-shot fee ([Pricing](#pricing)). Many VQE and benchmarking circuits use only a few of the qubits of a Forte-class system ("Forte-class systems (which have 36 qubits)", [IonQ docs](https://docs.ionq.com/features/simulation-with-noise-models)).

For *k* identical blocks of gate cost *c* with a per-circuit minimum *m* (the `floor` model), packing saves a factor

    savings = min(k, m / c)   if c < m,      and 1 if c >= m.

On Braket (per-task + per-shot) the price does not depend on gates, and one packed circuit run for S shots replaces *k* circuits each run for S shots, so the saving is exactly *k*. **Large circuits under gate-shot pricing save nothing**: once a single block already costs more than the minimum, the packed circuit costs the same as the separate ones.

The price is time. ionpack assumes a machine that executes gates one at a time in submitted order, so a packed shot lasts about *k* times longer. On Forte, single-qubit gates take 110 µs and ZZ gates about 900 µs on average (Chen et al. 2024).

## The sequential-block lemma

Because blocks run strictly one after another and all measurements are terminal, every block's extra idle time falls either before its first gate (its ions are still in |0⟩) or after its last gate (it only waits for the final Z-basis readout). Noise that is diagonal in the computational basis — dephasing, including dephasing correlated across ions through shared *classical* fluctuations — commutes with Z-basis preparation and readout, so it leaves the Z-basis statistics of those idle windows exactly unchanged. The scope is limited: population-changing errors (T1 decay, X/Y-type crosstalk from other blocks' gates) are not covered, and time-in-shot effects such as motional heating or drift can still affect later blocks. ionpack does not prove this lemma; it **guarantees and checks** the sequential order (`pack`, `is_sequential`).

## Install

```text
pip install ionpack            # not yet on PyPI; for now: pip install -e .
pip install "ionpack[ionq]"    # adds qiskit-ionq for run_packed / packed_counts
```

Runtime dependencies: `numpy`, `scipy`, `qiskit`. Python ≥ 3.10.

## Quickstart

Runs offline on Qiskit's built-in `BasicSimulator`:

```python
from qiskit import QuantumCircuit
from qiskit.providers.basic_provider import BasicSimulator

from ionpack import (block_moments, is_sequential, linear_estimate, make_layout, pack,
                     sentinel_excitation, split_counts, z_parity)

bell = QuantumCircuit(2, 2)
bell.h(0)
bell.cx(0, 1)
bell.measure([0, 1], [0, 1])

ghz = QuantumCircuit(3, 3)
ghz.h(0)
ghz.cx(0, 1)
ghz.cx(1, 2)
ghz.measure([0, 1, 2], [0, 1, 2])

rot = QuantumCircuit(1, 1)
rot.ry(1.0, 0)
rot.measure(0, 0)

layout = make_layout([2, 3, 1], n_ions=36, gap=1)   # blocks (0,1), (3,4,5), (7,); gap ions 2, 6
packed = pack([bell, ghz, rot], layout)
assert is_sequential(packed)

counts = BasicSimulator().run(packed.circuit, shots=4000, seed_simulator=7).result().get_counts()
per_block = split_counts(counts, packed)              # one histogram per input circuit
print(per_block[0])                                   # Bell pair: only '00' and '11'
print(sentinel_excitation(counts, packed))            # gap ions should read 0

# <Z0 Z1> (Bell) + <Z0 Z2> (GHZ) + <Z> (rotation), with the within-shot covariance
m = block_moments(counts, packed, [z_parity([0, 1]), z_parity([0, 2]), z_parity([0])])
est, se = linear_estimate(m, [1.0, 1.0, 1.0])
print(f"estimate = {est:.3f} +/- {se:.3f}  (exact: 2 + cos(1) = 2.540)")
```

## Using with IonQ

Not executed in the tests (it needs an IonQ account). The order of operations matters:

1. transpile **each sub-circuit** to IonQ native gates, on a target of its own width;
2. `pack(..., native=True)`;
3. submit the packed circuit **unchanged** with `run_packed`; **never transpile the packed circuit** (that could interleave blocks).

Native circuits bypass IonQ's compiler and optimizer ([IonQ docs](https://docs.ionq.com/features/getting-started-with-native-gates)) and the native gateset has no delay instruction (qiskit-ionq rejects `delay`), so the submitted gate order is the executed order.

```python-ionq
from qiskit import transpile
from qiskit.circuit import Parameter
from qiskit.transpiler import Target
from qiskit_ionq import GPI2Gate, GPIGate, IonQProvider, ZZGate, add_equivalences
from ionpack import make_layout, pack, packed_counts, run_packed, split_counts

add_equivalences()                       # register IonQ native-gate equivalences with Qiskit
p = Parameter("p")
native = {"gpi": GPIGate(p), "gpi2": GPI2Gate(p), "zz": ZZGate(p)}   # Forte: zz; Aria: ms

def to_native(circ):                     # a target exactly as wide as the sub-circuit
    tgt = Target.from_configuration(["gpi", "gpi2", "zz", "measure"], num_qubits=circ.num_qubits,
                                    custom_name_mapping=native)
    return transpile(circ, target=tgt, optimization_level=1)

subs = [to_native(c) for c in circuits]
backend = IonQProvider().get_backend("qpu.forte-1", gateset="native")
packed = pack(subs, make_layout([c.num_qubits for c in subs], n_ions=36, gap=1), native=True)
job = run_packed(backend, packed, shots=1000)      # error mitigation (debiasing) off by default
per_block = split_counts(packed_counts(job), packed)
```

`run_packed` refuses a non-sequential circuit, warns if the backend is not native, and requests `ErrorMitigation.NO_DEBIASING` unless `debias=True`. `packed_counts` requests `sharpen=False`: sharpening is plurality voting across debiasing executions, which "could distort the true probability distribution" ([IonQ docs](https://docs.ionq.com/features/error-mitigation-debiasing); Maksymov et al. 2023) and hence expectation values.

## Statistics

* **Within one job:** `block_moments` computes the per-shot mean and covariance of one scalar per block (e.g. `z_parity` or `indicator`); `linear_estimate` returns θ = wᵀμ with SE = sqrt(wᵀΣw / S). This is exact for i.i.d. shots and includes the correlation between blocks in the same shot — e.g. a VQE energy Σ c_b ⟨P_b⟩ with blocks measuring different Pauli groups. It does **not** include job-to-job drift.
* **Across jobs:** `welch_tost` compares per-job estimates (e.g. packed vs. solo) with Welch's standard error and the two one-sided tests (TOST) procedure; the 1 − 2α interval must lie inside ±margin. Use it to check that packing does not change results beyond a tolerance, with drift between jobs accounted for.
* **Design effect:** for *k* identical copies, `design_effect(cov)` = 1ᵀΣ1 / tr Σ = 1 + (k−1)ρ̄ for equal variances; the effective number of shots per copy is k·S / design_effect. It applies only to identical copies.

## Pricing

| model | cost of one circuit (c = shots · (n1q·p1 + n2q·p2)) |
|---|---|
| `floor` | max(minimum, c) — IonQ: small jobs are "rounded up to this minimum value" |
| `additive` | minimum + c — the literal Azure formula AQT = m + … |
| `per_shot` | per_task + per_shot · shots — AWS Braket |

Published rates (retrieved 2026-10-01):

* **Azure Quantum, IonQ Forte 1 / Forte Enterprise 1** ([source](https://learn.microsoft.com/azure/quantum/pricing)): $0.0001645 per 1-qubit gate-shot, $0.001121 per 2-qubit gate-shot; minimum per program $25.7899 with error mitigation off, $168.195 with it on (the default). Preset: `AZURE_FORTE_RATES`.
* **AWS Braket, IonQ Forte** ([source](https://aws.amazon.com/braket/pricing/)): $0.30 per task + $0.08 per shot; error mitigation requires at least 2,500 shots per task. Preset: `BRAKET_FORTE`.
* **IonQ direct** ([source](https://docs.ionq.com/guides/job-cost-and-usage)): rates depend on your agreement with IonQ, so there is no preset; supply your contracted rates in a `Pricing` and use IonQ's cost estimator on that page. For multicircuit jobs "the total cost is the sum of individual child circuit job costs", so only packing into one circuit saves.

```python
from ionpack import AZURE_FORTE_RATES, BRAKET_FORTE, Pricing, packed_savings

azure_em_off = Pricing("floor", price_1q=AZURE_FORTE_RATES["price_1q"],
                       price_2q=AZURE_FORTE_RATES["price_2q"],
                       minimum=AZURE_FORTE_RATES["minimum_em_off"])
blocks = [(30, 12)] * 7                                   # 7 blocks of 30 1q + 12 2q gates
print(round(packed_savings(blocks, 300, azure_em_off), 2))  # 4.68
print(round(packed_savings(blocks, 300, BRAKET_FORTE), 2))  # 7.0
```

## Limitations

* IonQ does not document whether qubit index equals physical ion position, so gap ions are a hedge, not a guarantee of spatial separation.
* No mid-circuit measurement, reset or control flow: all measurements must be terminal.
* Debiasing (IonQ's error mitigation) permutes the qubit assignment, so gap and sentinel geometry is not preserved; aggregate such results by averaging.
* The serial-execution assumption does not hold for research systems that run gates in parallel.

## Citing

See [`CITATION.cff`](CITATION.cff).

## References

* J.-S. Chen et al., "Benchmarking a trapped-ion quantum computer with 30 qubits," *Quantum* 8, 1516 (2024), [arXiv:2308.05071](https://arxiv.org/abs/2308.05071).
* A. Maksymov, J. Nguyen, Y. Nam, I. Markov, "Enhancing quantum computer performance via symmetrization," [arXiv:2301.07233](https://arxiv.org/abs/2301.07233).
* P. Das, S. S. Tannu, P. J. Nair, M. Qureshi, "A case for multi-programming quantum computers," MICRO '52 (2019).
* Y. Ohkura, T. Satoh, R. Van Meter, "Simultaneous execution of quantum circuits on current and near-future NISQ systems," [arXiv:2112.07091](https://arxiv.org/abs/2112.07091).
* S. Niu, A. Todri-Sanial, "Multi-programming cross platform benchmarking for quantum computing hardware," [arXiv:2206.03144](https://arxiv.org/abs/2206.03144).
* A. Chatterjee et al., "Parallel circuit execution for scalable quantum computation," [arXiv:2609.12172](https://arxiv.org/abs/2609.12172).
* M. Palma et al., "Hardware-aware and resource-efficient circuit packing and scheduling on trapped-ion quantum computers," *IEEE Transactions on Quantum Engineering* (2025), [arXiv:2512.20554](https://arxiv.org/abs/2512.20554).
* D. J. Schuirmann, "A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability," *J. Pharmacokinet. Biopharm.* (1987).
