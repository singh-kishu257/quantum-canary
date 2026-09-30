"""Hamiltonian simulation before vs after Quantum Canary on a Forte-like emulator built from IonQ's own data.

Device model per session ("a day on the machine"):
  * 12 ions chosen at random from the 30 qubits benchmarked by Chen et al. (arXiv:2308.05071).
  * Each ion pair's ZZ process infidelity is one draw from that pair's own direct-RB history in IonQ's public
    dataset (github.com/ionq-publications/forte_aq29_data, results/data_DRB2Q.json), so both persistent and
    episodic bad pairs appear at their measured frequencies.
  * Single-qubit pulse error 2.0e-4 and symmetrised readout error uniform in [0.2%, 0.8%] with 2:1 asymmetry
    (Chen et al. medians); coherent ZZ over-rotation N(0, 0.002 turns) per pair (assumption).
Workload: IonQ's 8-qubit Hamiltonian-simulation benchmark circuit HSIMN08V01 (63 ZZ interactions, 7 pairs x 9).
Every method receives the same total number of shots as Canary spends (algorithm + calibration).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_canary import Canary, CanaryConfig, LogicalProgram, compile_native
from quantum_canary.emulator import EmulatorTruth, _EmulatorExecutor
from quantum_canary.results import expectation, merge_frequencies, normalize_counts

FORTE = Path("/tmp/forte")


def load_pair_history() -> dict[tuple[int, int], list[float]]:
    d = json.loads((FORTE / "results" / "data_DRB2Q.json").read_text())
    hist: dict[tuple[int, int], list[float]] = collections.defaultdict(list)
    for k, e in d["err_2q"].items():
        a, b = int(d["q0"][k]), int(d["q1"][k])
        hist[(min(a, b), max(a, b))].append(float(e))
    return hist


def build_device(hist, rng, n_ions=12, over_sd=0.002, pulse=2.0e-4, seed=0):
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, coherent_unitary_error, depolarizing_error

    physical = sorted({q for p in hist for q in p})
    chosen = sorted(rng.choice(physical, size=n_ions, replace=False).tolist())
    pair_error, over = {}, {}
    for a in range(n_ions):
        for b in range(a + 1, n_ions):
            key = (min(chosen[a], chosen[b]), max(chosen[a], chosen[b]))
            pair_error[(a, b)] = min(float(rng.choice(hist[key])), 0.5)
            over[(a, b)] = float(over_sd * rng.standard_normal())
    readout = {}
    for i in range(n_ions):
        p = float(rng.uniform(0.002, 0.008))
        p10 = 2.0 * p * 2.0 / 3.0
        readout[i] = (2.0 * p - p10, p10)
    noise = NoiseModel()
    pe = depolarizing_error(4.0 * pulse / 3.0, 1)
    for i in range(n_ions):
        noise.add_quantum_error(pe, "u", [i])
        p01, p10 = readout[i]
        noise.add_readout_error(ReadoutError([[1 - p01, p01], [p10, 1 - p10]]), [i])
    parity = np.array([1.0, -1.0, -1.0, 1.0])
    for (a, b), e in pair_error.items():
        err = depolarizing_error(16.0 * e / 15.0, 2).compose(
            coherent_unitary_error(np.diag(np.exp(-1j * math.pi * over[(a, b)] * parity)))
        )
        noise.add_quantum_error(err, "rzz", [a, b])
        noise.add_quantum_error(err, "rzz", [b, a])
    ex = _EmulatorExecutor(AerSimulator(noise_model=noise), range(n_ions), seed)
    return ex, EmulatorTruth(pair_error, over, readout, pulse), chosen


def ideal_distribution(qc: QuantumCircuit) -> dict[str, float]:
    meas = [(qc.find_bit(i.qubits[0]).index, qc.find_bit(i.clbits[0]).index) for i in qc.data if i.operation.name == "measure"]
    probs = Statevector.from_instruction(qc.remove_final_measurements(inplace=False)).probabilities()
    out: dict[str, float] = {}
    for idx in np.flatnonzero(probs > 1e-14):
        bits = ["0"] * qc.num_clbits
        for q, c in meas:
            if (idx >> q) & 1:
                bits[qc.num_clbits - 1 - c] = "1"
        k = "".join(bits)
        out[k] = out.get(k, 0.0) + float(probs[idx])
    return out


def hellinger(p, q) -> float:
    return float(sum(math.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in set(p) | set(q)) ** 2)


def observable_rmse(freqs, ideal, n) -> float:
    obs = [(i,) for i in range(n)] + [(i, i + 1) for i in range(n - 1)]
    err = [expectation(freqs, o) - expectation(ideal, o) for o in obs]
    return float(np.sqrt(np.mean(np.square(err))))


def pair_quality(program, assignment, truth) -> float:
    return float(sum(w * truth.pair_infidelity(assignment[a], assignment[b]) for (a, b), w in program.interaction_weights.items()))


def run_session(s, args, hist, circ, program, ideal):
    rng = np.random.default_rng(1000 + s)
    ex, truth, chosen = build_device(hist, rng, args.ions, args.over_sd, seed=int(rng.integers(2**31)))
    n = program.num_qubits
    clock_t = [0.0]
    canary = Canary(ex, CanaryConfig(seed=int(rng.integers(2**31)), anneal_steps=4000), clock=lambda: clock_t[0])
    delivered, late = [], []
    ex_shots0 = 0
    for b in range(args.batches):
        clock_t[0] = 600.0 * (b + 1)
        res = canary.run(program, shots=args.shots)
        runs = [r for r in res.runs if r.trusted] or res.runs
        parts = [(r.frequencies, r.shots) for r in runs]
        delivered += parts
        if b >= args.batches - 2:
            late += parts
    plan = canary.plan(program, args.shots)
    total = args.batches * plan.total_shots
    final = canary.recommend(program)

    def fixed(assign, twirls=0, mt=False):
        return canary.run_fixed(program, assign, total, twirls=twirls, measure_twirl=mt).frequencies()

    ident = tuple(range(n))
    rand = tuple(int(x) for x in rng.permutation(args.ions)[:n])
    f_ident = fixed(ident)
    f_rand = fixed(rand)
    f_hard = fixed(ident, twirls=4, mt=True)
    k = 25
    assigns = [tuple(int(x) for x in rng.permutation(args.ions)[:n]) for _ in range(k)]
    per = max(1, total // k)
    raw = ex.run([compile_native(program, a, twirl=False) for a in assigns], per)
    f_debias = merge_frequencies([(normalize_counts(r), per) for r in raw])
    f_canary = merge_frequencies(delivered)
    f_late = merge_frequencies(late)
    f_final = fixed(final, twirls=4, mt=True)
    methods = {
        "identity (raw native)": f_ident,
        "random map (raw native)": f_rand,
        "debias-style avg (25 maps)": f_debias,
        "hardened only (twirl, identity map)": f_hard,
        "Canary delivered (all batches)": f_canary,
        "Canary last 2 batches": f_late,
        "Canary final map (deployed)": f_final,
    }
    row = {"session": s, "ions": chosen, "total_shots": total,
           "cal_fraction": plan.calibration_circuits / len(plan.circuits),
           "worst_pair_err": max(truth.pair_infidelity(a, b) for (a, b) in truth.pair_error),
           "pairs_over_1pct": sum(1 for (a, b) in truth.pair_error if truth.pair_infidelity(a, b) > 0.01),
           "Q_identity": pair_quality(program, ident, truth), "Q_random": pair_quality(program, rand, truth),
           "Q_final": pair_quality(program, final, truth),
           "degraded": [":".join(map(str, r)) for r in canary.degraded()]}
    for name, f in methods.items():
        row[f"F::{name}"] = hellinger(f, ideal)
        row[f"R::{name}"] = observable_rmse(f, ideal, n)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument("--ions", type=int, default=12)
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--over-sd", type=float, default=0.002)
    ap.add_argument("--circuit", default="HSIMN08V01")
    ap.add_argument("--out", default=str(ROOT / "research" / "hsim_results.json"))
    args = ap.parse_args()
    hist = load_pair_history()
    circ = QuantumCircuit.from_qasm_file(str(FORTE / "circuits" / "input" / f"{args.circuit}.qasm"))
    program = LogicalProgram.from_circuit(circ, name=args.circuit)
    ideal = ideal_distribution(program.to_circuit())
    rows = []
    t0 = time.time()
    for s in range(args.sessions):
        rows.append(run_session(s, args, hist, circ, program, ideal))
        r = rows[-1]
        print(f"session {s}: worst pair {r['worst_pair_err']:.4f}, >1%: {r['pairs_over_1pct']}, "
              f"F ident {r['F::identity (raw native)']:.3f} rand {r['F::random map (raw native)']:.3f} "
              f"debias {r['F::debias-style avg (25 maps)']:.3f} hard {r['F::hardened only (twirl, identity map)']:.3f} "
              f"canary {r['F::Canary delivered (all batches)']:.3f} final {r['F::Canary final map (deployed)']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        Path(args.out).write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
