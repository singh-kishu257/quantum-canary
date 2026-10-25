"""Fast RB on the ten qubits of amplitude_test, against qiskit-experiments' own RB on the same qubits right after.
Fast RB: per qubit 3 lengths (1/16, 1/2, 1 of m = 0.3 / IBM sx error, where the decay reaches ~e^-0.5) x 10/20/30 random
Clifford sequences x 24 shots = 1,440 shots. Every qubit gets its own sequences; each Clifford is compiled as qiskit-experiments
compiles it (rz, at most one sx or x pulse, rz); half the sequences end on |0>, half on |1>, so D(m) = P0(ends |0>) - P0(ends |1>)
= A p^m with readout cancelled. Fit D = A p^m, EPC = (1 - p) / 2, compared with the reference's EPC (same gate set); EPG_sx with
the reference's own EPG_sx / EPC conversion. Error bars from the spread between sequences, chi2/dof from the one spare length.
Reference: the sandwich's RB settings. The sx-only version that made the fast_rb_ibm_kingston_* runs is saved in those folders.
Usage: python research/fast_rb_test.py BACKEND"""
import json, pathlib, sys, time
import numpy as np
from scipy.optimize import curve_fit
from qiskit import QuantumCircuit
OUT, SEQS, SHOTS, SEED, FRACTIONS = pathlib.Path(__file__).resolve().parent / "data" / "hardware", (10, 20, 30), 24, 20261024, (1 / 16, 1 / 2, 1)


def clifford_table():
    """The 24 one-qubit Cliffords as (a, pulse, c) of rz(c).pulse.rz(a) with the fewest pulses (4 none, 16 sx, 4 x),
    their products and the inverse to each end state."""
    h, s, x = np.array([[1, 1], [1, -1]]) / np.sqrt(2), np.diag([1, 1j]), np.array([[0, 1], [1, 0]])
    same = lambda u, v: abs(np.trace(u.conj().T @ v)) > 1.999
    us = [np.eye(2)]
    for u in us:
        us += [g @ u for g in (h, s) if not any(same(g @ u, v) for v in us)]
    find = lambda u: next(i for i, v in enumerate(us) if same(u, v))
    rz = lambda t: np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])
    pulses = {None: np.eye(2), "sx": np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2, "x": x}
    forms = [(a, g, c) for g in pulses for a in np.arange(4) * np.pi / 2 for c in np.arange(4) * np.pi / 2]
    gates = [next(f for f in forms if same(rz(f[2]) @ pulses[f[1]] @ rz(f[0]), u)) for u in us]
    mult = [[find(v @ u) for v in us] for u in us]
    finish = [[find(t @ u.conj().T) for t in (np.eye(2), x)] for u in us]
    return gates, mult, finish


def lengths(eps):
    m = min(5000, max(16, round(0.3 / eps)))
    return [max(1, round(m * f)) for f in FRACTIONS]


def circuits(n, qubits, lens, pulse_dt, seed=SEED):
    """One circuit per (length, sequence); every qubit its own random sequence; even sequences end on |0>, odd on |1>.
    Shorter sequences wait in |0> at the start (in steps of 2 pulses) so every qubit ends at the measurement."""
    gates, mult, finish = clifford_table()
    out = []
    for s, k in enumerate(SEQS):
        for j in range(k):
            qc, seqs = QuantumCircuit(n, len(qubits)), {}
            for q in qubits:
                idx = list(np.random.default_rng([seed, q, s, j]).integers(24, size=lens[q][s]))
                net = 0
                for i in idx:
                    net = mult[net][i]
                idx.append(finish[net][j % 2])
                ops, carry = [], 0.0
                for i in idx:
                    a, g, c = gates[i]
                    if g:
                        ops += [("rz", a + carry), (g, None)]
                        carry = c
                    else:
                        carry += a + c
                seqs[q] = ops
            pulses = {q: sum(g != "rz" for g, _ in ops) for q, ops in seqs.items()}
            for q, ops in seqs.items():
                if pulse_dt and max(pulses.values()) - pulses[q] >= 2:
                    qc.delay(2 * pulse_dt * ((max(pulses.values()) - pulses[q]) // 2), q)
                for g, t in ops:
                    qc.rz(t, q) if g == "rz" else getattr(qc, g)(q)
            qc.barrier()
            qc.measure(qubits, range(len(qubits)))
            out.append(qc)
    return out


def fit(bits, qubits, lens):
    """bits: the shots of each circuit, in circuit order, as bitstrings. Returns {qubit: (EPC, sigma, chi2/dof)}."""
    rows, starts = {}, np.cumsum((0,) + SEQS)
    for i, q in enumerate(qubits):
        p0 = [np.mean([b[-1 - i] == "0" for b in shots]) for shots in bits]
        y = [(np.array(p0[starts[s]:starts[s + 1]][0::2]), np.array(p0[starts[s]:starts[s + 1]][1::2])) for s in range(len(SEQS))]
        d = np.array([a.mean() - b.mean() for a, b in y])
        sd = np.array([max(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)), 1e-3) for a, b in y])
        m = np.array(lens[q], dtype=float)
        model = lambda m, amp, p: amp * p ** m
        (amp, p), cov = curve_fit(model, m, d, p0=[max(d[0], 0.1), 1 - 0.8 / m[-1]], sigma=sd, absolute_sigma=True, bounds=([0, 0.5], [1.5, 1]))
        rows[q] = ((1 - p) / 2, float(np.sqrt(cov[1, 1])) / 2, float(np.sum(((d - model(m, amp, p)) / sd) ** 2)))
    return rows


def reference(child):
    v = {r["name"]: r["value"] for r in child.analysis_results(dataframe=True).to_dict("records")}
    return v["EPC"].nominal_value, v["EPC"].std_dev, v["EPG_sx"].nominal_value


if __name__ == "__main__":
    import quantum_canary as canary
    from provenance import write_meta
    from qiskit_ibm_runtime import SamplerV2
    from qiskit_experiments.framework import ParallelExperiment
    from qiskit_experiments.library import StandardRB
    name = sys.argv[1]
    group = json.loads((sorted(OUT.glob(f"{name}_*/qubits.json"))[-1]).read_text())["groups"][0]
    qubits = group[:: max(1, len(group) // 10)][:10]
    b = canary.credentials.ibm_service(OUT.parents[2] / "ibm_api_token.txt", OUT.parents[2] / "ibm_crn_instance.txt").backend(name)
    if b.service.usage().get("usage_remaining_seconds", 90) < 90:
        sys.exit("under 90 s of IBM time left: nothing submitted")
    p = b.properties(refresh=True)
    lens = {q: lengths(p.gate_error("sx", q)) for q in qubits}
    job = SamplerV2(mode=b).run(circuits(b.num_qubits, qubits, lens, round(b.target["sx"][(qubits[0],)].duration / b.dt)), shots=SHOTS)
    bits = [r.data.c.get_bitstrings() for r in job.result()]
    lmax = max(max(50, min(2000, int(0.5 / p.gate_error("sx", q)))) for q in qubits)
    ref = ParallelExperiment([StandardRB((q,), sorted({int(round(x)) for x in np.geomspace(1, lmax, 10)}), num_samples=10, seed=q) for q in qubits], backend=b, flatten_results=False).run(shots=200).block_for_results()
    usage = {"fast": float(b.service.job(job.job_id()).usage() or 0), "reference": sum(float(b.service.job(j).usage() or 0) for j in ref.job_ids)}
    rows = {q: (*f, *reference(rc)) for (q, f), rc in zip(fit(bits, qubits, lens).items(), ref.child_data())}
    run = OUT / f"fast_rb_qe_{name}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    run.mkdir(parents=True)
    (run / "calibration.json").write_text(json.dumps(p.to_dict(), default=str))
    (run / "fast_rb.json").write_text(json.dumps({"qubits": qubits, "lengths": lens, "sequences": SEQS, "shots": SHOTS, "seed": SEED,
        "usage_s": usage, "job_ids": {"fast": job.job_id(), "reference": ref.job_ids}, "results": rows,
        "raw": {"fast": bits, "reference": [c.data() for c in ref.child_data()]}}, default=str))
    write_meta(run / "fast_rb.json")
    print(f"fast RB: 3 lengths per qubit x {SEQS} sequences x {SHOTS} shots = {sum(SEQS) * SHOTS} shots, IBM usage {usage['fast']:.1f} s"
          f" | reference RB: 100 circuits x 200 shots = 20,000 shots, IBM usage {usage['reference']:.1f} s")
    print(f"{'qubit':>5} {'lengths':>18} {'fast EPC':>22} {'chi2':>6} {'reference EPC':>20} {'fast/ref':>8} {'z':>6} {'fast EPG_sx':>11} {'ref EPG_sx':>10}")
    for q, (fe, fs, chi, re_, rs, rg) in rows.items():
        print(f"{q:>5} {str(lens[q]):>18} {fe:>10.2e} ± {fs:.1e} ({fs / fe:4.0%}) {chi:>6.2f} {re_:>10.2e} ± {rs:.1e} {fe / re_:>8.2f} {(fe - re_) / np.hypot(fs, rs):>+6.2f} {fe * rg / re_:>11.2e} {rg:>10.2e}")
