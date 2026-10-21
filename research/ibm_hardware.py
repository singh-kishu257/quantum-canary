"""IBM runs for the paper; every prior comes from IBM's latest calibration, refreshed at each step.
Usage: sandwich BACKEND  |  canary BACKEND TAG [QUBITS]   e.g. canary ibm_kingston smoke 13  |  canary ibm_kingston C3"""
import json, pathlib, sys, time, numpy as np, quantum_canary as canary
from provenance import write_meta
from qiskit_experiments.framework import BatchExperiment, ParallelExperiment
from qiskit_experiments.library import StandardRB, T1, T2Hahn, T2Ramsey
OUT, QE_SHOTS = pathlib.Path(__file__).resolve().parent / "data" / "hardware", {"t1": 500, "ramsey": 500, "hahn": 500, "rb": 200}

def step(name, tag, run, need_s, work):
    b = canary.credentials.ibm_service(OUT.parents[2] / "ibm_api_token.txt", OUT.parents[2] / "ibm_crn_instance.txt").backend(name)
    if b.service.usage().get("usage_remaining_seconds", need_s) < need_s: sys.exit(f"under {need_s} s of IBM time left: not submitted")
    result = work(b, p := b.properties(refresh=True))
    (run / f"calibration_{tag}.json").write_text(json.dumps(p.to_dict(), default=str))
    jsonable = lambda v: [v.nominal_value, v.std_dev] if hasattr(v, "std_dev") else str(v)
    result.save(run / f"{tag}.json") if hasattr(result, "save") else (run / f"{tag}.json").write_text(json.dumps(result, default=jsonable))
    write_meta(run / f"{tag}.json")
    return result

def chip(b, p, run):
    full = lambda q: all(p.qubit_property(q).get(k) for k in ("T1", "T2", "prob_meas0_prep1", "prob_meas1_prep0"))
    qubits = [q for q in range(b.num_qubits) if p.is_qubit_operational(q) and full(q) and (q,) in p.gate_property("sx")]
    (run / "qubits.json").write_text(json.dumps({"qubits": qubits, "groups": canary.runner.probe_groups(b, qubits)}))
    return qubits

def qe(b, p, groups):
    grid = lambda span, n: [float(x) for x in np.linspace(0, span, n + 1)[1:]]
    lengths = sorted({int(round(x)) for x in np.geomspace(1, max(max(50, min(2000, int(0.5 / p.gate_error("sx", q)))) for g in groups for q in g), 10)})
    make = {"t1": lambda q: T1((q,), grid(3 * p.t1(q), 20)), "ramsey": lambda q: T2Ramsey((q,), grid(2 * p.t2(q), 40), osc_freq=4 / p.t2(q)),
            "hahn": lambda q: T2Hahn((q,), grid(3 * p.t2(q), 20), num_echoes=1), "rb": lambda q: StandardRB((q,), lengths, num_samples=10, seed=q)}
    runs = {f: BatchExperiment([ParallelExperiment([make[f](q) for q in g]) for g in groups], backend=b).run(shots=s).block_for_results() for f, s in QE_SHOTS.items()}
    return {f: {"job_ids": d.job_ids, "raw": d.data(), "results": d.analysis_results(dataframe=True).to_dict("records")} for f, d in runs.items()}

if __name__ == "__main__":
    cmd, name, stamp = sys.argv[1], sys.argv[2], time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    saved = lambda run: json.loads((run / "qubits.json").read_text())
    if cmd == "canary":
        qubits = [int(q) for q in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
        (run := OUT / f"{sys.argv[3]}_{name}_{stamp}" if qubits else sorted(OUT.glob(f"{name}_*/qubits.json"))[-1].parent).mkdir(parents=True, exist_ok=True)
        step(name, sys.argv[3], run, 60, lambda b, p: canary.check(b, qubits or saved(run)["qubits"])).print()
    elif cmd == "sandwich":
        (run := OUT / f"{name}_{stamp}").mkdir(parents=True)
        step(name, "C1", run, 420, lambda b, p: canary.check(b, chip(b, p, run))).print()
        step(name, "QE", run, 300, lambda b, p: qe(b, p, saved(run)["groups"]))
        step(name, "C2", run, 60, lambda b, p: canary.check(b, saved(run)["qubits"])).print()
