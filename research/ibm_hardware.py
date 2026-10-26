"""IBM runs for the paper; every Canary prior comes from IBM's latest calibration, refreshed at each step.
QE: IBM's standard configuration (Real-time benchmarking tutorial; T2Ramsey as in the qiskit-experiments manual).
Timing, identical for both methods (timing_TAG.json): wall_s and cpu_s of the method call on this machine (Canary =
check() without its wait for IBM's usage figure); per job IBM's created/running/finished timestamps and measured QPU
seconds, fetched after the timer; ibm_s and queue_s = their merged intervals; classical_s = wall_s - ibm_s.
Usage: sandwich BACKEND  |  canary BACKEND TAG [QUBITS]   e.g. canary ibm_kingston smoke 13  |  canary ibm_kingston C3"""
import json, pathlib, sys, time, quantum_canary as canary
from datetime import datetime, timezone
from qiskit_experiments.framework import BatchExperiment, ParallelExperiment
from qiskit_experiments.library import StandardRB, T1, T2Hahn, T2Ramsey
OUT, QE_SHOTS = pathlib.Path(__file__).resolve().parent / "data" / "hardware", {"t1": 1000, "hahn": 1000, "ramsey": 2000, "rb": 1000}
DELAYS, RAMSEY_DELAYS = [d * 1e-6 for d in (1, 20, 40, 80, 200, 400)], [d * 1e-6 for d in range(1, 50, 2)]

def merged_s(jobs, start, end):
    ts = lambda v: v if isinstance(v, datetime) else datetime.fromisoformat(str(v))
    total, last = 0.0, None
    for a, z in sorted((ts(j[start]), ts(j[end])) for j in jobs):
        total += max(0.0, (z - max(a, last or a)).total_seconds()); last = max(z, last or z)
    return total

def ibm_timing(service, ids, wall):
    jobs = []
    for i in ids:
        job, t = service.job(i), time.time()
        while not job.usage() and time.time() - t < 300: time.sleep(5)
        jobs.append({"job_id": i, "qpu_s": float(job.usage()) or None, **job.metrics().get("timestamps", {})})
    ibm = merged_s(jobs, "created", "finished")
    return {"qpu_s": sum(j["qpu_s"] for j in jobs) if all(j["qpu_s"] for j in jobs) else None, "ibm_s": ibm,
            "queue_s": merged_s(jobs, "created", "running"), "classical_s": wall - ibm, "jobs": jobs}

def step(name, tag, run, need_s, work, job_ids):
    b = canary.credentials.ibm_service(OUT.parents[2] / "ibm_api_token.txt", OUT.parents[2] / "ibm_crn_instance.txt").backend(name)
    if b.service.usage().get("usage_remaining_seconds", need_s) < need_s: sys.exit(f"under {need_s} s of IBM time left: not submitted")
    p, start = b.properties(refresh=True), datetime.now(timezone.utc).isoformat()
    t0, c0 = time.perf_counter(), time.process_time()
    result = work(b, p)
    timing = {"start_utc": start, "wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0, "job_ids": job_ids(result)}
    (run / f"timing_{tag}.json").write_text(json.dumps(timing, indent=1, default=str))
    try:
        timing.update(ibm_timing(b.service, job_ids(result), timing["wall_s"]))
    except Exception as e:
        timing["error"] = repr(e)
    (run / f"timing_{tag}.json").write_text(json.dumps(timing, indent=1, default=str))
    if hasattr(result, "ibm_usage_s"): result.ibm_usage_s = timing.get("qpu_s")
    (run / f"calibration_{tag}.json").write_text(json.dumps(p.to_dict(), default=str))
    jsonable = lambda v: [v.nominal_value, v.std_dev] if hasattr(v, "std_dev") else str(v)
    result.save(run / f"{tag}.json") if hasattr(result, "save") else (run / f"{tag}.json").write_text(json.dumps(result, default=jsonable))
    return result

def chip(b, p, run):
    full = lambda q: all(p.qubit_property(q).get(k) for k in ("T1", "T2", "prob_meas0_prep1", "prob_meas1_prep0"))
    qubits = [q for q in range(b.num_qubits) if p.is_qubit_operational(q) and full(q) and (q,) in p.gate_property("sx")]
    (run / "qubits.json").write_text(json.dumps({"qubits": qubits, "groups": canary.runner.probe_groups(b, qubits)}))
    return qubits

def qe(b, groups):
    make = {"t1": lambda q: T1((q,), DELAYS), "hahn": lambda q: T2Hahn((q,), DELAYS, num_echoes=1),
            "ramsey": lambda q: T2Ramsey((q,), RAMSEY_DELAYS, osc_freq=1e5), "rb": lambda q: StandardRB((q,), [10, 100, 500], num_samples=10, seed=q)}
    runs = {f: BatchExperiment([ParallelExperiment([make[f](q) for q in g]) for g in groups], backend=b).run(shots=s).block_for_results() for f, s in QE_SHOTS.items()}
    return {f: {"job_ids": d.job_ids, "raw": d.data(), "results": d.analysis_results(dataframe=True).to_dict("records")} for f, d in runs.items()}

if __name__ == "__main__":
    cmd, name, stamp = sys.argv[1], sys.argv[2], time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    saved = lambda run: json.loads((run / "qubits.json").read_text())
    check, c_ids, qe_ids = lambda b, q: canary.Report(canary.runner.run_ibm(b, q)), lambda r: [r.job_id], lambda r: [i for f in r.values() for i in f["job_ids"]]
    if cmd == "canary":
        qubits = [int(q) for q in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
        (run := OUT / f"{sys.argv[3]}_{name}_{stamp}" if qubits else sorted(OUT.glob(f"{name}_*/qubits.json"))[-1].parent).mkdir(parents=True, exist_ok=True)
        step(name, sys.argv[3], run, 60, lambda b, p: check(b, qubits or saved(run)["qubits"]), c_ids).print()
    elif cmd == "sandwich":
        (run := OUT / f"{name}_{stamp}").mkdir(parents=True)
        step(name, "C1", run, 200, lambda b, p: check(b, chip(b, p, run)), c_ids).print()
        for k in (1, 2):
            step(name, f"QE{k}", run, 140, lambda b, p: qe(b, saved(run)["groups"]), qe_ids)
            step(name, f"C{k + 1}", run, 45, lambda b, p: check(b, saved(run)["qubits"]), c_ids).print()
