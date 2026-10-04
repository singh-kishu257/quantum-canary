"""IBM hardware runs for the paper: is Canary right on real qubits, and does
it tell you something the backend's calibration does not?

Steps, each its own command, all on the same qubits of one backend:
  plan      pick the qubits by a fixed rule from the reported calibration
  smoke     Canary on one qubit, to measure what a run really costs
  canary    one Canary run (--tag A, B, C or D)
  qe        qiskit-experiments T1, T2Hahn and T2Ramsey on the same qubits
  usage     fill in QPU seconds for jobs whose usage was not ready yet
  analyze   the tables for the paper

Schedule: A, qe, B back to back; C late in the calibration cycle; D after
the next recalibration. Every run is its own job with its own timestamps.

Qubit rule: operational qubits with reported T1, T2 > 0, T2 <= 2 T1,
readout error <= 5% and sx error <= 1e-3, sorted by reported T1; one qubit
per evenly spaced rank, never two neighbours.

Fixed before any data:
  split-half    odd vs even shot blocks of one run share its time window, so
                honest error bars give |z| <= 2 with probability 0.954;
                pooled over all parameters and runs, passes unless a
                one-sided binomial test rejects that rate at p < 0.05
  qe            Canary vs QE within 2 sigma, the A-B difference counted as
                drift; pooled over T1, T2, T2* and delta_omega, passes
                unless a binomial test rejects a 90% rate at p < 0.05
  model check   a fit is broken if its chi2 has p < 0.01; Canary's T1,
                Ramsey and echo checks are scored against QE's dense curves
Live vs reported values, delta_omega, T2*/T2 and drift are reported as
measured.

--fake fake_fez runs the whole thing on a local fake backend.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import pathlib
import sys
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import binom, chi2 as chi2_dist, kstest

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_SRC = SCRIPT_DIR.parent / "src"
if REPO_SRC.is_dir():
    sys.path.insert(0, str(REPO_SRC))

import quantum_canary as canary
from quantum_canary import inversion as inv
from quantum_canary.runner import DEFAULT_PROBE_BLOCKS, DEFAULT_SHOTS, _counts_from_pub

DATA_DIR = SCRIPT_DIR / "data" / "hardware"
TAGS = ("A", "B", "C", "D")
RO_MAX, SX_MAX = 0.05, 1e-3
QE_SHOTS = 500
QE_N_DELAYS = {"t1": 20, "ramsey": 40, "hahn": 20}
QE_T1_SPAN, QE_RAMSEY_SPAN, QE_HAHN_SPAN = 3.0, 2.0, 3.0
QE_RAMSEY_OSC = 4.0
QE_N_PARAMS = {"t1": 3, "ramsey": 5, "hahn": 3}
CANARY_DOF = {"t1": 2, "ramsey": 6, "gate": 2, "echo": 2}
P_BROKEN = 0.01
SPLIT_TARGET, QE_TARGET, ALPHA = 0.9545, 0.90, 0.05
RESERVE_S = 30.0
PARAMS = {"T1": ("T1_s", "T1_sigma_s"), "T2": ("T2_s", "T2_sigma_s"),
          "T2star": ("T2_ramsey_s", "T2_ramsey_sigma_s"),
          "delta_omega": ("delta_omega", "delta_omega_sigma"),
          "epsilon_sx": ("epsilon_sx", "epsilon_sx_sigma")}
QE_PARAMS = ("T1", "T2", "T2star", "delta_omega")
FAMILY_MATCH = {"t1": "t1", "ramsey": "ramsey", "hahn": "echo"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_backend(args):
    if args.fake:
        from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return FakeProviderForBackendV2().backend(args.fake), None
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(channel="ibm_quantum_platform", instance=args.instance)
    return service.backend(args.backend), service


def out_dir(args) -> pathlib.Path:
    d = pathlib.Path(args.out) if args.out else DATA_DIR / (
        f"dryrun_{args.fake}" if args.fake else args.backend)
    d.mkdir(parents=True, exist_ok=True)
    return d


def load(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    return json.loads(path.read_text()) if path.exists() else None


def save(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=1, default=_jsonable))
    print(f"saved {path}")


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, datetime):
        return x.isoformat()
    return str(x)


def neighbours(backend) -> Dict[int, set]:
    out: Dict[int, set] = {q: set() for q in range(backend.num_qubits)}
    for a, b in backend.coupling_map.get_edges():
        out[a].add(b)
        out[b].add(a)
    return out


def select_qubits(backend, n: int) -> List[int]:
    props = backend.properties()
    cands = []
    for q in range(backend.num_qubits):
        try:
            t1, t2 = props.t1(q), props.t2(q)
            ro, sx = props.readout_error(q), props.gate_error("sx", q)
        except Exception:
            continue
        try:
            ok = props.is_qubit_operational(q)
        except Exception:
            ok = True
        if ok and t1 and t2 and t1 > 0 and t2 > 0 and t2 <= 2 * t1 and ro <= RO_MAX and sx <= SX_MAX:
            cands.append((float(t1), q))
    cands.sort()
    if len(cands) < n:
        raise RuntimeError(f"only {len(cands)} qubits pass the rule, need {n}")
    nbr, chosen = neighbours(backend), []
    for target in np.linspace(0, len(cands) - 1, n):
        for i in sorted(range(len(cands)), key=lambda i: (abs(i - target), i)):
            q = cands[i][1]
            if q not in chosen and not (nbr[q] & set(chosen)):
                chosen.append(q)
                break
    if len(chosen) < n:
        raise RuntimeError(f"could only place {len(chosen)} non-adjacent qubits")
    return chosen


def reported(backend, qubits: List[int], at: str) -> Dict[str, Any]:
    props = backend.properties()
    t_run = datetime.fromisoformat(at)
    out: Dict[str, Any] = {"last_update_date": props.last_update_date, "qubits": {}}
    for q in qubits:
        row: Dict[str, Any] = {}
        for key in ("T1", "T2", "readout_error", "prob_meas1_prep0", "prob_meas0_prep1"):
            try:
                value, date = props.qubit_property(q, key)
                row[key] = float(value)
                row[f"{key}_date"] = date
                if date is not None:
                    d = date if date.tzinfo else date.replace(tzinfo=timezone.utc)
                    row[f"{key}_age_h"] = (t_run - d).total_seconds() / 3600.0
            except Exception:
                row[key] = None
        try:
            value, date = props.gate_property("sx", [q], "gate_error")
            row["sx_error"], row["sx_error_date"] = float(value), date
        except Exception:
            row["sx_error"] = None
        out["qubits"][str(q)] = row
    return out


class Recorder:
    """A SamplerV2 that keeps the job, the PUBs and the raw result."""

    def __init__(self, backend):
        from qiskit_ibm_runtime import SamplerV2
        self.sampler = SamplerV2(mode=backend)
        self.options = self.sampler.options
        self.pubs: List[Any] = []
        self.job = None
        self.raw = None

    def run(self, pubs):
        self.pubs = list(pubs)
        self.job = self.sampler.run(self.pubs)
        rec = self

        class _Job:
            def job_id(self):
                return rec.job.job_id()

            def result(self):
                if rec.raw is None:
                    rec.raw = rec.job.result()
                return rec.raw

        return _Job()


def execution_spans(result) -> Optional[List[Dict[str, Any]]]:
    md = getattr(result, "metadata", None)
    ex = md.get("execution") if isinstance(md, dict) else None
    spans = ex.get("execution_spans") if isinstance(ex, dict) else None
    if not spans:
        return None
    return [{"start": sp.start, "stop": sp.stop, "pubs": list(sp.pub_idxs), "size": int(sp.size)}
            for sp in spans]


def qpu_seconds(job, fake: bool, wait_s: float = 300.0) -> Optional[float]:
    if fake:
        return None
    t0 = time.time()
    while True:
        try:
            u = job.usage()
        except Exception:
            u = None
        if u:
            return float(u)
        if time.time() - t0 > wait_s:
            return None
        time.sleep(10)


def remaining_seconds(service) -> Optional[float]:
    if service is None:
        return None
    try:
        return service.usage().get("usage_remaining_seconds")
    except Exception:
        return None


def guard(args, service, d: pathlib.Path, cost_s: Optional[float]) -> None:
    left = remaining_seconds(service)
    if service is None:
        return
    print(f"QPU time left: {left if left is not None else 'unknown'} s; "
          f"this job: {f'~{cost_s:.0f} s' if cost_s else 'unknown (run smoke first)'}")
    if cost_s is None and not args.force:
        raise SystemExit("no cost estimate yet: run smoke first, or pass --force")
    if left is not None and cost_s is not None and cost_s > left - RESERVE_S and not args.force:
        raise SystemExit("not enough QPU time left for this job; pass --force to try anyway")


def family_shots(meta: Dict[str, Any], shots: Dict[str, int]) -> List[int]:
    return ([shots["t1"]] * meta["n_t1"] + [shots["ramsey"]] * meta["n_ramsey"]
            + [shots["gate"]] * meta["n_gate"] + [shots["echo"]] * meta["n_echo"])


def invert(counts, meta, profile, shots, qubit) -> Dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = inv.lindblad_inversion(counts, meta, profile, shots_t1=shots["t1"],
                                   shots_ramsey=shots["ramsey"], shots_gate=shots["gate"],
                                   shots_echo=shots["echo"], qubit_id=qubit, timestamp=now())
    out = dataclasses.asdict(r)
    out.pop("calibration_source", None)
    return out


def split_half(report, rec: Recorder, probe_blocks: int) -> Dict[str, Any]:
    """Refit each qubit from the even and the odd shot blocks separately."""
    shots = report.shots
    metas = {qr.qubit: inv.build_probe_circuits(qr.profile)[1] for qr in report.qubits}
    shots_list = [s for qr in report.qubits for s in family_shots(metas[qr.qubit], shots)]
    n = len(shots_list)
    _, blk_shots, blk_index = inv.interleave_probe_blocks(list(range(n)), shots_list, probe_blocks)
    blk_no = [b for b in range(probe_blocks) for sh in shots_list
              if divmod(sh, probe_blocks)[0] + (1 if b < divmod(sh, probe_blocks)[1] else 0) > 0]
    if blk_shots != [p[2] for p in rec.pubs] or len(blk_no) != len(blk_index):
        raise RuntimeError("block plan does not match the submitted job")
    counts = [_counts_from_pub(rec.raw[j], rec.pubs[j][0], rec.pubs[j][2]) for j in range(len(rec.pubs))]
    merged = {"all": inv.merge_block_counts(counts, blk_index, n)}
    for h in (0, 1):
        idx = [j for j in range(len(counts)) if blk_no[j] % 2 == h]
        merged[h] = inv.merge_block_counts([counts[j] for j in idx], [blk_index[j] for j in idx], n)
    out, start = {}, 0
    for qr in report.qubits:
        meta = metas[qr.qubit]
        k = len(family_shots(meta, shots))
        full = invert(merged["all"][start:start + k], meta, qr.profile, shots, qr.qubit)
        if not np.isclose(full["T1_s"], qr.result.T1_s, rtol=1e-9):
            raise RuntimeError(f"refit of qubit {qr.qubit} does not reproduce the run")
        halves = []
        for h in (0, 1):
            c = merged[h][start:start + k]
            hs = {}
            for fam, lo, hi in (("t1", 0, meta["n_t1"]),
                                ("ramsey", meta["n_t1"], meta["n_t1"] + meta["n_ramsey"]),
                                ("gate", meta["n_t1"] + meta["n_ramsey"],
                                 meta["n_t1"] + meta["n_ramsey"] + meta["n_gate"]),
                                ("echo", k - meta["n_echo"], k)):
                tot = {sum(x.values()) for x in c[lo:hi]}
                if len(tot) != 1:
                    raise RuntimeError(f"uneven half-shots in the {fam} family")
                hs[fam] = tot.pop()
            halves.append(invert(c, meta, qr.profile, hs, qr.qubit))
        out[str(qr.qubit)] = halves
        start += k
    return {"halves": out, "block_counts": counts, "block_index": blk_index, "block_no": blk_no}


def run_canary(args, backend, service, qubits: List[int], tag: str, d: pathlib.Path,
               cost_s: Optional[float]) -> Dict[str, Any]:
    guard(args, service, d, cost_s)
    rec = Recorder(backend)
    t_submit = now()
    rep_before = reported(backend, qubits, t_submit)
    report = canary.run_ibm(backend, qubits, probe_blocks=args.probe_blocks, sampler=rec)
    t_done = now()
    sh = split_half(report, rec, args.probe_blocks)
    payload = {"tag": tag, "backend": backend.name, "fake": bool(args.fake),
               "qubits": qubits, "t_submit": t_submit, "t_done": t_done,
               "job_id": report.job_id, "qpu_seconds": qpu_seconds(rec.job, bool(args.fake)),
               "n_pubs": len(rec.pubs), "shots_total": int(sum(p[2] for p in rec.pubs)),
               "probe_blocks": args.probe_blocks, "sampler_options": str(rec.options),
               "execution_spans": execution_spans(rec.raw), "reported": rep_before,
               "report": report.to_dict(), "split_half": sh["halves"],
               "block_counts": sh["block_counts"], "block_index": sh["block_index"],
               "block_no": sh["block_no"]}
    report.print()
    return payload


def qe_experiment(backend, qubits: List[int], priors: Dict[int, Dict[str, float]]):
    from qiskit_experiments.framework import BatchExperiment, ParallelExperiment
    from qiskit_experiments.library import T1, T2Hahn, T2Ramsey
    grid = lambda span, n: [float(x) for x in np.linspace(0.0, span, n + 1)[1:]]
    osc = {q: QE_RAMSEY_OSC / priors[q]["T2"] for q in qubits}
    fams = {
        "t1": [T1((q,), delays=grid(QE_T1_SPAN * priors[q]["T1"], QE_N_DELAYS["t1"]))
               for q in qubits],
        "ramsey": [T2Ramsey((q,), delays=grid(QE_RAMSEY_SPAN * priors[q]["T2"], QE_N_DELAYS["ramsey"]),
                            osc_freq=osc[q]) for q in qubits],
        "hahn": [T2Hahn((q,), delays=grid(QE_HAHN_SPAN * priors[q]["T2"], QE_N_DELAYS["hahn"]),
                        num_echoes=1) for q in qubits],
    }
    for exps in fams.values():
        for e in exps:
            e.set_transpile_options(optimization_level=0)
            e.analysis.set_options(plot=False)
    pars = [ParallelExperiment(exps, backend=backend, flatten_results=True) for exps in fams.values()]
    return BatchExperiment(pars, backend=backend, flatten_results=True), osc


def _qubit_of(components) -> Optional[int]:
    text = "".join(ch for ch in str(components) if ch.isdigit())
    return int(text) if text else None


def qe_results(data, qubits: List[int], osc: Dict[int, float]) -> Dict[str, Any]:
    frame = data.analysis_results(dataframe=True)
    out: Dict[str, Any] = {str(q): {} for q in qubits}
    names = {"T1": "T1", "T2": "T2", "T2star": "T2star", "Frequency": "delta_omega"}
    for row in frame.to_dict("records"):
        q, name = _qubit_of(row.get("components")), str(row.get("name"))
        if q not in qubits or name not in names:
            continue
        v = row["value"]
        value = float(v.nominal_value) if hasattr(v, "nominal_value") else float(v)
        sigma = float(v.std_dev) if hasattr(v, "std_dev") else float("nan")
        if name == "Frequency":
            value, sigma = 2 * np.pi * (value - osc[q]), 2 * np.pi * sigma
        out[str(q)][names[name]] = {"value": value, "sigma": sigma, "quality": str(row.get("quality")),
                                    "experiment": str(row.get("experiment"))}
    fits = {str(q): {} for q in qubits}
    fam_of = {"T1": "t1", "T2Ramsey": "ramsey", "T2Hahn": "hahn"}
    try:
        arts = data.artifacts("fit_summary")
        arts = arts if isinstance(arts, list) else [arts]
    except Exception:
        arts = []
    for a in arts:
        q, fam = _qubit_of(a.device_components), fam_of.get(str(a.experiment))
        if q in qubits and fam:
            r = a.data
            fits[str(q)][fam] = {"reduced_chisq": float(getattr(r, "reduced_chisq", np.nan)),
                                 "dof": int(getattr(r, "dof", 0) or 0)}
    return {"values": out, "fits": fits}


def run_qe(args, backend, service, qubits: List[int], d: pathlib.Path,
           per_shot_s: Optional[float]) -> Dict[str, Any]:
    from qiskit_experiments.framework import ExperimentData
    priors = {}
    for q in qubits:
        with contextlib.redirect_stdout(io.StringIO()):
            p = inv.BackendProfile.from_ibm_backend(backend, qubit=q)
        priors[q] = {"T1": float(p.T1_prior_s), "T2": float(p.T2_prior_s)}
    batch, osc = qe_experiment(backend, qubits, priors)
    circuits = batch._transpiled_circuits()
    shots_total = QE_SHOTS * len(circuits)
    guard(args, service, d, 2.0 * per_shot_s * shots_total if per_shot_s else None)
    rec = Recorder(backend)
    t_submit = now()
    rep_before = reported(backend, qubits, t_submit)
    job = rec.run([(c, None, QE_SHOTS) for c in circuits])
    print(f"qe job {job.job_id()}: {len(circuits)} circuits x {QE_SHOTS} shots")
    result = job.result()
    t_done = now()
    counts = [result[i].join_data().get_counts() for i in range(len(circuits))]
    data = ExperimentData(experiment=batch)
    data.add_data([{"counts": c, "metadata": circ.metadata, "shots": QE_SHOTS}
                   for c, circ in zip(counts, circuits)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        batch.analysis.run(data, replace_results=True).block_for_results()
    res = qe_results(data, qubits, osc)
    for q in qubits:
        v = res["values"][str(q)]
        print(f"  q{q}: " + "  ".join(f"{k}={v[k]['value']:.4g}" for k in QE_PARAMS if k in v))
    return {"backend": backend.name, "fake": bool(args.fake), "qubits": qubits,
            "t_submit": t_submit, "t_done": t_done, "job_id": job.job_id(),
            "qpu_seconds": qpu_seconds(rec.job, bool(args.fake)), "n_pubs": len(circuits),
            "shots_total": shots_total, "shots_per_qubit": shots_total,
            "priors": {str(q): priors[q] for q in qubits},
            "osc_freq_hz": {str(q): osc[q] for q in qubits},
            "execution_spans": execution_spans(result), "reported": rep_before,
            "results": res, "counts": counts,
            "circuit_metadata": [c.metadata for c in circuits]}


def cmd_plan(args, backend, service, d):
    qubits = select_qubits(backend, args.n_qubits)
    t = now()
    save(d / "plan.json", {"backend": backend.name, "fake": bool(args.fake), "created": t,
                           "n_qubits": args.n_qubits, "qubits": qubits,
                           "rule": {"readout_error_max": RO_MAX, "sx_error_max": SX_MAX,
                                    "t2_le_2t1": True, "spread": "evenly spaced reported-T1 ranks",
                                    "adjacent": "never"},
                           "reported": reported(backend, qubits, t)})
    print(f"qubits: {qubits}")


def plan_qubits(d: pathlib.Path) -> List[int]:
    plan = load(d / "plan.json")
    if plan is None:
        raise SystemExit("run plan first")
    return plan["qubits"]


def cmd_smoke(args, backend, service, d):
    qubits = plan_qubits(d)
    payload = run_canary(args, backend, service, qubits[:1], "smoke", d, cost_s=60.0)
    save(d / "smoke.json", payload)
    s = payload["qpu_seconds"]
    if s:
        per_qubit, per_shot = s, s / payload["shots_total"]
        canary_run = per_qubit * len(qubits)
        qe_est = 2.0 * per_shot * QE_SHOTS * sum(QE_N_DELAYS.values())
        total = len(TAGS) * canary_run + qe_est
        print(f"\nsmoke: {s:.1f} QPU s for one qubit ({payload['shots_total']:,} shots)")
        print(f"projected: Canary {canary_run:.0f} s per run x {len(TAGS)} + QE ~{qe_est:.0f} s "
              f"= ~{total:.0f} s for {len(qubits)} qubits")
        left = remaining_seconds(service)
        if left is not None:
            print(f"QPU time left: {left:.0f} s"
                  + ("" if total <= left - RESERVE_S else
                     "  -> too much: rerun plan with fewer qubits"))


def smoke_costs(d: pathlib.Path):
    smoke = load(d / "smoke.json")
    if not smoke or not smoke.get("qpu_seconds"):
        return None, None
    return smoke["qpu_seconds"], smoke["qpu_seconds"] / smoke["shots_total"]


def cmd_canary(args, backend, service, d):
    qubits = plan_qubits(d)
    path = d / f"canary_{args.tag}.json"
    if path.exists() and not args.force:
        raise SystemExit(f"{path.name} exists; pass --force to overwrite")
    per_qubit, _ = smoke_costs(d)
    save(path, run_canary(args, backend, service, qubits, args.tag, d,
                          per_qubit * len(qubits) if per_qubit else None))


def cmd_qe(args, backend, service, d):
    qubits = plan_qubits(d)
    path = d / "qe.json"
    if path.exists() and not args.force:
        raise SystemExit(f"{path.name} exists; pass --force to overwrite")
    _, per_shot = smoke_costs(d)
    save(path, run_qe(args, backend, service, qubits, d, per_shot))


def cmd_usage(args, backend, service, d):
    if service is None:
        raise SystemExit("usage needs a real backend")
    for path in sorted(d.glob("*.json")):
        p = load(path)
        if p and p.get("job_id") and not p.get("qpu_seconds"):
            u = service.job(p["job_id"]).usage()
            if u:
                p["qpu_seconds"] = float(u)
                save(path, p)


def zscore(a, sa, b, sb) -> float:
    den = np.sqrt(sa ** 2 + sb ** 2)
    return float((a - b) / den) if np.isfinite(den) and den > 0 else float("nan")


def frac(xs, cond) -> str:
    xs = [x for x in xs if np.isfinite(x)]
    return f"{sum(cond(x) for x in xs)}/{len(xs)}" + (f" ({np.mean([cond(x) for x in xs]):.0%})" if xs else "")


def pooled_verdict(groups: Dict[str, List[float]], rate: float) -> Dict[str, Any]:
    zs = [z for g in groups.values() for z in g if np.isfinite(z)]
    k, n = sum(abs(z) <= 2 for z in zs), len(zs)
    p = float(binom.cdf(k, n, rate)) if n else float("nan")
    verdict = "pass" if n and p >= ALPHA else ("fail" if n else "no data")
    print(f"   pooled: {k}/{n} within 2 sigma, binomial p = {p:.3g} against {rate:.1%} -> {verdict}")
    return {"within": k, "n": n, "binomial_p": p, "rate": rate, "verdict": verdict}


def canary_values(run: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(q["qubit"]): q for q in run["report"]["qubits"]}


def cmd_analyze(args, backend, service, d):
    plan = load(d / "plan.json")
    runs = {t: load(d / f"canary_{t}.json") for t in TAGS}
    runs = {t: r for t, r in runs.items() if r}
    qe = load(d / "qe.json")
    if not runs:
        raise SystemExit("no Canary runs yet")
    qubits = [str(q) for q in plan["qubits"]]
    out: Dict[str, Any] = {"created": now(), "backend": plan["backend"], "qubits": plan["qubits"]}

    print("\n1. Are Canary's error bars honest? (even vs odd blocks of the same run)")
    split = {}
    for t, r in runs.items():
        for q, (h0, h1) in r["split_half"].items():
            for p, (vk, sk) in PARAMS.items():
                split.setdefault(p, []).append(zscore(h0[vk], h0[sk], h1[vk], h1[sk]))
    for p, zs in split.items():
        print(f"   {p:<12} |z|<=2 in {frac(zs, lambda z: abs(z) <= 2)}   mean z^2 "
              f"{np.nanmean(np.square(zs)):.2f}   (honest: 95%, mean z^2 ~1)")
    out["split_half_z"] = split
    out["split_half_verdict"] = pooled_verdict(split, SPLIT_TARGET)

    print("\n2. Drift between runs")
    tags = list(runs)
    drift = {}
    for a, b in list(zip(tags, tags[1:])) + ([(tags[0], tags[-1])] if len(tags) > 2 else []):
        ra, rb = canary_values(runs[a]), canary_values(runs[b])
        gap = (datetime.fromisoformat(runs[b]["t_submit"])
               - datetime.fromisoformat(runs[a]["t_submit"])).total_seconds() / 3600
        rows = {}
        for p, (vk, sk) in PARAMS.items():
            zs = [zscore(ra[q]["result"][vk], ra[q]["result"][sk], rb[q]["result"][vk],
                         rb[q]["result"][sk]) for q in qubits]
            rel = [abs(rb[q]["result"][vk] / ra[q]["result"][vk] - 1) for q in qubits
                   if p != "delta_omega" and ra[q]["result"][vk]]
            rows[p] = {"z": zs, "median_rel_change": float(np.median(rel)) if rel else None}
            print(f"   {a}->{b} ({gap:5.1f} h) {p:<12} moved beyond 2 sigma: "
                  f"{frac(zs, lambda z: abs(z) > 2)}"
                  + (f"   median change {np.median(rel):.1%}" if rel else ""))
        drift[f"{a}{b}"] = {"hours": gap, "params": rows}
    out["drift"] = drift

    if qe:
        print("\n3. Canary vs qiskit-experiments (A-B difference counted as drift)")
        qv = qe["results"]["values"]
        near = [t for t in ("A", "B") if t in runs] or tags[:1]
        agree = {}
        for p in QE_PARAMS:
            vk, sk = PARAMS[p]
            zs = []
            for q in qubits:
                if p not in qv.get(q, {}):
                    zs.append(float("nan"))
                    continue
                vals = [canary_values(runs[t])[q]["result"] for t in near]
                c = np.mean([v[vk] for v in vals])
                var = sum(v[sk] ** 2 for v in vals) / len(vals) ** 2
                if len(vals) == 2:
                    var += ((vals[0][vk] - vals[1][vk]) / 2) ** 2
                zs.append(float((c - qv[q][p]["value"]) / np.sqrt(var + qv[q][p]["sigma"] ** 2)))
            agree[p] = zs
            print(f"   {p:<12} within 2 sigma: {frac(zs, lambda z: abs(z) <= 2)}")
        out["canary_vs_qe_z"] = agree
        out["canary_vs_qe_verdict"] = pooled_verdict(agree, QE_TARGET)

        print("\n4. Do Canary's checks catch fits that QE's dense curves show are broken?")
        fits, table = qe["results"]["fits"], {}
        for fam, cfam in FAMILY_MATCH.items():
            cells = {"both": 0, "qe_only": 0, "canary_only": 0, "neither": 0, "no_fit": 0}
            for q in qubits:
                f = fits.get(q, {}).get(fam)
                if not f or not np.isfinite(f["reduced_chisq"]):
                    cells["no_fit"] += 1
                    continue
                dof = f["dof"] or (QE_N_DELAYS[fam] - QE_N_PARAMS[fam])
                qe_broken = chi2_dist.sf(f["reduced_chisq"] * dof, dof) < P_BROKEN
                k = CANARY_DOF[cfam]
                c_broken = any(chi2_dist.sf(canary_values(runs[t])[q]["result"][f"{cfam}_chi2_dof"] * k, k)
                               < P_BROKEN for t in near)
                cells[{(True, True): "both", (True, False): "qe_only",
                       (False, True): "canary_only", (False, False): "neither"}[(qe_broken, c_broken)]] += 1
            table[fam] = cells
            print(f"   {fam:<7} QE broken & Canary flagged {cells['both']}, QE broken & Canary quiet "
                  f"{cells['qe_only']}, Canary flagged & QE fine {cells['canary_only']}, both fine "
                  f"{cells['neither']}" + (f", no QE fit {cells['no_fit']}" if cells["no_fit"] else ""))
        out["model_check"] = table

    print("\n5. Live vs the backend's reported values")
    live = {}
    for t, r in runs.items():
        cv = canary_values(r)
        for key, vk, sk in (("T1", "T1_s", "T1_sigma_s"), ("T2", "T2_s", "T2_sigma_s")):
            ratios, zs, ages = [], [], []
            for q in qubits:
                rep = r["reported"]["qubits"][q]
                if not rep.get(key):
                    continue
                ratios.append(cv[q]["result"][vk] / rep[key])
                zs.append((cv[q]["result"][vk] - rep[key]) / cv[q]["result"][sk])
                ages.append(rep.get(f"{key}_age_h", float("nan")))
            live[f"{t}_{key}"] = {"ratio": ratios, "z": zs, "age_h": ages}
            print(f"   run {t} {key}: live/reported median {np.median(ratios):.2f} "
                  f"(range {min(ratios):.2f}-{max(ratios):.2f}), beyond 2 sigma "
                  f"{frac(zs, lambda z: abs(z) > 2)}, report age {np.nanmedian(ages):.1f} h")
    if qe:
        for key, p in (("T1", "T1"), ("T2", "T2")):
            r = [qe["results"]["values"][q][p]["value"] / qe["reported"]["qubits"][q][key]
                 for q in qubits if p in qe["results"]["values"].get(q, {})
                 and qe["reported"]["qubits"][q].get(key)]
            if r:
                print(f"   QE {key}: live/reported median {np.median(r):.2f} "
                      f"(range {min(r):.2f}-{max(r):.2f})")
    out["live_vs_reported"] = live

    print("\n6. What the backend does not report (per run, per qubit)")
    unrep = {}
    for t, r in runs.items():
        for q, v in canary_values(r).items():
            res = v["result"]
            dw = res["delta_omega"] / (2 * np.pi * 1e3)
            ratio = res["T2_ramsey_s"] / res["T2_s"] if res["T2_s"] else float("nan")
            unrep[f"{t}_{q}"] = {"delta_omega_khz": dw,
                                 "delta_omega_sigma_khz": res["delta_omega_sigma"] / (2 * np.pi * 1e3),
                                 "t2star_over_t2": ratio,
                                 "phase_per_10us_rad": abs(res["delta_omega"]) * 10e-6,
                                 "health": v["health"]}
        print(f"   run {t}: |dw|/2pi " + ", ".join(
            f"q{q} {abs(unrep[f'{t}_{q}']['delta_omega_khz']):.1f}" for q in qubits) + " kHz")
        print(f"          T2*/T2   " + ", ".join(
            f"q{q} {unrep[f'{t}_{q}']['t2star_over_t2']:.2f}" for q in qubits))
        print(f"          health   " + ", ".join(f"q{q} {unrep[f'{t}_{q}']['health']}" for q in qubits))
    out["unreported"] = unrep

    print("\n7. Canary chi2/dof on hardware vs theory")
    chi = {}
    for fam, k in CANARY_DOF.items():
        xs = [canary_values(r)[q]["result"][f"{fam}_chi2_dof"] for r in runs.values() for q in qubits]
        xs = [x for x in xs if np.isfinite(x)]
        if not xs:
            continue
        ks = kstest(np.array(xs) * k, chi2_dist(k).cdf)
        p_small = np.mean([chi2_dist.sf(x * k, k) < P_BROKEN for x in xs])
        chi[fam] = {"values": xs, "ks_p": float(ks.pvalue), "frac_p_below": float(p_small)}
        print(f"   {fam:<7} median {np.median(xs):.2f} (theory {chi2_dist(k).median() / k:.2f}), "
              f"p<{P_BROKEN} in {p_small:.0%} (theory {P_BROKEN:.0%}), KS p {ks.pvalue:.2g}")
    out["chi2"] = chi

    print("\n8. Cost")
    cost = {}
    for t, r in runs.items():
        cost[t] = {"qpu_seconds": r["qpu_seconds"], "shots_per_qubit": r["shots_total"] // len(qubits)}
        print(f"   Canary {t}: {r['qpu_seconds'] or 'n/a'} QPU s for {len(qubits)} qubits "
              f"({r['shots_total'] // len(qubits):,} shots per qubit, run one qubit after another)")
    if qe:
        cost["qe"] = {"qpu_seconds": qe["qpu_seconds"], "shots_per_qubit": qe["shots_per_qubit"]}
        print(f"   QE: {qe['qpu_seconds'] or 'n/a'} QPU s for {len(qubits)} qubits "
              f"({qe['shots_per_qubit']:,} shots per qubit, all qubits at once)")
    out["cost"] = cost
    save(d / "analysis.json", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("plan", "smoke", "canary", "qe", "usage", "analyze"))
    ap.add_argument("--backend", default="ibm_kingston")
    ap.add_argument("--instance", default=None)
    ap.add_argument("--fake", default=None, help="local fake backend, e.g. fake_fez")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-qubits", type=int, default=8)
    ap.add_argument("--tag", choices=TAGS)
    ap.add_argument("--probe-blocks", type=int, default=DEFAULT_PROBE_BLOCKS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.command == "canary" and not args.tag:
        ap.error("canary needs --tag")
    if args.probe_blocks < 2 or args.probe_blocks % 2:
        ap.error("--probe-blocks must be even for the split-half test")
    d = out_dir(args)
    if args.command == "analyze":
        cmd_analyze(args, None, None, d)
        return
    backend, service = get_backend(args)
    {"plan": cmd_plan, "smoke": cmd_smoke, "canary": cmd_canary, "qe": cmd_qe,
     "usage": cmd_usage}[args.command](args, backend, service, d)


if __name__ == "__main__":
    main()
