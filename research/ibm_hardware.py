"""IBM hardware runs for the paper: is Canary right on real qubits, and does
using it give a user better results than trusting the backend's calibration?

Steps, each its own command, all on the same qubits of one backend:
  plan      pick four 3-qubit chains by a fixed rule from the reported calibration
  smoke     Canary on one qubit, to check the whole pipeline on hardware
  canary    one Canary run on all chain qubits (--tag A, B, C or D)
  utility   the before/after test right after a Canary run (--tag A or C)
  qe        qiskit-experiments T1, T2Hahn and T2Ramsey on the same qubits
  usage     fill in QPU seconds for jobs whose usage was not ready yet
  analyze   the tables for the paper

Schedule: canary A, utility A, qe, canary B back to back; canary C and
utility C late in the calibration cycle; canary D after the next
recalibration. Every run is its own job with its own timestamps. Canary
probes qubits in parallel, never two neighbours in the same circuit, and qe
uses the same groups.

Chain rule: chains a-b-c of operational qubits with reported T1, T2 > 0,
T2 <= 2 T1, readout error <= 5%, sx error <= 1e-3 and CZ error <= 2% on both
edges, ranked by the fidelity the calibration predicts for the utility
workload; the best four that share no qubit and touch no other chain.

Utility workload: a GHZ state on the chain (H on b, CNOT b->a, CNOT b->c)
that waits 10 us before it is measured, the way circuits wait for
mid-circuit measurement and feed-forward. Fidelity
F = (P000 + P111)/2 + sum_k (-1)^k <P(k pi/3)>/12 from one Z circuit and six
parity circuits. Every chain runs three ways: plain, with each qubit's
measured delta_omega undone by a virtual rz(-delta_omega t) after the wait,
and with an XX echo during the wait. Both predictions use

  F = G R [ (1 + prod p + prod (1 - p))/4 + prod exp(-t/T2) cos(phi)/2 ],
  p = exp(-t/T1), G = prod (1 - e_cz) prod (1 - e_sx)^n, R = prod (1 - e_ro)

with the backend's CZ, sx and readout errors. "Before" uses the reported T1
and T2 and phi = 0, which is all the calibration has. "After" uses Canary's
live T1, T2* (T2 with the echo) and phi = sum delta_omega t (0 when
corrected). Before picks the chain the calibration ranks first, run plain;
after picks the chain and treatment Canary ranks first. All twelve
configurations are measured, so both picks are scored against the best one.
The predictions are saved before the job is sent.

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
  utility       measured fidelity of before's and after's picks, regret to
                the best configuration, and Spearman correlation between
                each prediction and the measured fidelities
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
from scipy.stats import binom, chi2 as chi2_dist, kstest, spearmanr

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_SRC = SCRIPT_DIR.parent / "src"
if REPO_SRC.is_dir():
    sys.path.insert(0, str(REPO_SRC))

import quantum_canary as canary
from quantum_canary import inversion as inv
from quantum_canary.runner import DEFAULT_PROBE_BLOCKS, ibm_sampler, probe_groups, pub_executions

DATA_DIR = SCRIPT_DIR / "data" / "hardware"
TAGS = ("A", "B", "C", "D")
UTILITY_TAGS = ("A", "C")
N_CHAINS = 4
RO_MAX, SX_MAX, CZ_MAX = 0.05, 1e-3, 0.02
T_WAIT = 10e-6
N_SX = 4
GHZ_N = 3
UTIL_SHOTS = 1000
TREATMENTS = ("plain", "corrected", "dd")
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
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
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


def edge_error(backend, a: int, b: int) -> Optional[float]:
    for name in ("cz", "ecr", "cx"):
        if name not in backend.target.operation_names:
            continue
        for pair in ((a, b), (b, a)):
            props = backend.target[name].get(pair)
            if props is not None and props.error is not None:
                return float(props.error)
    return None


def qubit_ok(props, q: int) -> bool:
    try:
        t1, t2 = props.t1(q), props.t2(q)
        ro, sx = props.readout_error(q), props.gate_error("sx", q)
    except Exception:
        return False
    try:
        ok = props.is_qubit_operational(q)
    except Exception:
        ok = True
    return bool(ok and t1 and t2 and t1 > 0 and t2 > 0 and t2 <= 2 * t1
                and ro <= RO_MAX and sx <= SX_MAX)


def predict(t1s, t2s, phase, czs, sxs, ros, extra_sx: int = 0) -> float:
    p = np.exp(-T_WAIT / np.asarray(t1s, dtype=float))
    pop = (1.0 + np.prod(p) + np.prod(1.0 - p)) / 4.0
    coh = np.prod(np.exp(-T_WAIT / np.asarray(t2s, dtype=float))) * np.cos(phase) / 2.0
    g = np.prod(1.0 - np.asarray(czs)) * np.prod((1.0 - np.asarray(sxs)) ** (N_SX + extra_sx))
    return float(g * np.prod(1.0 - np.asarray(ros)) * (pop + coh))


def select_chains(backend, n_chains: int) -> List[List[int]]:
    props = backend.properties()
    nbr = neighbours(backend)
    good = {q for q in range(backend.num_qubits) if qubit_ok(props, q)}
    scored = []
    for b in sorted(good):
        ends = sorted(n for n in nbr[b] if n in good)
        for i, a in enumerate(ends):
            for c in ends[i + 1:]:
                if c in nbr[a]:
                    continue
                czs = [edge_error(backend, a, b), edge_error(backend, b, c)]
                if any(e is None or e > CZ_MAX for e in czs):
                    continue
                chain = [a, b, c]
                f = predict([props.t1(q) for q in chain], [props.t2(q) for q in chain], 0.0, czs,
                            [props.gate_error("sx", q) for q in chain],
                            [props.readout_error(q) for q in chain])
                scored.append((-f, chain))
    scored.sort()
    chosen: List[List[int]] = []
    used: set = set()
    for _, chain in scored:
        if used & set(chain) or any(nbr[q] & used for q in chain):
            continue
        chosen.append(chain)
        used |= set(chain)
        if len(chosen) == n_chains:
            return chosen
    raise RuntimeError(f"only {len(chosen)} chains pass the rule, need {n_chains}")


def reported(backend, qubits: List[int], at: str, chains: Optional[List[List[int]]] = None):
    props = backend.properties()
    t_run = datetime.fromisoformat(at)
    out: Dict[str, Any] = {"last_update_date": props.last_update_date, "qubits": {}, "edges": {}}
    for q in qubits:
        row: Dict[str, Any] = {}
        for key in ("T1", "T2", "readout_error", "prob_meas1_prep0", "prob_meas0_prep1"):
            try:
                value, date = props.qubit_property(q, key)
                row[key] = float(value)
                row[f"{key}_date"] = date
                if date is not None:
                    dd = date if date.tzinfo else date.replace(tzinfo=timezone.utc)
                    row[f"{key}_age_h"] = (t_run - dd).total_seconds() / 3600.0
            except Exception:
                row[key] = None
        try:
            value, date = props.gate_property("sx", [q], "gate_error")
            row["sx_error"], row["sx_error_date"] = float(value), date
        except Exception:
            row["sx_error"] = None
        out["qubits"][str(q)] = row
    for chain in chains or []:
        for a, b in zip(chain, chain[1:]):
            out["edges"][f"{a}-{b}"] = edge_error(backend, a, b)
    return out


class Recorder:
    """A sampler that keeps the job, the PUBs and the raw result.

    On a fake backend the PUBs are simulated in one call per set of qubits:
    Aer's noisy simulation of a fake backend lets delay noise leak between
    circuits on different qubits within one call, so a whole job in one
    call would not be simulated faithfully. Real backends get one ordinary
    job. With limit_s, IBM's own estimate of the job (usage_estimation) is
    read right after submission and the job is cancelled before it runs if
    the estimate is above limit_s, unless force."""

    def __init__(self, backend, fake: bool = False, limit_s: Optional[float] = None,
                 force: bool = False):
        if fake:
            from qiskit_aer.primitives import SamplerV2 as AerSampler
            self.sampler = AerSampler.from_backend(backend)
        else:
            self.sampler = ibm_sampler(backend)
        self.options = getattr(self.sampler, "options", None)
        self.fake = fake
        self.limit_s = limit_s
        self.force = force
        self.ibm_estimate_s: Optional[float] = None
        self.pubs: List[Any] = []
        self.job = None
        self.raw = None

    def run(self, pubs):
        from qiskit.primitives import PrimitiveResult
        self.pubs = list(pubs)
        rec = self
        if self.fake:
            plain, bound = [], {}
            for c, values, shots in self.pubs:
                if values is not None and c.parameters:
                    if id(c) not in bound:
                        bound[id(c)] = c.assign_parameters({p: 0.0 for p in c.parameters})
                    plain.append((bound[id(c)], None, shots * len(values)))
                else:
                    plain.append((c, values, shots))
            by_set: Dict[frozenset, List[int]] = {}
            for i, p in enumerate(plain):
                c = p[0]
                key = frozenset(c.find_bit(q).index for inst in c.data for q in inst.qubits)
                by_set.setdefault(key, []).append(i)
            results: List[Any] = [None] * len(plain)
            for idx in by_set.values():
                for i, r in zip(idx, self.sampler.run([plain[i] for i in idx]).result()):
                    results[i] = r
            self.raw = PrimitiveResult(results, metadata={})
        else:
            self.job = self.sampler.run(self.pubs)
            try:
                self.ibm_estimate_s = self.job.usage_estimation.get("quantum_seconds")
            except Exception:
                self.ibm_estimate_s = None
            print(f"job {self.job.job_id()}: IBM estimates "
                  f"{self.ibm_estimate_s if self.ibm_estimate_s is not None else 'nothing'} s")
            if (self.limit_s is not None and self.ibm_estimate_s is not None
                    and self.ibm_estimate_s > self.limit_s and not self.force):
                self.job.cancel()
                raise SystemExit(f"cancelled before running: IBM estimates {self.ibm_estimate_s} s, "
                                 f"more than the {self.limit_s:.0f} s left; pass --force to run anyway")

        class _Job:
            def job_id(self):
                return f"local-{len(rec.pubs)}" if rec.fake else rec.job.job_id()

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


def recorder(args, backend, service) -> "Recorder":
    """A Recorder whose jobs are cancelled before running if IBM's own
    estimate is above the QPU time IBM reports as left (minus RESERVE_S)."""
    left = remaining_seconds(service)
    if service is not None:
        print(f"QPU time left according to IBM: {left if left is not None else 'unknown'} s")
    return Recorder(backend, bool(args.fake), None if left is None else left - RESERVE_S, args.force)


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


def split_half(report) -> Dict[str, Any]:
    """Refit each qubit from its even and its odd shot blocks separately. A
    qubit that cannot be split gets an error entry; the run is still saved."""
    out: Dict[str, Any] = {}
    for qr in report.qubits:
        try:
            meta = inv.build_probe_circuits(qr.profile)[1]
            k = len(family_shots(meta, report.shots))
            entries = report.block_counts[qr.qubit]
            merge = lambda es: inv.merge_block_counts([e["counts"] for e in es],
                                                      [e["circuit"] for e in es], k)
            full = invert(merge(entries), meta, qr.profile, report.shots, qr.qubit)
            same = all(np.isclose(full[key], getattr(qr.result, key), rtol=1e-9, equal_nan=True)
                       for key in ("T1_s", "T2_s", "delta_omega", "epsilon_sx"))
            bounds = {"t1": (0, meta["n_t1"]),
                      "ramsey": (meta["n_t1"], meta["n_t1"] + meta["n_ramsey"]),
                      "gate": (meta["n_t1"] + meta["n_ramsey"], k - meta["n_echo"]),
                      "echo": (k - meta["n_echo"], k)}
            halves = []
            for h in (0, 1):
                c = merge([e for e in entries if e["block"] % 2 == h])
                hs = {}
                for fam, (lo, hi) in bounds.items():
                    tot = {sum(x.values()) for x in c[lo:hi]}
                    if len(tot) != 1:
                        raise RuntimeError(f"uneven half-shots in the {fam} family")
                    hs[fam] = tot.pop()
                halves.append(invert(c, meta, qr.profile, hs, qr.qubit))
            out[str(qr.qubit)] = {"halves": halves, "refit_matches_run": bool(same)}
            if not same:
                print(f"warning: refit of qubit {qr.qubit} differs from the run")
        except Exception as exc:
            out[str(qr.qubit)] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"warning: no split-half for qubit {qr.qubit}: {exc}")
    return out


def run_canary(args, backend, service, qubits: List[int], tag: str) -> Dict[str, Any]:
    rec = recorder(args, backend, service)
    t_submit = now()
    rep_before = reported(backend, qubits, t_submit)
    report = canary.run_ibm(backend, qubits, probe_blocks=args.probe_blocks, sampler=rec)
    t_done = now()
    payload = {"tag": tag, "backend": backend.name, "fake": bool(args.fake),
               "qubits": qubits, "groups": report.groups, "t_submit": t_submit, "t_done": t_done,
               "job_id": report.job_id, "qpu_seconds": qpu_seconds(rec.job, bool(args.fake)),
               "ibm_usage_estimate_s": rec.ibm_estimate_s,
               "n_pubs": len(rec.pubs), "shots_executed": sum(pub_executions(p) for p in rec.pubs),
               "shots_per_qubit": report.shots_per_qubit,
               "probe_blocks": args.probe_blocks, "sampler_options": str(rec.options),
               "execution_spans": execution_spans(rec.raw), "reported": rep_before,
               "report": report.to_dict(), "split_half": split_half(report),
               "block_counts": {str(q): v for q, v in report.block_counts.items()}}
    report.print()
    return payload


def qe_experiment(backend, qubits: List[int], priors: Dict[int, Dict[str, float]]):
    from qiskit_experiments.framework import BatchExperiment, ParallelExperiment
    from qiskit_experiments.library import T1, T2Hahn, T2Ramsey
    grid = lambda span, n: [float(x) for x in np.linspace(0.0, span, n + 1)[1:]]
    osc = {q: QE_RAMSEY_OSC / priors[q]["T2"] for q in qubits}
    make = {
        "t1": lambda q: T1((q,), delays=grid(QE_T1_SPAN * priors[q]["T1"], QE_N_DELAYS["t1"])),
        "ramsey": lambda q: T2Ramsey((q,), delays=grid(QE_RAMSEY_SPAN * priors[q]["T2"],
                                                       QE_N_DELAYS["ramsey"]), osc_freq=osc[q]),
        "hahn": lambda q: T2Hahn((q,), delays=grid(QE_HAHN_SPAN * priors[q]["T2"],
                                                   QE_N_DELAYS["hahn"]), num_echoes=1),
    }
    groups = probe_groups(backend, qubits)
    pars = []
    for fam, fn in make.items():
        for g in groups:
            exps = [fn(q) for q in g]
            for e in exps:
                e.set_transpile_options(optimization_level=0)
                e.analysis.set_options(plot=False)
            pars.append(ParallelExperiment(exps, backend=backend, flatten_results=True))
    return BatchExperiment(pars, backend=backend, flatten_results=True), osc, groups


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


def run_qe(args, backend, service, qubits: List[int]):
    from qiskit_experiments.framework import ExperimentData
    priors = {}
    for q in qubits:
        with contextlib.redirect_stdout(io.StringIO()):
            p = inv.BackendProfile.from_ibm_backend(backend, qubit=q)
        priors[q] = {"T1": float(p.T1_prior_s), "T2": float(p.T2_prior_s)}
    batch, osc, groups = qe_experiment(backend, qubits, priors)
    circuits = batch._transpiled_circuits()
    shots_executed = QE_SHOTS * len(circuits)
    rec = recorder(args, backend, service)
    t_submit = now()
    rep_before = reported(backend, qubits, t_submit)
    job = rec.run([(c, None, QE_SHOTS) for c in circuits])
    print(f"qe job {job.job_id()}: {len(circuits)} circuits x {QE_SHOTS} shots, groups {groups}")
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
    return {"backend": backend.name, "fake": bool(args.fake), "qubits": qubits, "groups": groups,
            "t_submit": t_submit, "t_done": t_done, "job_id": job.job_id(),
            "qpu_seconds": qpu_seconds(rec.job, bool(args.fake)), "n_pubs": len(circuits),
            "ibm_usage_estimate_s": rec.ibm_estimate_s, "shots_executed": shots_executed,
            "shots_per_qubit": QE_SHOTS * sum(QE_N_DELAYS.values()),
            "priors": {str(q): priors[q] for q in qubits},
            "osc_freq_hz": {str(q): osc[q] for q in qubits},
            "execution_spans": execution_spans(result), "reported": rep_before,
            "results": res, "counts": counts,
            "circuit_metadata": [c.metadata for c in circuits]}


def dt_count(backend, seconds: float) -> int:
    return int(round(seconds / backend.dt))


def ghz_circuits(backend, chain: List[int], treatment: str, live: Dict[str, Dict[str, Any]]):
    """The Z circuit and the six parity circuits of one chain and treatment."""
    from qiskit import QuantumCircuit, transpile
    n_wait = dt_count(backend, T_WAIT)
    out = []
    for basis in ["z"] + list(range(2 * GHZ_N)):
        qc = QuantumCircuit(GHZ_N, GHZ_N, name=f"ghz_{'-'.join(map(str, chain))}_{treatment}_{basis}")
        qc.h(1)
        qc.cx(1, 0)
        qc.cx(1, 2)
        qc.barrier()
        for j, q in enumerate(chain):
            if treatment == "dd":
                n_x = dt_count(backend, backend.target["x"][(q,)].duration)
                d1 = (n_wait - 2 * n_x) // 4
                qc.delay(d1, j, unit="dt")
                qc.x(j)
                qc.delay(n_wait - 2 * n_x - 2 * d1, j, unit="dt")
                qc.x(j)
                qc.delay(d1, j, unit="dt")
            else:
                qc.delay(n_wait, j, unit="dt")
            if treatment == "corrected":
                qc.rz(-live[str(q)]["delta_omega"] * n_wait * backend.dt, j)
        qc.barrier()
        if basis != "z":
            phi = basis * np.pi / GHZ_N
            for j in range(GHZ_N):
                qc.rz(-phi, j)
                qc.h(j)
        qc.measure(range(GHZ_N), range(GHZ_N))
        out.append(transpile(qc, backend=backend, initial_layout=chain, optimization_level=0))
    return out


def ghz_fidelity(counts_list: List[Dict[str, int]]) -> Dict[str, float]:
    z = counts_list[0]
    n = sum(z.values())
    pop = (z.get("0" * GHZ_N, 0) + z.get("1" * GHZ_N, 0)) / n
    parities, var = [], pop * (1 - pop) / n / 4
    for k, c in enumerate(counts_list[1:]):
        m = sum(c.values())
        par = sum(v * (-1) ** key.count("1") for key, v in c.items()) / m
        parities.append(par)
        var += (1 - par ** 2) / m / (4 * GHZ_N) ** 2
    coh = sum((-1) ** k * p for k, p in enumerate(parities)) / (4 * GHZ_N)
    return {"fidelity": pop / 2 + coh, "sigma": float(np.sqrt(var)),
            "populations": pop, "coherence": coh, "parities": parities}


def predictions(chains, rep, live) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {"before": {}, "after": {}}
    for chain in chains:
        key = "-".join(map(str, chain))
        r = [rep["qubits"][str(q)] for q in chain]
        lv = [live[str(q)] for q in chain]
        czs = [rep["edges"][f"{a}-{b}"] for a, b in zip(chain, chain[1:])]
        sxs, ros = [x["sx_error"] for x in r], [x["readout_error"] for x in r]
        phase = sum(x["delta_omega"] for x in lv) * T_WAIT
        t1r, t2r = [x["T1"] for x in r], [x["T2"] for x in r]
        t1l = [x["T1_s"] for x in lv]
        out["before"][key] = {"plain": predict(t1r, t2r, 0.0, czs, sxs, ros),
                              "corrected": predict(t1r, t2r, 0.0, czs, sxs, ros),
                              "dd": predict(t1r, t2r, 0.0, czs, sxs, ros, extra_sx=4)}
        out["after"][key] = {"plain": predict(t1l, [x["T2_ramsey_s"] for x in lv], phase, czs, sxs, ros),
                             "corrected": predict(t1l, [x["T2_ramsey_s"] for x in lv], 0.0, czs, sxs, ros),
                             "dd": predict(t1l, [x["T2_s"] for x in lv], 0.0, czs, sxs, ros, extra_sx=4)}
    return out


def cmd_utility(args, backend, service, d):
    plan = load(d / "plan.json")
    run = load(d / f"canary_{args.tag}.json")
    if plan is None or run is None:
        raise SystemExit(f"needs plan.json and canary_{args.tag}.json")
    path = d / f"utility_{args.tag}.json"
    if path.exists() and not args.force:
        raise SystemExit(f"{path.name} exists; pass --force to overwrite")
    chains = plan["chains"]
    live = {str(q["qubit"]): q["result"] for q in run["report"]["qubits"]}
    t_pred = now()
    rep = reported(backend, plan["qubits"], t_pred, chains)
    pred = predictions(chains, rep, live)
    before = max(pred["before"], key=lambda c: pred["before"][c]["plain"])
    after = max(((c, t) for c in pred["after"] for t in TREATMENTS),
                key=lambda ct: pred["after"][ct[0]][ct[1]])
    picks = {"before": [before, "plain"], "before_dd": [before, "dd"], "after": list(after)}
    save(d / f"utility_{args.tag}_predictions.json",
         {"created": t_pred, "canary_run": args.tag, "canary_job": run["job_id"],
          "t_wait_s": T_WAIT, "reported": rep, "predictions": pred, "picks": picks})
    configs, circuits = [], []
    for chain in chains:
        for t in TREATMENTS:
            cs = ghz_circuits(backend, chain, t, live)
            configs.append(("-".join(map(str, chain)), t, len(cs)))
            circuits.extend(cs)
    rec = recorder(args, backend, service)
    t_submit = now()
    job = rec.run([(c, None, UTIL_SHOTS) for c in circuits])
    print(f"utility job {job.job_id()}: {len(circuits)} circuits x {UTIL_SHOTS} shots")
    result = job.result()
    counts = [result[i].join_data().get_counts() for i in range(len(circuits))]
    measured: Dict[str, Dict[str, Any]] = {}
    i = 0
    for key, t, n in configs:
        measured.setdefault(key, {})[t] = ghz_fidelity(counts[i:i + n])
        i += n
    summary = utility_summary(pred, measured, picks)
    save(path, {"tag": args.tag, "backend": backend.name, "fake": bool(args.fake),
                "chains": chains, "t_wait_s": T_WAIT, "t_predictions": t_pred,
                "t_submit": t_submit, "t_done": now(), "job_id": job.job_id(),
                "qpu_seconds": qpu_seconds(rec.job, bool(args.fake)), "n_pubs": len(circuits),
                "ibm_usage_estimate_s": rec.ibm_estimate_s, "shots_executed": UTIL_SHOTS * len(circuits),
                "execution_spans": execution_spans(result), "predictions": pred,
                "picks": picks, "measured": measured, "summary": summary, "counts": counts})


def utility_summary(pred, measured, picks) -> Dict[str, Any]:
    keys = [(c, t) for c in measured for t in TREATMENTS]
    f = {k: measured[k[0]][k[1]]["fidelity"] for k in keys}
    s = {k: measured[k[0]][k[1]]["sigma"] for k in keys}
    best = max(keys, key=lambda k: f[k])
    print(f"\n   {'chain':<14}" + "".join(f"{t:>26}" for t in TREATMENTS))
    for c in measured:
        print(f"   {c:<14}" + "".join(
            f"   {f[(c, t)]:.3f}+-{s[(c, t)]:.3f} ({pred['before'][c][t]:.2f}/{pred['after'][c][t]:.2f})"
            for t in TREATMENTS))
    print("   (measured fidelity, then predicted before/after)")
    out: Dict[str, Any] = {"best": list(best), "best_fidelity": f[best], "picks": {}}
    for name, (c, t) in picks.items():
        out["picks"][name] = {"config": [c, t], "fidelity": f[(c, t)], "sigma": s[(c, t)],
                              "regret": f[best] - f[(c, t)]}
        print(f"   {name:<10} {c} {t:<9} F = {f[(c, t)]:.3f} +- {s[(c, t)]:.3f}   "
              f"regret {f[best] - f[(c, t)]:.3f}")
    for side in ("before", "after"):
        rho = spearmanr([pred[side][k[0]][k[1]] for k in keys], [f[k] for k in keys]).correlation
        out[f"spearman_{side}"] = float(rho)
        print(f"   {side} prediction vs measured, Spearman {rho:+.2f} over {len(keys)} configurations")
    gain = f[tuple(picks["after"])] - f[tuple(picks["before"])]
    out["after_minus_before"] = gain
    out["after_minus_before_sigma"] = float(np.hypot(s[tuple(picks["after"])], s[tuple(picks["before"])]))
    print(f"   after - before = {gain:+.3f} +- {out['after_minus_before_sigma']:.3f}")
    return out


def cmd_plan(args, backend, service, d):
    chains = select_chains(backend, args.n_chains)
    qubits = [q for chain in chains for q in chain]
    t = now()
    save(d / "plan.json", {"backend": backend.name, "fake": bool(args.fake), "created": t,
                           "chains": chains, "qubits": qubits, "groups": probe_groups(backend, qubits),
                           "rule": {"readout_error_max": RO_MAX, "sx_error_max": SX_MAX,
                                    "cz_error_max": CZ_MAX, "t2_le_2t1": True,
                                    "rank": "calibration-predicted utility fidelity",
                                    "chains": "disjoint, never adjacent"},
                           "reported": reported(backend, qubits, t, chains)})
    print(f"chains: {chains}   probe groups: {probe_groups(backend, qubits)}")


def plan_of(d: pathlib.Path) -> Dict[str, Any]:
    plan = load(d / "plan.json")
    if plan is None:
        raise SystemExit("run plan first")
    return plan


def cmd_smoke(args, backend, service, d):
    plan = plan_of(d)
    payload = run_canary(args, backend, service, plan["qubits"][:1], "smoke")
    save(d / "smoke.json", payload)
    print(f"\nsmoke: IBM measured {payload['qpu_seconds'] or 'n/a'} s for one qubit "
          f"({payload['shots_executed']:,} shots); QPU time left according to IBM: "
          f"{remaining_seconds(service) or 'unknown'} s")


def cmd_canary(args, backend, service, d):
    plan = plan_of(d)
    path = d / f"canary_{args.tag}.json"
    if path.exists() and not args.force:
        raise SystemExit(f"{path.name} exists; pass --force to overwrite")
    save(path, run_canary(args, backend, service, plan["qubits"], args.tag))


def cmd_qe(args, backend, service, d):
    plan = plan_of(d)
    path = d / "qe.json"
    if path.exists() and not args.force:
        raise SystemExit(f"{path.name} exists; pass --force to overwrite")
    save(path, run_qe(args, backend, service, plan["qubits"]))


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
        for q, entry in r["split_half"].items():
            if "halves" not in entry:
                print(f"   run {t} qubit {q}: no split-half ({entry.get('error')})")
                continue
            h0, h1 = entry["halves"]
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
            ratio = res["T2_ramsey_s"] / res["T2_s"] if res["T2_s"] else float("nan")
            unrep[f"{t}_{q}"] = {"delta_omega_khz": res["delta_omega"] / (2 * np.pi * 1e3),
                                 "delta_omega_sigma_khz": res["delta_omega_sigma"] / (2 * np.pi * 1e3),
                                 "t2star_over_t2": ratio,
                                 "phase_per_10us_rad": abs(res["delta_omega"]) * 10e-6,
                                 "health": v["health"]}
        print(f"   run {t}: |dw|/2pi " + ", ".join(
            f"q{q} {abs(unrep[f'{t}_{q}']['delta_omega_khz']):.1f}" for q in qubits) + " kHz")
        print("          T2*/T2   " + ", ".join(
            f"q{q} {unrep[f'{t}_{q}']['t2star_over_t2']:.2f}" for q in qubits))
        print("          health   " + ", ".join(f"q{q} {unrep[f'{t}_{q}']['health']}" for q in qubits))
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

    print("\n8. Before/after: trusting the calibration vs using Canary")
    util = {}
    for t in UTILITY_TAGS:
        u = load(d / f"utility_{t}.json")
        if not u:
            continue
        print(f"   after Canary run {t}:")
        util[t] = utility_summary(u["predictions"], u["measured"], u["picks"])
    out["utility"] = util

    print("\n9. Cost")
    cost = {}
    files = {**{f"Canary {t}": r for t, r in runs.items()}, "QE": qe,
             **{f"utility {t}": load(d / f"utility_{t}.json") for t in UTILITY_TAGS}}
    for name, r in files.items():
        if not r:
            continue
        cost[name] = {"qpu_seconds": r.get("qpu_seconds"), "shots_executed": r.get("shots_executed"),
                      "shots_per_qubit": r.get("shots_per_qubit")}
        print(f"   {name:<10} {r.get('qpu_seconds') or 'n/a'} QPU s, {r.get('shots_executed'):,} shots "
              f"executed" + (f", {r['shots_per_qubit']:,} per qubit" if r.get("shots_per_qubit") else ""))
    out["cost"] = cost
    save(d / "analysis.json", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("plan", "smoke", "canary", "utility", "qe", "usage", "analyze"))
    ap.add_argument("--backend", default="ibm_kingston")
    ap.add_argument("--instance", default=None)
    ap.add_argument("--fake", default=None, help="local fake backend, e.g. fake_fez")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-chains", type=int, default=N_CHAINS)
    ap.add_argument("--tag", choices=TAGS)
    ap.add_argument("--probe-blocks", type=int, default=DEFAULT_PROBE_BLOCKS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.command in ("canary", "utility") and not args.tag:
        ap.error(f"{args.command} needs --tag")
    if args.probe_blocks < 2 or args.probe_blocks % 2:
        ap.error("--probe-blocks must be even for the split-half test")
    d = out_dir(args)
    if args.command == "analyze":
        cmd_analyze(args, None, None, d)
        return
    backend, service = get_backend(args)
    {"plan": cmd_plan, "smoke": cmd_smoke, "canary": cmd_canary, "utility": cmd_utility,
     "qe": cmd_qe, "usage": cmd_usage}[args.command](args, backend, service, d)


if __name__ == "__main__":
    main()
