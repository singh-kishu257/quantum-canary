"""The job workflow: check qubits before a job, watch them during it, compare after.

    before = canary.check(backend, qubits=[13, 14, 2], spares=[3, 4])
    run    = canary.run_monitored(backend, my_circuits, before, interval=20)
    after  = canary.check(backend, qubits=before.layout)
    drift  = canary.compare(before, after)

Everything here is a wrapper around the existing pieces: run_ibm() for the
probe, CanaryInterleaver for interleaved rounds, and the same two-sample
z-test with Holm correction that CanaryInterleaver.drift_report() uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import norm as _norm

from . import online
from .runner import (DEFAULT_SHOTS, CanaryReport, QubitReport, _captured,
                     _counts_from_pub, _shots_list, flatten_result, run_ibm,
                     sample)

__all__ = ["CheckReport", "MonitoredRun", "DriftReport",
           "check", "run_monitored", "compare",
           "DD_T2_RATIO", "DD_PHASE_RAD", "DD_SEQUENCE"]

# Dynamical-decoupling heuristics, applied per qubit (both are simple, stated
# rules of thumb, not fitted thresholds):
#   T2*/T2 < DD_T2_RATIO: Ramsey loses coherence that the echo recovers, i.e.
#       slow frequency noise that a decoupling sequence refocuses.
#   |delta_omega| * T2* > DD_PHASE_RAD: the qubit's frequency offset turns
#       more than this many radians of phase in one coherence time while
#       idle, a coherent error decoupling cancels.
DD_T2_RATIO = 0.8
DD_PHASE_RAD = 1.0
# Sequence suggested when DD is recommended (a Qiskit Runtime option value;
# XY4 tolerates pulse imperfections better than XX).
DD_SEQUENCE = "XY4"
_HEALTH_RANK = {"GREEN": 0, "YELLOW": 1, "RED": 2}


def _dd_reasons(q: QubitReport) -> List[str]:
    r = q.result
    reasons = []
    if (np.isfinite(r.T2_ramsey_s) and np.isfinite(r.T2_s) and r.T2_s > 0
            and r.T2_ramsey_s / r.T2_s < DD_T2_RATIO):
        reasons.append(f"T2*/T2 = {r.T2_ramsey_s / r.T2_s:.2f} < {DD_T2_RATIO}")
    if np.isfinite(r.delta_omega) and np.isfinite(r.T2_ramsey_s):
        phase = abs(r.delta_omega) * r.T2_ramsey_s
        if phase > DD_PHASE_RAD:
            reasons.append(f"|Δω|·T2* = {phase:.1f} rad > {DD_PHASE_RAD:g}")
    return reasons


@dataclass
class CheckReport:
    """Result of check(): per-qubit health, the layout to use and the
    dynamical-decoupling recommendation."""
    report: CanaryReport
    requested: List[int]
    layout: List[int]
    swaps: Dict[int, int]
    unresolved: List[int]
    dd_reasons: Dict[int, List[str]]

    @property
    def qubits(self) -> Dict[int, QubitReport]:
        return {q.qubit: q for q in self.report.qubits}

    @property
    def health(self) -> Dict[int, str]:
        return {q: self.qubits[q].health for q in self.layout}

    @property
    def dd_recommended(self) -> bool:
        return any(self.dd_reasons.get(q) for q in self.layout)

    @property
    def dd_options(self) -> Dict[str, Any]:
        """Qiskit Runtime dynamical_decoupling options for the job."""
        if self.dd_recommended:
            return {"enable": True, "sequence_type": DD_SEQUENCE}
        return {"enable": False}

    def monitor(self, interval: int = 20) -> online.CanaryInterleaver:
        """A CanaryInterleaver on the layout, built from the checked profiles."""
        return online.CanaryInterleaver(
            qubit_ids=list(self.layout),
            profiles={q: self.qubits[q].profile for q in self.layout},
            interval_n=int(interval))

    def to_dict(self) -> Dict[str, Any]:
        return {"requested": self.requested, "layout": self.layout, "swaps": self.swaps,
                "unresolved": self.unresolved, "dd_recommended": self.dd_recommended,
                "dd_options": self.dd_options, "dd_reasons": self.dd_reasons,
                "report": self.report.to_dict()}

    def render(self, color: Optional[bool] = None) -> str:
        from .display import render_check
        return render_check(self, color=color)

    def print(self, color: Optional[bool] = None, file=None) -> None:
        _print(self.render(color=color), file)


def check(backend, qubits: Sequence[int], spares: Sequence[int] = (),
          shots: Optional[Dict[str, int]] = None, sampler=None) -> CheckReport:
    """Probe `qubits` (and any `spares`) before a job, in one Canary run.

    Returns the health of each qubit, a layout in which every RED qubit is
    replaced by the healthiest spare (GREEN before YELLOW, then the longest
    T2), and whether dynamical decoupling is recommended. Spares should be
    chosen so they fit your circuit's connectivity; check() does not route.
    """
    requested = [int(q) for q in qubits]
    spare_list = [int(q) for q in spares if int(q) not in requested]
    report = run_ibm(backend, requested + spare_list, shots=shots, sampler=sampler)
    by_q = {q.qubit: q for q in report.qubits}

    pool = sorted((q for q in spare_list if by_q[q].health != "RED"),
                  key=lambda q: (_HEALTH_RANK[by_q[q].health],
                                 -(by_q[q].result.T2_s if np.isfinite(by_q[q].result.T2_s) else 0.0)))
    layout, swaps, unresolved = [], {}, []
    for q in requested:
        if by_q[q].health == "RED" and pool:
            new = pool.pop(0)
            swaps[q] = new
            layout.append(new)
        else:
            if by_q[q].health == "RED":
                unresolved.append(q)
            layout.append(q)
    dd = {q.qubit: _dd_reasons(q) for q in report.qubits}
    return CheckReport(report=report, requested=requested, layout=layout, swaps=swaps,
                       unresolved=unresolved, dd_reasons=dd)


@dataclass
class MonitoredRun:
    """Result of run_monitored(): your circuits' results, untouched, plus the
    health of the monitored qubits in every Canary round."""
    results: List[Any]
    rounds: List[online.OnlineInversionResult]
    monitor: online.CanaryInterleaver
    layout: List[int]
    interval: int
    job_id: Optional[str] = None
    execution_segments: Optional[List[Dict[str, Any]]] = None

    @property
    def execution_order(self) -> str:
        """Whether the job ran in the submitted order, from IBM's execution
        spans: "held", "violated", "unresolved" or "unavailable" (see
        _execution_order)."""
        return _execution_order(self.execution_segments)

    @property
    def flagged_rounds(self) -> List[int]:
        """Rounds in which any monitored qubit was not GREEN."""
        return [r.round_index for r in self.rounds
                if any(h != "GREEN" for h in r.health_summary().values())]

    def circuits_in_round(self, round_index: int) -> range:
        """Indices (into your circuit list) submitted just before a round."""
        start = round_index * self.interval
        return range(start, min(start + self.interval, len(self.results)))

    def render(self, color: Optional[bool] = None) -> str:
        from .display import render_monitored
        return render_monitored(self, color=color)

    def print(self, color: Optional[bool] = None, file=None) -> None:
        _print(self.render(color=color), file)


def run_monitored(backend, circuits: Sequence[Any], check_report: CheckReport,
                  interval: int = 20, shots: int = 1024, sampler=None) -> MonitoredRun:
    """Run your circuits with a Canary round after every `interval` of them,
    on check_report.layout, all in one job.

    `circuits` must already be transpiled for `backend` (ISA circuits, as
    SamplerV2 requires), laid out on check_report.layout. Their results are
    returned untouched, in order, in MonitoredRun.results.

    The rounds are interleaved in the submitted list. Qiskit Runtime does
    not guarantee that a job's circuits run in that order, so the job's
    execution spans are kept in MonitoredRun.execution_segments, and
    MonitoredRun.execution_order says whether the order held.

    Leave dynamical decoupling off here: Qiskit Runtime applies it to every
    circuit in the job, which would put pulses inside Canary's probe delays
    and distort its measurements. Use check_report.dd_options for runs
    without Canary rounds.
    """
    import warnings
    from qiskit import transpile

    dd = getattr(getattr(sampler, "options", None), "dynamical_decoupling", None)
    if getattr(dd, "enable", False) is True:
        warnings.warn("dynamical decoupling is enabled on this sampler: it also acts on "
                      "Canary's probe delays, so the monitored rounds will be distorted",
                      stacklevel=2)

    monitor = check_report.monitor(interval)
    (job_circuits, round_map), _ = _captured(monitor.prepare, list(circuits))

    algo_pos = {i for rmap in round_map for i in rmap["algo_indices"]}
    circs: List[Any] = [None] * len(job_circuits)
    shots_of: List[int] = [0] * len(job_circuits)
    cache: Dict[int, Any] = {}
    for rmap in round_map:
        for qid, meta in monitor.canary_metadata.items():
            start = rmap["canary_start_idx"] + meta["circuit_start_idx"]
            for k, sh in enumerate(_shots_list(meta, DEFAULT_SHOTS)):
                qc = job_circuits[start + k]
                if id(qc) not in cache:
                    cache[id(qc)] = transpile(qc, backend=backend, initial_layout=[qid],
                                              optimization_level=0)
                circs[start + k], shots_of[start + k] = cache[id(qc)], sh
    for i in algo_pos:
        circs[i], shots_of[i] = job_circuits[i], int(shots)
    pub_results, job_id = sample(backend, circs, shots_of, sampler)

    job_results = [flatten_result(pub_results[i]) if i in algo_pos
                   else _counts_from_pub(pub_results[i], circs[i], shots_of[i])
                   for i in range(len(circs))]
    (algo_per_round, rounds), _ = _captured(monitor.process, job_results, round_map)
    results = [res for per_round in algo_per_round for res in per_round]
    return MonitoredRun(results=results, rounds=rounds, monitor=monitor,
                        layout=list(check_report.layout), interval=int(interval),
                        job_id=job_id,
                        execution_segments=_execution_segments(pub_results, round_map))


def _execution_segments(pub_results, round_map) -> Optional[List[Dict[str, Any]]]:
    """The submitted job in segments (a block of your circuits, then a Canary
    round, and so on), each with the window [start, stop] (UTC) covering
    every execution span that holds any of its shots.

    Qiskit Runtime guarantees that a span's data was taken inside the span,
    not that the span is tight, so a segment's window always contains its
    real run time but may be wider. start/stop are None if no span names
    the segment's PUBs. Returns None when the result has no execution spans
    (local simulators, older qiskit-ibm-runtime)."""
    metadata = getattr(pub_results, "metadata", None)
    execution = metadata.get("execution") if isinstance(metadata, dict) else None
    spans = execution.get("execution_spans") if isinstance(execution, dict) else None
    if spans is None or len(spans) == 0:
        return None
    span_pubs = [(sp, set(sp.pub_idxs)) for sp in spans]
    segments: List[Dict[str, Any]] = []
    for rmap in round_map:
        first = rmap["canary_start_idx"]
        canary_idx = list(range(first, first + rmap["n_canary_circuits"]))
        for kind, idx in (("circuits", list(rmap["algo_indices"])), ("canary", canary_idx)):
            if not idx:
                continue
            hits = [sp for sp, pubs in span_pubs if not pubs.isdisjoint(idx)]
            segments.append({"kind": kind, "round": rmap["round_index"],
                             "first_pub": idx[0], "last_pub": idx[-1],
                             "start": min(sp.start for sp in hits) if hits else None,
                             "stop": max(sp.stop for sp in hits) if hits else None})
    return segments


def _execution_order(segments: Optional[List[Dict[str, Any]]]) -> str:
    """Whether the segments ran in the submitted order.

    "held"        every segment's window ends no later than the next one starts
    "violated"    some later segment's window ends before an earlier one's
                  starts, so it certainly ran first
    "unresolved"  windows overlap (or a segment has none), so the spans
                  cannot tell
    "unavailable" no execution spans were returned
    """
    if not segments:
        return "unavailable"
    timed = [s for s in segments if s["start"] is not None and s["stop"] is not None]
    for i, a in enumerate(timed):
        for b in timed[i + 1:]:
            if b["stop"] < a["start"]:
                return "violated"
    if len(timed) == len(segments) and all(
            a["stop"] <= b["start"] for a, b in zip(segments, segments[1:])):
        return "held"
    return "unresolved"


_DRIFT_PARAMS = (("T1_s", "T1_sigma_s", "t1_chi2_dof"),
                 ("T2_s", "T2_sigma_s", None),
                 ("delta_omega", "delta_omega_sigma", "ramsey_chi2_dof"),
                 ("epsilon_sx", "epsilon_sx_sigma", "gate_chi2_dof"))


@dataclass
class DriftReport:
    """Result of compare(): per qubit, z-scores between the two checks and
    which parameters drifted."""
    z_scores: Dict[int, Dict[str, float]]
    drifted: Dict[int, Dict[str, bool]]
    health_before: Dict[int, str]
    health_after: Dict[int, str]
    z_threshold: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def drifted_qubits(self) -> List[int]:
        return [q for q, d in self.drifted.items()
                if any(d.values()) or self.health_after[q] == "RED"]

    @property
    def stable(self) -> bool:
        return not self.drifted_qubits

    @property
    def changed(self) -> List[int]:
        """Qubits that moved beyond their error bars (or are RED after)."""
        return self.drifted_qubits

    def render(self, color: Optional[bool] = None) -> str:
        """The plain table quantum-canary compare prints (color is ignored)."""
        from .api import drift_table
        return drift_table(self)

    def print(self, color: Optional[bool] = None, file=None) -> None:
        _print(self.render(color=color), file)


def _qubit_map(x) -> Dict[int, QubitReport]:
    if isinstance(x, CheckReport):
        return {q: x.qubits[q] for q in x.layout}
    if isinstance(x, CanaryReport):
        return {q.qubit: q for q in x.qubits}
    raise TypeError(f"expected a CheckReport or CanaryReport, got {type(x).__name__}")


def compare(before, after, z_threshold: float = 2.0) -> DriftReport:
    """Compare two checks of the same qubits (e.g. before and after a job).

    Uses the same test as CanaryInterleaver.drift_report(): a two-sample
    z-score per parameter with chi2-inflated sigmas (T2 as reported), and a
    Holm correction across each qubit's four parameters at the family level
    2*(1 - Phi(z_threshold)).
    """
    b, a = _qubit_map(before), _qubit_map(after)
    common = [q for q in b if q in a]
    if not common:
        raise ValueError("the two checks share no qubits")
    alpha = 2.0 * _norm.sf(z_threshold)
    z_all, drifted = {}, {}
    for q in common:
        rb, ra = b[q].result, a[q].result
        z = {}
        for name, sig, chi in _DRIFT_PARAMS:
            if chi is None:
                z[name] = online.chi2_scaled_z(getattr(ra, name) - getattr(rb, name),
                                               getattr(ra, sig), getattr(rb, sig))
            else:
                z[name] = online.chi2_scaled_z(getattr(ra, name) - getattr(rb, name),
                                               getattr(ra, sig), getattr(rb, sig),
                                               getattr(ra, chi), getattr(rb, chi))
        p = {k: float(2.0 * _norm.sf(abs(v))) if np.isfinite(v) else float("nan")
             for k, v in z.items()}
        holm = online._holm_adjust(p)
        z_all[q] = z
        drifted[q] = {k: bool(np.isfinite(v) and v < alpha) for k, v in holm.items()}
    return DriftReport(z_scores=z_all, drifted=drifted,
                       health_before={q: b[q].health for q in common},
                       health_after={q: a[q].health for q in common},
                       z_threshold=float(z_threshold))


def _print(text: str, file=None) -> None:
    import sys
    stream = sys.stdout if file is None else file
    try:
        stream.write(text + "\n")
    except UnicodeEncodeError:
        enc = getattr(stream, "encoding", None) or "ascii"
        stream.write((text + "\n").encode(enc, errors="replace").decode(enc))
    stream.flush()
