"""The simple interface: check qubits, compare checks, check between jobs.

    import quantum_canary as canary

    r = canary.check(backend, [13, 14])        # one short job -> Report
    r.print()                                  # table: numbers, fit, health, DD advice
    r.save("before.json")
    r[13].t1, r[13].t2_star, r[13].health      # plain attributes per qubit

    d = canary.compare(r, canary.load("after.json"))
    d.print(); d.changed                       # qubits that moved beyond their error bars

    mon = canary.Monitor(backend, [13, 14])    # between jobs, as often as you like
    mon.check()                                # first call is the baseline; repeat between jobs
    mon.changed; mon.print(); mon.save("checks/")

Every check is its own short job, never inserted into yours. Qubits are probed
in at most two parallel groups (no two neighbours in one circuit).
"""
from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
from scipy.stats import norm

from . import runner, workflow
from .online import OnlineInversionResult, _green_chi2_threshold

__all__ = ["QubitResult", "Report", "check", "compare", "load", "simulate", "Monitor"]

PROBES = (("t1", "T1"), ("ramsey", "Ramsey"), ("gate", "gate"), ("echo", "echo"))
T2_DISAGREE = OnlineInversionResult.yellow_t2_disagree_threshold
DRIFT_Z = 2.0
USAGE_WAIT_S = 60.0
EXPLAIN = """\
Fit     chi2/dof of each probe against its limit. On a qubit that follows the model
        each probe goes over its limit 1.25% of the time; when it does, that probe's
        numbers may be wrong.
Health  GREEN   every fit converged and passed, and T2* is within 20% of T2
        YELLOW  a fit failed its limit, or T2* differs from T2 by more than 20%
                (slow frequency noise, which dynamical decoupling refocuses)
        RED     a fit did not converge; re-run the check
        Health describes the measurement, not how good the qubit is: read the
        numbers for that."""
AT_BOUND_NOTE = """\
        "at bound": a fit stopped at its allowed limit (e.g. T2 = 2*T1), so that
        value's error bar is not reliable."""


def fit_over_limit(result) -> List[str]:
    """Probes whose chi2/dof is over its limit, as 'Ramsey 5.1 > 2.7'."""
    out = []
    for key, label in PROBES:
        value, limit = getattr(result, f"{key}_chi2_dof"), _green_chi2_threshold(key)
        if np.isfinite(value) and value > limit:
            out.append(f"{label} {value:.1f} > {limit:.1f}")
    return out


def bound_hits(q) -> List[str]:
    """Parameters whose fit stopped at a bound of the engine's fit (the
    covariance, and so the error bar, is not valid there): T1 in
    [T1_min_s, T1_max_s], T2 (echo) in [T2_min_s, 2*T1], T2* in
    [T2_min_s, min(20*T2_prior, 2*T1)], eps in [0, eps_max]."""
    r, c = q.result, q.profile.constants

    def hit(value, bound):
        return bool(np.isfinite(value) and abs(value - bound) <= 1e-6 * abs(bound))

    out = []
    if hit(r.T1_s, c["T1_min_s"]) or hit(r.T1_s, c["T1_max_s"]):
        out.append("T1")
    if np.isfinite(r.T1_s):
        if hit(r.T2_echo_s, c["T2_min_s"]) or hit(r.T2_echo_s, 2.0 * r.T1_s):
            out.append("T2")
        if hit(r.T2_ramsey_s, c["T2_min_s"]) or hit(
                r.T2_ramsey_s, min(20.0 * q.profile.T2_prior_s, 2.0 * r.T1_s)):
            out.append("T2*")
    if np.isfinite(r.epsilon_sx) and (r.epsilon_sx <= 1e-9 or hit(r.epsilon_sx, c["eps_max"])):
        out.append("eps")
    return out


def health_reasons(health: str, result) -> List[str]:
    """Why a qubit is YELLOW or RED (empty for GREEN)."""
    if health == "RED":
        failed = [label for sigma, label in (
            (result.T1_sigma_s, "T1 fit failed"),
            (result.T2_echo_sigma_s, "echo fit failed"),
            (result.T2_ramsey_sigma_s, "Ramsey fit failed"),
            (result.delta_omega_sigma, "Δω not measurable (no Ramsey contrast)"),
            (result.epsilon_sx_sigma, "gate fit failed")) if not np.isfinite(sigma)]
        return failed or ["fit did not converge"]
    reasons = ["fit"] if fit_over_limit(result) else []
    if result.T2_s > 0 and np.isfinite(result.T2_ramsey_s) and np.isfinite(result.T2_echo_s):
        if abs(result.T2_ramsey_s - result.T2_echo_s) / result.T2_s > T2_DISAGREE:
            reasons.append(f"T2*/T2 {result.T2_ramsey_s / result.T2_s:.2f}")
    return reasons


@dataclass(frozen=True)
class QubitResult:
    """One qubit of a check. Times in seconds, delta_omega in rad/s."""
    qubit: int
    t1: float
    t1_sigma: float
    t2: float
    t2_sigma: float
    t2_star: float
    t2_star_sigma: float
    delta_omega: float
    delta_omega_sigma: float
    eps: float
    eps_sigma: float
    chi2: Dict[str, float]
    fit: List[str]
    health: str
    health_reason: str
    dd_advice: List[str]
    truth: Optional[Dict[str, float]] = None
    frame_advice: List[str] = field(default_factory=list)
    at_bound: List[str] = field(default_factory=list)

    @classmethod
    def from_qubit_report(cls, q) -> "QubitResult":
        r = q.result
        return cls(qubit=int(q.qubit), t1=r.T1_s, t1_sigma=r.T1_sigma_s, t2=r.T2_s,
                   t2_sigma=r.T2_sigma_s, t2_star=r.T2_ramsey_s,
                   t2_star_sigma=r.T2_ramsey_sigma_s, delta_omega=r.delta_omega,
                   delta_omega_sigma=r.delta_omega_sigma, eps=r.epsilon_sx,
                   eps_sigma=r.epsilon_sx_sigma,
                   chi2={key: getattr(r, f"{key}_chi2_dof") for key, _ in PROBES},
                   fit=fit_over_limit(r), health=q.health,
                   health_reason=", ".join(health_reasons(q.health, r)),
                   dd_advice=workflow._dd_reasons(q), truth=q.truth,
                   frame_advice=workflow._frame_reasons(q), at_bound=bound_hits(q))


class Report:
    """The result of one check: report[q] is qubit q's QubitResult."""

    def __init__(self, raw, ibm_usage_s: Optional[float] = None):
        self.raw = raw
        self.ibm_usage_s = ibm_usage_s
        self.qubits: List[QubitResult] = [QubitResult.from_qubit_report(q) for q in raw.qubits]

    @property
    def backend(self) -> str:
        return self.raw.backend

    @property
    def job_id(self) -> Optional[str]:
        return self.raw.job_id

    @property
    def timestamp(self) -> str:
        return self.raw.timestamp

    def __getitem__(self, qubit: int) -> QubitResult:
        for q in self.qubits:
            if q.qubit == int(qubit):
                return q
        raise KeyError(f"qubit {qubit} is not in this check")

    def __iter__(self) -> Iterator[QubitResult]:
        return iter(self.qubits)

    def __len__(self) -> int:
        return len(self.qubits)

    def render(self) -> str:
        arch = self.raw.architecture
        scale, unit = (1e6, "us") if arch == "superconducting" else (1e3, "ms")
        dw_scale, dw_unit = (2e3 * np.pi, "kHz") if arch == "superconducting" else (2.0 * np.pi, "Hz")

        def t(value, sigma):
            if not np.isfinite(value):
                return "n/a"
            return (f"{value * scale:.3g} ± {sigma * scale:.2g} {unit}" if np.isfinite(sigma)
                    else f"{value * scale:.3g} {unit}")

        head = (f"{'qubit':<6}{'T1':<17}{'T2':<17}{'T2*':<17}{'dw/2pi (' + dw_unit + ')':<17}"
                f"{'eps':<20}{'fit':<26}health")
        lines = [head, "-" * len(head)]
        for q in self.qubits:
            dw = (f"{q.delta_omega / dw_scale:+.3g} ± {q.delta_omega_sigma / dw_scale:.2g}"
                  if np.isfinite(q.delta_omega) else "n/a")
            eps = f"{q.eps:.2e} ± {q.eps_sigma:.0e}" if np.isfinite(q.eps) else "n/a"
            health = q.health + (f" ({q.health_reason})" if q.health_reason else "")
            lines.append(f"{q.qubit:<6}{t(q.t1, q.t1_sigma):<17}{t(q.t2, q.t2_sigma):<17}"
                         f"{t(q.t2_star, q.t2_star_sigma):<17}{dw:<17}{eps:<20}"
                         f"{'; '.join(q.fit + [p + ' at bound' for p in q.at_bound]) or 'ok':<26}"
                         f"{health}")
        lines.append("")
        advised = [q for q in self.qubits if q.dd_advice]
        lines += ([f"dynamical decoupling advised on qubit {q.qubit}: {'; '.join(q.dd_advice)}"
                   for q in advised] or ["dynamical decoupling: not needed on these qubits"])
        lines += [f"frequency offset on qubit {q.qubit}: {'; '.join(q.frame_advice)}; correct it "
                  "with a virtual-Z frame update or frequency recalibration (DD only if it drifts)"
                  for q in self.qubits if q.frame_advice]
        lines += ["", EXPLAIN] + ([AT_BOUND_NOTE] if any(q.at_bound for q in self.qubits) else []) + [""]
        usage = (f", IBM measured usage {self.ibm_usage_s:.1f} s" if self.ibm_usage_s is not None
                 else "")
        try:
            shots = f", {self.raw.shots_per_qubit:,} shots per qubit"
        except Exception:
            shots = ""
        lines.append(f"backend {self.backend}, job {self.job_id or 'local'}{shots}"
                     + (f", probed in groups {self.raw.groups}" if self.raw.groups else "") + usage)
        return "\n".join(lines)

    def print(self) -> None:
        workflow._print(self.render())

    def to_dict(self) -> Dict[str, Any]:
        return {"backend": self.backend, "job_id": self.job_id, "ibm_usage_s": self.ibm_usage_s,
                "health": {q.qubit: q.health for q in self.qubits},
                "health_reason": {q.qubit: q.health_reason for q in self.qubits},
                "fit_over_limit": {q.qubit: q.fit for q in self.qubits},
                "dynamical_decoupling": {q.qubit: q.dd_advice for q in self.qubits},
                "frequency_offset": {q.qubit: q.frame_advice for q in self.qubits},
                "at_bound": {q.qubit: q.at_bound for q in self.qubits},
                "report": self.raw.to_dict()}

    def save(self, path) -> None:
        pathlib.Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str),
                                      encoding="utf-8")


def _ibm_usage(backend, job_id: Optional[str]) -> Optional[float]:
    """IBM's measured usage of a finished job, if IBM reports it within USAGE_WAIT_S."""
    service = getattr(backend, "service", None)
    if service is None or not job_id:
        return None
    deadline = time.time() + USAGE_WAIT_S
    while time.time() < deadline:
        try:
            usage = service.job(job_id).usage()
        except Exception:
            return None
        if usage:
            return float(usage)
        time.sleep(5)
    return None


def check(backend, qubits: Sequence[int]) -> Report:
    """Check `qubits` of an IBM backend (or a qiskit-ibm-runtime fake backend) in
    one short job and return the Report."""
    raw = runner.run_ibm(backend, qubits)
    return Report(raw, ibm_usage_s=_ibm_usage(backend, raw.job_id))


def simulate(n_qubits: int = 3, architecture: str = "superconducting",
             regime: str = "nisq", seed: Optional[int] = None) -> Report:
    """Check simulated qubits, no backend needed; report[q].truth holds the true values."""
    return Report(runner.run_simulation(architecture, n_qubits=n_qubits, regime=regime, seed=seed))


def load(path) -> Report:
    """A Report from a JSON file written by Report.save() or `quantum-canary check --json`
    (CanaryReport.to_dict() files work too)."""
    from .inversion import ARCH_DEFAULTS, BackendProfile, InversionResult
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    rep = data.get("report", data)
    arch = rep.get("architecture", "superconducting")
    names = {f.name for f in fields(InversionResult)}
    qubits = []
    for q in rep["qubits"]:
        result = {k: v for k, v in q["result"].items() if k in names}
        result["calibration_source"] = None
        constants = dict(ARCH_DEFAULTS[arch])
        constants.update({k: q[k] for k in ("p0_given_1", "p1_given_0", "gate_time_ns")
                          if q.get(k) is not None})
        constants.update(T1_s=q["T1_prior_s"], T2_s=q["T2_prior_s"])
        if q.get("eps_prior") is not None:
            constants["eps_typical"] = q["eps_prior"]
        profile = BackendProfile(architecture=arch, T1_prior_s=q["T1_prior_s"],
                                 T2_prior_s=q["T2_prior_s"], dt_ns=q.get("dt_ns"),
                                 backend_name=rep.get("backend", ""), custom_arch=constants,
                                 prior_confidence=q.get("prior_confidence") or "arch_default")
        qubits.append(runner.QubitReport(qubit=int(q["qubit"]), result=InversionResult(**result),
                                         profile=profile, health=q["health"],
                                         truth=q.get("truth")))
    raw = runner.CanaryReport(mode=rep.get("mode", "hardware"), backend=rep.get("backend", ""),
                              architecture=arch,
                              shots=rep.get("shots", {}), probe_blocks=rep.get("probe_blocks", 1),
                              circuits_per_qubit=rep.get("circuits_per_qubit", 0), qubits=qubits,
                              timestamp=rep.get("timestamp", ""), job_id=rep.get("job_id"),
                              groups=rep.get("groups"),
                              block_counts=({int(q): e for q, e in rep["block_counts"].items()}
                                            if rep.get("block_counts") else None))
    return Report(raw, ibm_usage_s=data.get("ibm_usage_s"))


def _raw(x):
    return x.raw if isinstance(x, Report) else x


def compare(before, after, z_threshold: float = DRIFT_Z):
    """What changed between two checks (Reports, or the advanced CanaryReport and
    CheckReport): a two-sample z-test per parameter with a Holm correction over each
    qubit's four parameters. Returns a DriftReport (.changed, .print())."""
    return workflow.compare(_raw(before), _raw(after), z_threshold=z_threshold)


def repeated_threshold(n_comparisons: int, z: float = DRIFT_Z) -> float:
    """The |z| limit for each of n comparisons against one baseline so that the chance
    of any false alarm over all of them stays that of a single comparison at |z| > z."""
    alpha = 2.0 * norm.sf(z)
    return float(norm.isf(alpha / (2.0 * max(1, int(n_comparisons)))))


def drift_table(drift) -> str:
    names = (("T1_s", "T1"), ("T2_s", "T2"), ("delta_omega", "dw"), ("epsilon_sx", "eps"))
    head = (f"{'qubit':<6}{'before':<9}{'after':<9}" + "".join(f"{'z ' + n:>9}" for _, n in names)
            + "   changed")
    lines = [head, "-" * len(head)]
    for q, z in drift.z_scores.items():
        moved = [n for key, n in names if drift.drifted[q][key]]
        if drift.health_after[q] == "RED":
            moved.append("RED after")
        lines.append(f"{q:<6}{drift.health_before[q]:<9}{drift.health_after[q]:<9}"
                     + "".join(f"{z[key]:>+9.2f}" if np.isfinite(z[key]) else f"{'n/a':>9}"
                               for key, _ in names)
                     + "   " + (", ".join(moved) or "-"))
    lines += ["", f"changed = |z| > {drift.z_threshold:.2f} after a Holm correction over each "
                  "qubit's four parameters; z uses both checks' error bars",
              "stable: nothing changed beyond the error bars" if drift.stable else
              f"changed qubits: {drift.drifted_qubits}; results that depend on them may not hold"]
    return "\n".join(lines)


def timeline(baseline, later: Sequence[Any], labels: Optional[Sequence[str]] = None) -> str:
    """Every later check against the baseline, with the repeated-check threshold."""
    z = repeated_threshold(len(later))
    labels = list(labels) if labels else [f"check {i}" for i in range(1, len(later) + 1)]
    out = [f"{len(later)} check{'s' if len(later) != 1 else ''} against the baseline: changed = |z| > {z:.2f}, so that the "
           f"chance of any false alarm over all of them is that of one check at |z| > {DRIFT_Z:g}"]
    for label, report in zip(labels, later):
        out += ["", f"{label}", drift_table(compare(baseline, report, z_threshold=z))]
    return "\n".join(out)


@dataclass
class Monitor:
    """Checks of the same qubits between jobs, as many as you like. The first
    check() is the baseline; every later one is compared with it, with the |z| limit
    raised for the number of checks so far (repeated_threshold)."""
    backend: Any
    qubits: List[int]
    history: List[Report] = field(default_factory=list)

    def check(self) -> Report:
        report = check(self.backend, self.qubits)
        self.history.append(report)
        return report

    @property
    def baseline(self) -> Optional[Report]:
        return self.history[0] if self.history else None

    @property
    def threshold(self) -> float:
        return repeated_threshold(len(self.history) - 1)

    @property
    def changed(self) -> List[int]:
        """Qubits that moved since the baseline in the latest check."""
        if len(self.history) < 2:
            return []
        return compare(self.baseline, self.history[-1], z_threshold=self.threshold).drifted_qubits

    def render(self) -> str:
        if not self.history:
            return "no checks yet"
        if len(self.history) == 1:
            return "baseline only: call check() again between jobs"
        labels = [f"check {i} ({r.timestamp[:19]})" for i, r in enumerate(self.history[1:], 1)]
        return timeline(self.baseline, self.history[1:], labels)

    def print(self) -> None:
        workflow._print(self.render())

    def save(self, folder) -> None:
        folder = pathlib.Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for i, report in enumerate(self.history):
            report.save(folder / f"check_{i:03d}.json")
