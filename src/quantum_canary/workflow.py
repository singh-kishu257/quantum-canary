"""Per-qubit advice and the comparison of two checks, used by quantum_canary.api.

    compare(before, after)   two-sample z-test per parameter, Holm-corrected
                             across each qubit's four parameters -> DriftReport
    _dd_reasons, _frame_reasons
                             the dynamical-decoupling and frequency-offset advice
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import norm as _norm

from . import online
from .runner import CanaryReport, QubitReport

__all__ = ["DriftReport", "compare", "DD_T2_RATIO", "FRAME_PHASE_RAD"]

# Per-qubit advice (simple, stated rules of thumb, not fitted thresholds):
#   T2*/T2 < DD_T2_RATIO: Ramsey loses coherence that the echo recovers, i.e.
#       slow frequency noise that a decoupling sequence refocuses -> DD.
#   |delta_omega| * T2* > FRAME_PHASE_RAD: the qubit's measured frequency
#       offset turns more than this many radians of phase in one coherence
#       time while idle. A known, steady offset is removed by a virtual-Z
#       frame update (or a frequency recalibration); DD is the remedy only if
#       it drifts. Reported separately and never turns DD on.
DD_T2_RATIO = 0.8
FRAME_PHASE_RAD = 1.0


def _dd_reasons(q: QubitReport) -> List[str]:
    r = q.result
    reasons = []
    if (np.isfinite(r.T2_ramsey_s) and np.isfinite(r.T2_s) and r.T2_s > 0
            and r.T2_ramsey_s / r.T2_s < DD_T2_RATIO):
        reasons.append(f"T2*/T2 = {r.T2_ramsey_s / r.T2_s:.2f} < {DD_T2_RATIO}")
    return reasons


def _frame_reasons(q: QubitReport) -> List[str]:
    r = q.result
    if np.isfinite(r.delta_omega) and np.isfinite(r.T2_ramsey_s):
        phase = abs(r.delta_omega) * r.T2_ramsey_s
        if phase > FRAME_PHASE_RAD:
            return [f"|Δω|·T2* = {phase:.1f} rad > {FRAME_PHASE_RAD:g}"]
    return []


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
    if isinstance(x, CanaryReport):
        return {q.qubit: q for q in x.qubits}
    raise TypeError(f"expected a CanaryReport, got {type(x).__name__}")


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
