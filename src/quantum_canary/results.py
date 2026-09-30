"""Counts utilities, signal statistics and result containers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .resources import Resource, to_key


def normalize_counts(counts: Mapping[str, float]) -> dict[str, float]:
    """Return frequencies (summing to one) with register spaces removed."""
    out: dict[str, float] = {}
    for key, value in counts.items():
        v = float(value)
        if v <= 0:
            continue
        k = str(key).replace(" ", "")
        out[k] = out.get(k, 0.0) + v
    total = sum(out.values())
    if total <= 0:
        raise ValueError("empty counts")
    return {k: v / total for k, v in out.items()}


def merge_frequencies(parts: Sequence[tuple[Mapping[str, float], float]]) -> dict[str, float]:
    """Shot-weighted merge of (frequencies, shots) pairs into frequencies."""
    out: dict[str, float] = {}
    total = 0.0
    for freqs, shots in parts:
        total += shots
        for k, v in freqs.items():
            out[k] = out.get(k, 0.0) + v * shots
    if total <= 0:
        raise ValueError("no shots to merge")
    return {k: v / total for k, v in out.items()}


def unflip_frequencies(freqs: Mapping[str, float], flips: Iterable[int], num_clbits: int) -> dict[str, float]:
    """Undo a measurement twirl: flip the given clbits (Qiskit little-endian keys) in every outcome."""
    flips = tuple(int(c) for c in flips)
    width = int(num_clbits)
    out: dict[str, float] = {}
    for key, p in freqs.items():
        bits = list(key.zfill(width))
        for c in flips:
            idx = len(bits) - 1 - c
            bits[idx] = "1" if bits[idx] == "0" else "0"
        k = "".join(bits)
        out[k] = out.get(k, 0.0) + p
    return out


def marginal_frequencies(freqs: Mapping[str, float], clbits: Sequence[int]) -> dict[str, float]:
    """Marginal distribution of the given clbits, relabelled so that new clbit j is old clbit ``clbits[j]``."""
    out: dict[str, float] = {}
    for key, p in freqs.items():
        width = len(key)
        k = "".join(key[width - 1 - c] if c < width else "0" for c in reversed(clbits))
        out[k] = out.get(k, 0.0) + p
    return out


def expectation(freqs: Mapping[str, float], clbits: Sequence[int]) -> float:
    """Expectation of the Z-string on the given clbits (Qiskit little-endian keys)."""
    value = 0.0
    for key, p in freqs.items():
        parity = 0
        for c in clbits:
            idx = len(key) - 1 - c
            if idx >= 0 and key[idx] == "1":
                parity ^= 1
        value += -p if parity else p
    return value


def z_covariance(freqs: Mapping[str, float], a: Sequence[int], b: Sequence[int], shots: float) -> float:
    """Covariance of the sample means of Z-strings a and b estimated from the same shots."""
    sym = tuple(sorted(set(a) ^ set(b)))
    joint = expectation(freqs, sym) if sym else 1.0
    return (joint - expectation(freqs, a) * expectation(freqs, b)) / max(float(shots), 1.0)


def floored_log_signal(m: float, shots: float, model_sd: float) -> tuple[float, float]:
    """Log-magnitude of a known-positive expectation, floored at one standard error, and its variance."""
    n = max(float(shots), 1.0)
    se = math.sqrt(1.0 / n)
    m_eff = max(m, se)
    one_minus = max(1.0 - m_eff * m_eff, 1.0 / n)
    return math.log(m_eff), one_minus / (n * m_eff * m_eff) + model_sd**2


def _upper_gamma_q(a: float, x: float) -> float:
    """Regularised upper incomplete gamma function Q(a, x)."""
    if x <= 0.0:
        return 1.0
    log_pref = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        term = total = 1.0 / a
        ap = a
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(log_pref))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return min(1.0, math.exp(log_pref) * h)


def chi2_sf(x: float, dof: int) -> float:
    """Survival function of the chi-squared distribution."""
    if dof <= 0:
        raise ValueError("dof must be positive")
    return _upper_gamma_q(0.5 * dof, 0.5 * max(float(x), 0.0))


@dataclass
class AssignmentRun:
    """Algorithm results for one logical-to-ion assignment (frequencies are readout-untwirled)."""

    assignment: tuple[int, ...]
    frequencies: dict[str, float]
    shots: int
    trusted: bool = True
    predicted_error: float | None = None

    @property
    def counts(self) -> dict[str, float]:
        """Frequencies scaled to the number of shots."""
        return {k: v * self.shots for k, v in self.frequencies.items()}


@dataclass
class ProbeRun:
    """Results of one calibration circuit ('cb', 'spam' or 'phase'); frequencies are readout-untwirled."""

    ions: tuple[int, ...]
    depth: int
    frequencies: dict[str, float]
    shots: int
    kind: str = "cb"
    bases: tuple[str, ...] = ()
    setting: str = ""
    angle: float = 0.0


@dataclass
class Tier2Fit:
    """Consistency fit of one observable across assignments: m_a = mitigated + slope * Q_a.

    Q_a is the damping predicted for assignment a from calibrated (Tier-1) rates with every lightcone gate fully
    exposed. The intercept is a layout-diversity zero-noise extrapolation (cf. FF-ZNE and CLP-ZNE); it is only as
    good as the linear model and the spread of Q, so check ``mitigated_stderr`` and ``p_value``.
    """

    qubits: tuple[int, ...]
    predicted: tuple[float, ...]
    observed: tuple[float, ...]
    variances: tuple[float, ...]
    mitigated: float | None = None
    mitigated_stderr: float | None = None
    slope: float | None = None
    chi2: float | None = None
    dof: int = 0
    p_value: float | None = None


@dataclass
class BatchPlan:
    """Preview of one batch: what would be submitted and what it would cost in gate-shots."""

    circuits: list[Any]
    shots_per_circuit: int
    algorithm_circuits: int
    probe_circuits: int
    probe_pairs: tuple[tuple[int, int], ...]
    assignments: tuple[tuple[tuple[int, ...], ...], ...]
    single_qubit_gate_shots: int
    two_qubit_gate_shots: int
    spam_circuits: int = 0
    phase_circuits: int = 0
    phase_pairs: tuple[tuple[int, int], ...] = ()

    @property
    def calibration_circuits(self) -> int:
        """Number of Tier-1 calibration circuits (cycle-benchmarking, SPAM and phase probes)."""
        return self.probe_circuits + self.spam_circuits + self.phase_circuits

    @property
    def total_shots(self) -> int:
        """Total shots across all circuits in the job."""
        return self.shots_per_circuit * len(self.circuits)

    @property
    def probe_shot_fraction(self) -> float:
        """Fraction of the job's shots spent on calibration circuits."""
        n = len(self.circuits)
        return self.calibration_circuits / n if n else 0.0


@dataclass
class BatchResult:
    """Outcome of one Canary batch for one circuit."""

    name: str
    runs: list[AssignmentRun]
    recommended: tuple[int, ...]
    probes: list[ProbeRun] = field(default_factory=list)
    flagged_ions: tuple[int, ...] = ()
    degraded: tuple[Resource, ...] = ()
    timestamp: float = 0.0
    job_ids: tuple[str, ...] = ()
    tier2: dict[tuple[int, ...], Tier2Fit] = field(default_factory=dict)

    def _selected(self, trusted_only: bool) -> list[AssignmentRun]:
        runs = [r for r in self.runs if r.trusted] if trusted_only else list(self.runs)
        return runs if runs else list(self.runs)

    def frequencies(self, trusted_only: bool = True) -> dict[str, float]:
        """Shot-weighted frequencies over trusted runs (all runs if none are trusted)."""
        return merge_frequencies([(r.frequencies, r.shots) for r in self._selected(trusted_only)])

    def counts(self, trusted_only: bool = True) -> dict[str, float]:
        """Aggregated counts over trusted runs (all runs if none are trusted)."""
        total = sum(r.shots for r in self._selected(trusted_only))
        return {k: v * total for k, v in self.frequencies(trusted_only).items()}

    @property
    def shots(self) -> int:
        """Total algorithm shots in this batch."""
        return sum(r.shots for r in self.runs)

    def mitigated_expectation(self, qubits: Sequence[int]) -> tuple[float, float] | None:
        """Layout-diversity extrapolated expectation and its standard error, if the fit was possible."""
        fit = self.tier2.get(tuple(sorted(set(int(q) for q in qubits))))
        if fit is None or fit.mitigated is None or fit.mitigated_stderr is None:
            return None
        return fit.mitigated, fit.mitigated_stderr

    @property
    def consistency_p(self) -> float | None:
        """Bonferroni-combined p-value that calibrated rates explain the spread across assignments."""
        ps = [f.p_value for f in self.tier2.values() if f.p_value is not None]
        return min(1.0, len(ps) * min(ps)) if ps else None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "job_ids": list(self.job_ids),
            "recommended": list(self.recommended),
            "flagged_ions": list(self.flagged_ions),
            "degraded": [to_key(r) for r in self.degraded],
            "runs": [
                {
                    "assignment": list(r.assignment),
                    "shots": r.shots,
                    "trusted": r.trusted,
                    "predicted_error": r.predicted_error,
                    "frequencies": r.frequencies,
                }
                for r in self.runs
            ],
            "probes": [
                {
                    "kind": p.kind,
                    "ions": list(p.ions),
                    "depth": p.depth,
                    "bases": list(p.bases),
                    "setting": p.setting,
                    "angle": p.angle,
                    "shots": p.shots,
                    "frequencies": p.frequencies,
                }
                for p in self.probes
            ],
            "tier2": [
                {
                    "qubits": list(f.qubits),
                    "predicted": list(f.predicted),
                    "observed": list(f.observed),
                    "mitigated": f.mitigated,
                    "mitigated_stderr": f.mitigated_stderr,
                    "slope": f.slope,
                    "chi2": f.chi2,
                    "dof": f.dof,
                    "p_value": f.p_value,
                }
                for f in self.tier2.values()
            ],
        }
