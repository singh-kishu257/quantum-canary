"""Counts utilities, signal statistics and result containers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

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


def log_signal(
    m: float, shots: float, model_sd: float, min_z: float, min_abs: float
) -> tuple[float, float] | None:
    """Log-magnitude of an expectation and its variance, or None if too noisy."""
    n = max(float(shots), 1.0)
    one_minus = max(1.0 - m * m, 1.0 / n)
    se = math.sqrt(one_minus / n)
    if abs(m) < max(min_abs, min_z * se):
        return None
    return math.log(abs(m)), one_minus / (n * m * m) + model_sd**2


def floored_log_signal(m: float, shots: float, model_sd: float) -> tuple[float, float]:
    """Log-magnitude of a known-positive expectation, floored at one standard error."""
    n = max(float(shots), 1.0)
    se = math.sqrt(1.0 / n)
    m_eff = max(m, se)
    one_minus = max(1.0 - m_eff * m_eff, 1.0 / n)
    return math.log(m_eff), one_minus / (n * m_eff * m_eff) + model_sd**2


@dataclass
class AssignmentRun:
    """Algorithm results for one logical-to-ion assignment."""

    assignment: tuple[int, ...]
    frequencies: dict[str, float]
    shots: int
    trusted: bool = True

    @property
    def counts(self) -> dict[str, float]:
        """Frequencies scaled to the number of shots."""
        return {k: v * self.shots for k, v in self.frequencies.items()}


@dataclass
class ProbeRun:
    """Echo-probe results for one ion pair at one depth."""

    ions: tuple[int, int]
    depth: int
    frequencies: dict[str, float]
    shots: int


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

    def frequencies(self, trusted_only: bool = True) -> dict[str, float]:
        """Shot-weighted frequencies over trusted runs (all runs if none are trusted)."""
        runs = [r for r in self.runs if r.trusted] if trusted_only else list(self.runs)
        if not runs:
            runs = list(self.runs)
        return merge_frequencies([(r.frequencies, r.shots) for r in runs])

    def counts(self, trusted_only: bool = True) -> dict[str, float]:
        """Aggregated counts over trusted runs (all runs if none are trusted)."""
        runs = [r for r in self.runs if r.trusted] if trusted_only else list(self.runs)
        if not runs:
            runs = list(self.runs)
        total = sum(r.shots for r in runs)
        return {k: v * total for k, v in self.frequencies(trusted_only).items()}

    @property
    def shots(self) -> int:
        """Total algorithm shots in this batch."""
        return sum(r.shots for r in self.runs)

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
                    "frequencies": r.frequencies,
                }
                for r in self.runs
            ],
            "probes": [
                {"ions": list(p.ions), "depth": p.depth, "shots": p.shots, "frequencies": p.frequencies}
                for p in self.probes
            ],
        }
