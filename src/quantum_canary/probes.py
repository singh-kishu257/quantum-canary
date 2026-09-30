"""Tier-1 calibration circuits, their estimators and sampling models, plus the Z2 symmetry sentinel.

* Cycle-benchmarking (CB) probes of an ion pair: prepare a product eigenstate of X, Y or Z on each ion, apply
  ``depth`` ZZ(1/4 turn) gates (a multiple of 4, so the ideal circuit is the identity), rotate back and measure.
  Each circuit yields three Pauli expectations (P_a, P_b and P_a P_b, ideal +1). The per-gate decay is taken from
  the ratio of two depths, which cancels state preparation and measurement (cycle benchmarking, Erhard et al.,
  arXiv:1902.08543), and the three decays are weighted 1/5, 1/5, 3/5 so that, with bases drawn uniformly, the estimate is
  unbiased for the decay averaged over all 15 non-identity Paulis of the dressed cycle.
* SPAM probes: prepare |0> on a set of ions and measure under complementary measurement twirls.
* Phase probes: prepare |++>, apply n ZZ(theta) gates with n * theta = 1/4 (mod 1) and measure <X_a> and
  <Y_a Z_b>; their angle gives the accumulated ZZ phase, and hence the coherent over-rotation per gate, with sign.
  Phase probes use commuting-frame twirls, which leave the ZZ phase intact.

Every calibrated estimate is recorded as an :class:`Observation` that carries its sampling model rather than a
variance computed from its own outcome. The variance of a CB or SPAM estimate grows with the damping it measures;
a variance evaluated at the observed value would give lucky low draws small variances and large weights, biasing
pooled estimates downwards (severely for SPAM, where a batch sees only a few flipped outcomes). Consumers evaluate
the variance where it is needed instead: the health filter at its one-step prediction, and the verdict tests at
their null thresholds.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

from .program import Gate, LogicalProgram
from .results import expectation

BASES: tuple[str, ...] = ("x", "y", "z")
BASIS_PAIRS: tuple[tuple[str, str], ...] = tuple((a, b) for a in BASES for b in BASES)
PAULI_WEIGHTS = np.array([0.2, 0.2, 0.6])
_WEIGHTS: tuple[float, float, float] = (0.2, 0.2, 0.6)
_CB_OBSERVABLES: tuple[tuple[int, ...], ...] = ((0,), (1,), (0, 1))


def _h(q: int) -> list[Gate]:
    return [Gate("rz", (q,), (math.pi / 2,)), Gate("sx", (q,)), Gate("rz", (q,), (math.pi / 2,))]


def _prepare(basis: str, q: int) -> list[Gate]:
    if basis == "x":
        return _h(q)
    if basis == "y":
        return _h(q) + [Gate("rz", (q,), (math.pi / 2,))]
    return []


def _rotate_to_z(basis: str, q: int) -> list[Gate]:
    if basis == "x":
        return _h(q)
    if basis == "y":
        return [Gate("rz", (q,), (-math.pi / 2,))] + _h(q)
    return []


@lru_cache(maxsize=None)
def return_probe_program(depth: int) -> LogicalProgram:
    """H.H, then depth ZZ(pi/2) gates (identity when depth % 4 == 0), then H.H and measure; ideal outcome |00>."""
    if depth < 4 or depth % 4:
        raise ValueError("return-probe depth must be a positive multiple of 4")
    return cb_probe_program(("x", "x"), depth)


def cb_probe_program(
    bases: tuple[str, str], depth: int, frame: tuple[float, float] = (0.0, 0.0)
) -> LogicalProgram:
    """Cycle-benchmarking probe of a pair: eigenstate of the given bases, depth ZZ(pi/2) gates, measure back.

    ``frame`` prepends rz(beta) to each qubit. Acting on |0> this is only a global phase, so the probe is logically
    unchanged, but Z rotations are virtual on IonQ hardware: every later pulse on that ion is shifted by beta, which
    rotates the whole experiment about Z in the laboratory frame. Without it the X- and Y-basis probes are the same
    physical experiment and errors fixed in the laser frame are sampled along one equatorial axis only; with beta
    uniform the probed laboratory axes are uniform and the estimate is unbiased for the process fidelity.
    """
    bases = tuple(str(b) for b in bases)
    if len(bases) != 2 or any(b not in BASES for b in bases):
        raise ValueError(f"bases must be two of {BASES}")
    if depth < 4 or depth % 4:
        raise ValueError("probe depth must be a positive multiple of 4")
    beta = (float(frame[0]), float(frame[1]))
    gates = [Gate("rz", (q,), (beta[q],)) for q in (0, 1) if beta[q] != 0.0]
    gates += _prepare(bases[0], 0) + _prepare(bases[1], 1)
    gates += [Gate("rzz", (0, 1), (math.pi / 2,)) for _ in range(depth)]
    gates += _rotate_to_z(bases[0], 0) + _rotate_to_z(bases[1], 1)
    return LogicalProgram(2, 2, gates, [(0, 0), (1, 1)], name=f"canary_cb_{bases[0]}{bases[1]}_{depth}")


def _log_estimate(freqs: Mapping[str, float], shots: float) -> float:
    """Weighted log of <Z_a>, <Z_b>, <Z_a Z_b>, each floored at one standard error."""
    floor = math.sqrt(1.0 / max(float(shots), 1.0))
    ms = (expectation(freqs, (0,)), expectation(freqs, (1,)), expectation(freqs, (0, 1)))
    return sum(w * math.log(max(m, floor)) for w, m in zip(_WEIGHTS, ms))


def _log_variance(ma: float, mb: float, mab: float, shots: float) -> float:
    """Delta-method variance of the weighted log of the three Z-string means from ``shots`` shots."""
    n = max(float(shots), 1.0)
    floor = math.sqrt(1.0 / n)
    wa, wb, wab = _WEIGHTS
    ua, ub, uc = wa / max(ma, floor), wb / max(mb, floor), wab / max(mab, floor)
    vaa = max(1.0 - ma * ma, 1.0 / n)
    vbb = max(1.0 - mb * mb, 1.0 / n)
    vcc = max(1.0 - mab * mab, 1.0 / n)
    q = ua * ua * vaa + ub * ub * vbb + uc * uc * vcc
    q += 2.0 * (ua * ub * (mab - ma * mb) + ua * uc * (mb - ma * mab) + ub * uc * (ma - mb * mab))
    return max(q, 0.0) / n


def _slope_coefficients(depths: Sequence[float]) -> list[float]:
    d = [float(x) for x in depths]
    mean = sum(d) / len(d)
    ss = sum((x - mean) ** 2 for x in d)
    return [(x - mean) / ss for x in d]


def cb_estimate(runs: Sequence[tuple[int, Mapping[str, float], float]]) -> float | None:
    """Pauli-averaged per-gate damping from (depth, frequencies, shots) of one basis pair (least-squares slope)."""
    depths = [float(d) for d, _, _ in runs]
    if len(set(depths)) < 2:
        return None
    coef = _slope_coefficients(depths)
    return -sum(c * _log_estimate(freqs, shots) for c, (_, freqs, shots) in zip(coef, runs))


def cb_variance(
    x: float,
    depths: Sequence[int],
    shots: float,
    spam: Sequence[float] = (1.0, 1.0, 1.0),
    model_sd: float = 0.0,
    groups: int = 1,
) -> float:
    """Sampling variance of the mean of ``groups`` CB estimates if every Pauli decayed at rate ``x`` per gate.

    ``spam`` are the SPAM factors of <Z_a>, <Z_b> and <Z_a Z_b>; ``model_sd`` (log units per circuit) adds variation
    that shot noise does not explain, such as the spread of decay rates across basis pairs.
    """
    coef = _slope_coefficients(depths)
    rate = max(float(x), 0.0)
    total = 0.0
    for c, d in zip(coef, depths):
        f = math.exp(-rate * float(d))
        total += c * c * _log_variance(spam[0] * f, spam[1] * f, spam[2] * f, shots)
    total += model_sd**2 * sum(c * c for c in coef)
    return total / max(int(groups), 1)


def spam_variance(x: float, shots: float, model_sd: float = 0.0) -> float:
    """Sampling variance of a SPAM damping estimate from ``shots`` shots if the true damping were ``x``."""
    n = max(float(shots), 1.0)
    m = max(math.exp(-max(float(x), 0.0)), math.sqrt(1.0 / n))
    return max(1.0 - m * m, 1.0 / n) / (n * m * m) + model_sd**2


@dataclass(frozen=True)
class Observation:
    """One calibrated estimate ``x`` of a resource's damping at time ``t``, with the sampling model of its variance.

    ``kind`` is 'cb' (mean of ``groups`` cycle-benchmarking estimates at ``depths`` with ``shots`` per circuit and
    predicted SPAM factors ``spam``), 'spam' (``errors`` flipped outcomes in ``shots`` symmetrised |0> readouts) or
    'fixed' (an external estimate with variance ``fixed_variance``).
    """

    t: float
    x: float
    kind: str = "fixed"
    shots: float = 0.0
    depths: tuple[int, ...] = ()
    spam: tuple[float, float, float] = (1.0, 1.0, 1.0)
    groups: int = 1
    errors: float = 0.0
    model_sd: float = 0.0
    fixed_variance: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("cb", "spam", "fixed"):
            raise ValueError(f"unknown observation kind {self.kind!r}")
        if not (math.isfinite(self.t) and math.isfinite(self.x)):
            raise ValueError("an observation needs a finite time and estimate")
        if self.kind == "fixed" and not (math.isfinite(self.fixed_variance) and self.fixed_variance > 0.0):
            raise ValueError("a fixed observation needs a positive finite variance")
        if self.kind in ("cb", "spam") and self.shots < 1:
            raise ValueError("a probe observation needs at least one shot")
        if self.kind == "cb" and (len(set(self.depths)) < 2 or len(self.spam) != 3):
            raise ValueError("a CB observation needs two distinct depths and three SPAM factors")

    def variance(self, x: float) -> float:
        """Sampling variance of this estimate if the true damping were ``x``."""
        if self.kind == "cb":
            return cb_variance(x, self.depths, self.shots, self.spam, self.model_sd, self.groups)
        if self.kind == "spam":
            return spam_variance(x, self.shots, self.model_sd)
        return self.fixed_variance

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        data = asdict(self)
        data["depths"] = list(self.depths)
        data["spam"] = list(self.spam)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Observation":
        """Inverse of to_dict."""
        return cls(
            t=float(data["t"]),
            x=float(data["x"]),
            kind=str(data.get("kind", "fixed")),
            shots=float(data.get("shots", 0.0)),
            depths=tuple(int(d) for d in data.get("depths", ())),
            spam=tuple(float(a) for a in data.get("spam", (1.0, 1.0, 1.0))),
            groups=int(data.get("groups", 1)),
            errors=float(data.get("errors", 0.0)),
            model_sd=float(data.get("model_sd", 0.0)),
            fixed_variance=float(data.get("fixed_variance", 0.0)),
        )


@lru_cache(maxsize=None)
def spam_probe_program(num_ions: int) -> LogicalProgram:
    """Prepare |0> on every qubit and measure; the measurement twirl supplies the only gates."""
    n = int(num_ions)
    if n < 1:
        raise ValueError("a SPAM probe needs at least one ion")
    return LogicalProgram(n, n, [], [(q, q) for q in range(n)], name=f"canary_spam_{n}")


def spam_observation(
    freqs: Mapping[str, float], shots: float, clbit: int, model_sd: float, t: float
) -> Observation:
    """SPAM observation of one ion from readout-symmetrised |0> preparations (frequencies already untwirled)."""
    n = max(float(shots), 1.0)
    m = expectation(freqs, (clbit,))
    errors = min(max(0.5 * (1.0 - m) * n, 0.0), n)
    x = -math.log(max(m, math.sqrt(1.0 / n)))
    return Observation(float(t), x, "spam", n, errors=errors, model_sd=float(model_sd))


def phase_gate_count(theta: float, repetitions: int) -> int:
    """Number of ZZ(theta) gates in a phase probe: k (1 + 4 r) with theta = 1/(4k)."""
    k = 0.25 / float(theta)
    if abs(k - round(k)) > 1e-9 or round(k) < 1:
        raise ValueError("phase-probe angle must be 0.25 / k for integer k >= 1")
    return int(round(k)) * (1 + 4 * int(repetitions))


@lru_cache(maxsize=None)
def phase_probe_program(theta: float, repetitions: int, setting: str) -> LogicalProgram:
    """|++>, n ZZ(theta) gates with n * theta = 1/4 (mod 1), then measure X_a X_b ('x') or Y_a Z_b ('yz')."""
    if setting not in ("x", "yz"):
        raise ValueError("setting must be 'x' or 'yz'")
    n = phase_gate_count(theta, repetitions)
    gates = _h(0) + _h(1) + [Gate("rzz", (0, 1), (2.0 * math.pi * float(theta),)) for _ in range(n)]
    gates += _h(0) + _h(1) if setting == "x" else _rotate_to_z("y", 0)
    return LogicalProgram(2, 2, gates, [(0, 0), (1, 1)], name=f"canary_phase_{setting}_{n}")


def phase_estimate(
    x_freqs: Mapping[str, float],
    x_shots: float,
    yz_freqs: Mapping[str, float],
    yz_shots: float,
    theta: float,
    repetitions: int,
) -> tuple[float, float] | None:
    """Coherent over-rotation per ZZ(theta) gate in turns, and its variance, from the two phase-probe settings.

    With the accumulated angle Theta, <X_a> = <X_b> = cos(2 pi Theta) and <Y_a Z_b> = sin(2 pi Theta).
    """
    n = phase_gate_count(theta, repetitions)
    nx, ny = max(float(x_shots), 1.0), max(float(yz_shots), 1.0)
    xa = expectation(x_freqs, (0,))
    xb = expectation(x_freqs, (1,))
    cos_part = 0.5 * (xa + xb)
    var_cos = (max(1.0 - xa * xa, 1.0 / nx) + max(1.0 - xb * xb, 1.0 / nx)) / (4.0 * nx)
    var_cos += 2.0 * (expectation(x_freqs, (0, 1)) - xa * xb) / (4.0 * nx)
    sin_part = expectation(yz_freqs, (0, 1))
    var_sin = max(1.0 - sin_part * sin_part, 1.0 / ny) / ny
    radius2 = cos_part * cos_part + sin_part * sin_part
    if radius2 < 9.0 * (var_cos + var_sin):
        return None
    total = math.atan2(sin_part, cos_part) / (2.0 * math.pi)
    residual = math.remainder(total - 0.25, 1.0)
    var_total = (sin_part**2 * var_cos + cos_part**2 * var_sin) / (radius2**2 * (2.0 * math.pi) ** 2)
    return residual / n, var_total / (n * n)


def bonferroni_z(alpha: float, tests: int) -> float:
    """Two-sided z threshold controlling the family-wise error rate over the given number of tests."""
    return NormalDist().inv_cdf(1.0 - alpha / (2.0 * max(int(tests), 1)))


def z2_flags(
    program: LogicalProgram,
    assignment: Sequence[int],
    freqs: Mapping[str, float],
    shots: float,
    threshold: float,
) -> set[int]:
    """Ions whose <Z> deviates from the Z2-symmetric ideal value 0 by more than ``threshold`` standard errors.

    Stochastic Pauli noise commutes or anticommutes with the global X flip, so it cannot produce a signal; with a
    twirled readout the sentinel responds to non-unital processes (relaxation, leakage, optical pumping).
    """
    flagged: set[int] = set()
    n = max(float(shots), 1.0)
    for q, c in program.measurements:
        m = expectation(freqs, (c,))
        if abs(m) * math.sqrt(n) > threshold:
            flagged.add(int(assignment[q]))
    return flagged
