"""Configuration for Quantum Canary."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

_TWIRL_MODES = ("rc", "commutant", "none")


@dataclass
class CanaryConfig:
    """Tunable parameters of the Canary control loop.

    Batch design: ``assignments_per_batch`` assignments (one pessimistic exploit, the rest Thompson-sampled),
    each run as ``twirls_per_assignment`` independently twirled circuits whose measurement twirls are paired
    so that, for an even number of circuits, every clbit is flipped in exactly half of the circuits of an assignment.

    Calibration (Tier-1): ``probe_fraction`` of the algorithm circuit count is spent on cycle-benchmarking
    probes of ion pairs (two circuits per probed pair, one per depth in ``probe_depths``); ``spam_probe`` adds two
    SPAM circuits per batch that cover every available ion; ``phase_probe_fraction`` spends circuits on coherent ZZ
    phase probes (four per pair and angle). Setting ``probe_fraction`` to 0 disables all automatic calibration.

    The default depths (4, 20) keep the long circuit near the variance-per-gate optimum of a two-depth slope for
    pair infidelities between the healthy and degraded thresholds (about 2x lower variance than (4, 12) for 1.5x
    the ZZ gates).

    Health model: damping rates are damping units (-log of a Pauli or Z expectation per gate or per readout).
    The population prior of each resource kind mixes a typical component (``prior_*``, re-estimated with
    ``empirical_bayes``) with an outlier component of weight ``outlier_weight`` and spread ``outlier_sd``.
    ``model_sd`` (cycle-benchmarking probes) and ``spam_model_sd`` (SPAM probes) are floors, in log units, added to
    calibration measurements for variation that shot noise does not explain; check them with
    ``Canary.diagnostics()`` on hardware.

    Verdicts: ``alpha`` and ``beta`` are the false-discovery-rate levels of the e-BH procedures that declare
    resources degraded and healthy.
    """

    assignments_per_batch: int = 4
    twirls_per_assignment: int = 2
    twirl: str = "rc"
    measure_twirl: bool = True
    optimization_level: int = 1
    anneal_steps: int = 20000
    pessimism: float = 1.0
    degraded_penalty: float = 1.0e3
    angle_exponent: float = 0.0
    seed: int | None = None

    half_life_s: float = 7200.0
    prior_pair_infidelity: float = 0.005
    prior_pair_sd: float = 0.005
    prior_readout_error: float = 0.005
    prior_ion_sd: float = 0.005
    empirical_bayes: bool = True
    outlier_weight: float = 0.1
    outlier_sd: float = 0.05
    model_sd: float = 0.01
    spam_model_sd: float = 0.002

    healthy_pair_infidelity: float = 0.008
    degraded_pair_infidelity: float = 0.02
    healthy_readout_error: float = 0.01
    degraded_readout_error: float = 0.04
    alpha: float = 0.05
    beta: float = 0.05

    probe_depths: tuple[int, ...] = (4, 20)
    probe_fraction: float = 0.25
    probe_bases: int = 3
    max_probe_shots_per_ion: int = 5000
    spam_probe: bool = True

    phase_probe_fraction: float = 0.0
    phase_angles: tuple[float, ...] = (0.25, 0.0625)
    phase_repetitions: int = 2
    coherent_prior_sd: float = 0.01
    angle_correction: bool = False
    angle_resolution: float = 1.0e-3
    correction_z: float = 3.0

    symmetry_alpha: float = 0.01

    def __post_init__(self) -> None:
        if self.twirl is True:
            self.twirl = "rc"
        elif self.twirl is False or self.twirl is None:
            self.twirl = "none"
        self.probe_depths = tuple(int(d) for d in self.probe_depths)
        self.phase_angles = tuple(float(a) for a in self.phase_angles)
        self.validate()

    def validate(self) -> None:
        """Raise ValueError on inconsistent settings."""
        if self.assignments_per_batch < 1:
            raise ValueError("assignments_per_batch must be >= 1")
        if self.twirls_per_assignment < 1:
            raise ValueError("twirls_per_assignment must be >= 1")
        if self.twirl not in _TWIRL_MODES:
            raise ValueError(f"twirl must be one of {_TWIRL_MODES}")
        if self.optimization_level not in (0, 1, 2, 3):
            raise ValueError("optimization_level must be 0, 1, 2 or 3")
        if self.anneal_steps < 0:
            raise ValueError("anneal_steps must be non-negative")
        if self.pessimism < 0:
            raise ValueError("pessimism must be non-negative")
        if self.degraded_penalty <= 0:
            raise ValueError("degraded_penalty must be positive")
        if not math.isfinite(self.angle_exponent) or self.angle_exponent < 0:
            raise ValueError("angle_exponent must be a non-negative number")
        if self.half_life_s <= 0:
            raise ValueError("half_life_s must be positive")
        if not 0.0 <= self.outlier_weight < 1.0:
            raise ValueError("outlier_weight must lie in [0, 1)")
        for name in ("prior_pair_sd", "prior_ion_sd", "outlier_sd", "coherent_prior_sd"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.model_sd < 0 or self.spam_model_sd < 0:
            raise ValueError("model_sd and spam_model_sd must be non-negative")
        for name in ("prior_pair_infidelity", "healthy_pair_infidelity", "degraded_pair_infidelity"):
            if not 0.0 <= getattr(self, name) < 0.9375:
                raise ValueError(f"{name} must lie in [0, 0.9375)")
        if not 0.0 <= self.prior_readout_error < 0.5:
            raise ValueError("prior_readout_error must lie in [0, 0.5)")
        for name in ("healthy_readout_error", "degraded_readout_error"):
            if not 0.0 < getattr(self, name) < 0.5:
                raise ValueError(f"{name} must lie in (0, 0.5)")
        if self.healthy_pair_infidelity >= self.degraded_pair_infidelity:
            raise ValueError("healthy_pair_infidelity must be below degraded_pair_infidelity")
        if self.healthy_readout_error >= self.degraded_readout_error:
            raise ValueError("healthy_readout_error must be below degraded_readout_error")
        if not (0.0 < self.alpha < 1.0 and 0.0 < self.beta < 1.0):
            raise ValueError("alpha and beta must lie in (0, 1)")
        if not self.probe_depths or any(d < 4 or d % 4 for d in self.probe_depths):
            raise ValueError("probe_depths must be positive multiples of 4")
        if len(set(self.probe_depths)) < 2:
            raise ValueError("probe_depths needs at least two distinct depths")
        if not 0.0 <= self.probe_fraction <= 1.0:
            raise ValueError("probe_fraction must lie in [0, 1]")
        if not 1 <= self.probe_bases <= 9:
            raise ValueError("probe_bases must lie in [1, 9]")
        if self.max_probe_shots_per_ion < 0:
            raise ValueError("max_probe_shots_per_ion must be non-negative")
        if not 0.0 <= self.phase_probe_fraction <= 1.0:
            raise ValueError("phase_probe_fraction must lie in [0, 1]")
        if not self.phase_angles:
            raise ValueError("phase_angles must not be empty")
        for a in self.phase_angles:
            k = 0.25 / a if a > 0 else 0.0
            if not 0.0 < a <= 0.25 or abs(k - round(k)) > 1e-9:
                raise ValueError("phase_angles must be of the form 0.25 / k for integer k >= 1")
        if self.phase_repetitions < 0:
            raise ValueError("phase_repetitions must be non-negative")
        if self.angle_resolution < 0 or self.correction_z < 0:
            raise ValueError("angle_resolution and correction_z must be non-negative")
        if not 0.0 < self.symmetry_alpha < 1.0:
            raise ValueError("symmetry_alpha must lie in (0, 1)")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        data = asdict(self)
        data["probe_depths"] = list(self.probe_depths)
        data["phase_angles"] = list(self.phase_angles)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CanaryConfig":
        """Build a config from a dictionary, ignoring unknown keys."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})
