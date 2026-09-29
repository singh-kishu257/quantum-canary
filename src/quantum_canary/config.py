"""Configuration for Quantum Canary."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class CanaryConfig:
    """Tunable parameters of the Canary control loop."""

    assignments_per_batch: int = 4
    twirls_per_assignment: int = 2
    optimization_level: int = 1
    half_life_s: float = 7200.0
    prior_pair_infidelity: float = 0.005
    prior_pair_sd: float = 0.05
    prior_readout_error: float = 0.005
    prior_ion_sd: float = 0.05
    model_sd: float = 0.02
    healthy_pair_infidelity: float = 0.008
    degraded_pair_infidelity: float = 0.02
    healthy_readout_error: float = 0.01
    degraded_readout_error: float = 0.04
    alpha: float = 0.01
    beta: float = 0.01
    pessimism: float = 1.0
    min_signal_z: float = 4.0
    min_abs_signal: float = 0.05
    probe_depths: tuple[int, ...] = (4, 12)
    probe_fraction: float = 0.25
    max_probe_shots_per_ion: int = 5000
    symmetry_z: float = 4.0
    anneal_steps: int = 20000
    degraded_penalty: float = 1.0e3
    seed: int | None = None

    def __post_init__(self) -> None:
        self.probe_depths = tuple(int(d) for d in self.probe_depths)
        self.validate()

    def validate(self) -> None:
        """Raise ValueError on inconsistent settings."""
        if self.assignments_per_batch < 1:
            raise ValueError("assignments_per_batch must be >= 1")
        if self.twirls_per_assignment < 1:
            raise ValueError("twirls_per_assignment must be >= 1")
        if self.optimization_level not in (0, 1, 2, 3):
            raise ValueError("optimization_level must be 0, 1, 2 or 3")
        if self.half_life_s <= 0:
            raise ValueError("half_life_s must be positive")
        for name in ("prior_pair_sd", "prior_ion_sd"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.model_sd < 0:
            raise ValueError("model_sd must be non-negative")
        for name in ("prior_pair_infidelity", "healthy_pair_infidelity", "degraded_pair_infidelity"):
            if not 0.0 <= getattr(self, name) < 0.75:
                raise ValueError(f"{name} must lie in [0, 0.75)")
        for name in ("prior_readout_error", "healthy_readout_error", "degraded_readout_error"):
            if not 0.0 <= getattr(self, name) < 0.5:
                raise ValueError(f"{name} must lie in [0, 0.5)")
        if self.healthy_pair_infidelity >= self.degraded_pair_infidelity:
            raise ValueError("healthy_pair_infidelity must be below degraded_pair_infidelity")
        if self.healthy_readout_error >= self.degraded_readout_error:
            raise ValueError("healthy_readout_error must be below degraded_readout_error")
        if not (0.0 < self.alpha < 0.5 and 0.0 < self.beta < 0.5):
            raise ValueError("alpha and beta must lie in (0, 0.5)")
        if self.pessimism < 0:
            raise ValueError("pessimism must be non-negative")
        if self.min_signal_z < 0 or not 0.0 <= self.min_abs_signal < 1.0:
            raise ValueError("invalid signal thresholds")
        if not self.probe_depths or any(d < 4 or d % 4 for d in self.probe_depths):
            raise ValueError("probe_depths must be positive multiples of 4")
        if len(set(self.probe_depths)) < 2:
            raise ValueError("probe_depths needs at least two distinct depths")
        if not 0.0 <= self.probe_fraction <= 1.0:
            raise ValueError("probe_fraction must lie in [0, 1]")
        if self.max_probe_shots_per_ion < 0:
            raise ValueError("max_probe_shots_per_ion must be non-negative")
        if self.anneal_steps < 0:
            raise ValueError("anneal_steps must be non-negative")
        if self.degraded_penalty <= 0:
            raise ValueError("degraded_penalty must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        data = asdict(self)
        data["probe_depths"] = list(self.probe_depths)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CanaryConfig":
        """Build a config from a dictionary, ignoring unknown keys."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})
