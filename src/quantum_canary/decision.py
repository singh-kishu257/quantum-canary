"""Sequential probability ratio tests on resource health."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .config import CanaryConfig
from .estimator import HealthModel
from .resources import Resource, is_pair, pair_damping, readout_damping


class Status(str, Enum):
    """Health verdict for a resource."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNDECIDED = "undecided"


@dataclass(frozen=True)
class Decision:
    """Test outcome for one resource."""

    resource: Resource
    status: Status
    llr: float
    estimate: float
    precision: float


class SequentialTest:
    """Wald SPRT of H0: x = x_healthy versus H1: x = x_degraded on the Gaussian data likelihood."""

    def __init__(self, config: CanaryConfig | None = None) -> None:
        self.config = config or CanaryConfig()
        c = self.config
        self.upper = math.log((1.0 - c.beta) / c.alpha)
        self.lower = math.log(c.beta / (1.0 - c.alpha))

    def thresholds(self, resource: Resource) -> tuple[float, float]:
        """Healthy and degraded damping rates for the resource kind."""
        c = self.config
        if is_pair(resource):
            return pair_damping(c.healthy_pair_infidelity), pair_damping(c.degraded_pair_infidelity)
        return readout_damping(c.healthy_readout_error), readout_damping(c.degraded_readout_error)

    def test(self, model: HealthModel, resource: Resource) -> Decision:
        """Run the test for one resource."""
        x0, x1 = self.thresholds(resource)
        xhat, precision = model.data_marginal(resource)
        if precision <= 0.0:
            return Decision(resource, Status.UNDECIDED, 0.0, xhat, 0.0)
        llr = (x1 - x0) * precision * (xhat - 0.5 * (x0 + x1))
        if llr >= self.upper:
            status = Status.DEGRADED
        elif llr <= self.lower:
            status = Status.HEALTHY
        else:
            status = Status.UNDECIDED
        return Decision(resource, status, float(llr), float(xhat), float(precision))

    def decide(self, model: HealthModel) -> dict[Resource, Decision]:
        """Run the test for every observed resource."""
        return {r: self.test(model, r) for r in model.resources}

    def probability_degraded(self, model: HealthModel, resource: Resource) -> float:
        """Posterior probability that the damping rate exceeds the healthy/degraded midpoint."""
        x0, x1 = self.thresholds(resource)
        mean, sd = model.estimate(resource)
        mid = 0.5 * (x0 + x1)
        if sd <= 0:
            return float(mean > mid)
        return 0.5 * math.erfc((mid - mean) / (sd * math.sqrt(2.0)))
