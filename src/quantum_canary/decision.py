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
    """Test outcome for one resource; llr uses all evidence, llr_probe return-probe evidence only."""

    resource: Resource
    status: Status
    llr: float
    estimate: float
    precision: float
    llr_probe: float = 0.0


class SequentialTest:
    """Wald SPRT of H0: x = x_healthy versus H1: x = x_degraded on Gaussian data likelihoods.

    Algorithm contrasts count every lightcone gate as fully exposed, which can dilute damping
    estimates. A degraded verdict may therefore use all evidence (dilution only makes it harder),
    whereas a healthy verdict requires return-probe evidence, which is on a calibrated scale.
    """

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

    def _llr(self, model: HealthModel, resource: Resource, source: str) -> tuple[float, float, float]:
        x0, x1 = self.thresholds(resource)
        xhat, precision = model.data_marginal(resource, source)
        if precision <= 0.0:
            return 0.0, xhat, 0.0
        return (x1 - x0) * precision * (xhat - 0.5 * (x0 + x1)), xhat, precision

    def test(self, model: HealthModel, resource: Resource) -> Decision:
        """Run the test for one resource."""
        llr, xhat, precision = self._llr(model, resource, "all")
        llr_probe, _, _ = self._llr(model, resource, "probe")
        if llr >= self.upper:
            status = Status.DEGRADED
        elif llr_probe <= self.lower:
            status = Status.HEALTHY
        else:
            status = Status.UNDECIDED
        return Decision(resource, status, float(llr), float(xhat), float(precision), float(llr_probe))

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
