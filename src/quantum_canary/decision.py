"""Health verdicts with e-values and e-BH false-discovery-rate control.

A resource has healthy and degraded thresholds x0 < x1 in damping units (a = x1 - x0, mid = (x0 + x1) / 2). Every
calibrated observation s contributes one factor to each of two e-values:

* Cycle-benchmarking (and fixed-variance) observations x_s ~ N(x, v_s(x)) use Gaussian likelihood ratios whose
  variance is evaluated at the null threshold, never at the outcome:

      L_s = exp(a (x_s - mid) / v_s(x0))  against "healthy",   L'_s = exp(a (mid - x_s) / v_s(x1))  against "degraded".

  If v_s does not decrease with x, E[L_s] <= exp(a (x - x0) / v_s(x0)) <= 1 for every x <= x0; and E[L'_s] <= 1
  for every x >= x1 as long as v_s(x) <= v_s(x1) (1 + 2 (x - x1) / a), which the cycle-benchmarking variance model
  satisfies up to error rates far beyond the degraded threshold.
* SPAM observations with k flipped outcomes in n shots use the exact binomial likelihood ratio
  L_s = (p1 / p0)^k ((1 - p1) / (1 - p0))^(n - k) and L'_s = 1 / L_s, with p0 < p1 the healthy and degraded readout
  errors. By the AM-GM inequality these are e-values under "p <= p0" and "p >= p1" even when the two
  measurement-twirl settings flip with different probabilities, provided shots are independent. (A Gaussian
  approximation would not be valid here: a batch sees only a few flipped outcomes.)

A factor raised to a weight w in [0, 1] still has expectation at most 1 (Jensen), so the discounted products

    E_degraded = prod_s L_s ** w_s,   E_healthy = prod_s L'_s ** w_s,   w_s = 2 ** (-(now - t_s) / half_life)

are e-values for "healthy at every probe time" and "degraded at every probe time" (Ville/Markov:
P(E >= 1/a) <= a), even though probes are chosen adaptively. Declaring resources degraded (healthy) with the e-BH
procedure over the whole family of ions and pairs controls the false discovery rate at ``alpha`` (``beta``) at every
batch, under arbitrary dependence between resources (Wang and Ramdas, J. R. Stat. Soc. B, 2022). Validity of the
cycle-benchmarking factors rests on the Gaussian (delta-method) approximation of each per-batch estimate and on its
variance model, which the model standard deviation floor makes conservative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .config import CanaryConfig
from .estimator import HealthModel
from .probes import Observation
from .resources import Resource, is_pair, pair_damping, readout_damping


class Status(str, Enum):
    """Health verdict for a resource."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNDECIDED = "undecided"


@dataclass(frozen=True)
class Decision:
    """Verdict for one resource with its log e-values and posterior summary."""

    resource: Resource
    status: Status
    log_e_degraded: float
    log_e_healthy: float
    estimate: float
    sd: float
    evidence: int

    @property
    def llr(self) -> float:
        """Log e-value against health (the discounted log likelihood ratio of degraded vs healthy)."""
        return self.log_e_degraded


def e_bh(log_e: Mapping[Resource, float], level: float, family_size: int) -> set[Resource]:
    """e-BH rejections: the k largest e-values, k = max{k : e_(k) >= m / (level * k)}."""
    m = max(int(family_size), len(log_e), 1)
    ranked = sorted(log_e.items(), key=lambda kv: -kv[1])
    k_star = 0
    for k, (_, value) in enumerate(ranked, start=1):
        if value >= math.log(m / (level * k)):
            k_star = k
    return {r for r, _ in ranked[:k_star]}


class HealthTest:
    """e-value health tests with e-BH control of the false discovery rate."""

    def __init__(self, config: CanaryConfig | None = None) -> None:
        self.config = config or CanaryConfig()

    def thresholds(self, resource: Resource) -> tuple[float, float]:
        """Healthy and degraded damping rates for the resource kind."""
        c = self.config
        if is_pair(resource):
            return pair_damping(c.healthy_pair_infidelity), pair_damping(c.degraded_pair_infidelity)
        return readout_damping(c.healthy_readout_error), readout_damping(c.degraded_readout_error)

    def increments(self, resource: Resource, observation: Observation) -> tuple[float, float]:
        """Log e-value factors of one observation: (against 'healthy', against 'degraded')."""
        c = self.config
        if observation.kind == "spam":
            p0, p1 = c.healthy_readout_error, c.degraded_readout_error
            k, n = observation.errors, observation.shots
            ll = k * math.log(p1 / p0) + (n - k) * math.log((1.0 - p1) / (1.0 - p0))
            return ll, -ll
        x0, x1 = self.thresholds(resource)
        a, mid = x1 - x0, 0.5 * (x0 + x1)
        x = observation.x
        return a * (x - mid) / observation.variance(x0), a * (mid - x) / observation.variance(x1)

    def log_e_values(self, model: HealthModel, resource: Resource, now: float) -> tuple[float, float]:
        """Logs of the discounted e-values (against 'healthy', against 'degraded') of a resource at time ``now``."""
        half_life = self.config.half_life_s
        against_healthy = against_degraded = 0.0
        for obs in model.evidence(resource):
            if obs.t > now:
                continue
            w = 0.5 ** ((now - obs.t) / half_life)
            up, down = self.increments(resource, obs)
            against_healthy += w * up
            against_degraded += w * down
        return against_healthy, against_degraded

    def log_e_degraded(self, model: HealthModel, resource: Resource, now: float) -> float:
        """Log of the discounted e-value against 'healthy' (x <= x0)."""
        return self.log_e_values(model, resource, now)[0]

    def decide(
        self, model: HealthModel, family_size: int | None = None, now: float | None = None
    ) -> dict[Resource, Decision]:
        """Verdicts for every resource with evidence; the family defaults to those resources."""
        now = model._now(now)
        resources = model.resources
        m = max(int(family_size or 0), len(resources))
        values = {r: self.log_e_values(model, r, now) for r in resources}
        log_deg = {r: v[0] for r, v in values.items()}
        log_ok = {r: v[1] for r, v in values.items()}
        degraded = e_bh(log_deg, self.config.alpha, m)
        healthy = e_bh(log_ok, self.config.beta, m)
        out: dict[Resource, Decision] = {}
        for r in resources:
            if r in degraded:
                status = Status.DEGRADED
            elif r in healthy:
                status = Status.HEALTHY
            else:
                status = Status.UNDECIDED
            mean, sd = model.estimate(r, now)
            out[r] = Decision(r, status, log_deg[r], log_ok[r], mean, sd, len(model.evidence(r)))
        return out

    def probability_degraded(self, model: HealthModel, resource: Resource, now: float | None = None) -> float:
        """Posterior probability that the damping rate exceeds the healthy/degraded midpoint."""
        x0, x1 = self.thresholds(resource)
        mean, sd = model.estimate(resource, now)
        mid = 0.5 * (x0 + x1)
        if sd <= 0:
            return float(mean > mid)
        return 0.5 * math.erfc((mid - mean) / (sd * math.sqrt(2.0)))


SequentialTest = HealthTest
