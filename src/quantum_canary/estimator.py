"""Health model over per-resource damping rates, and the coherent ZZ over-rotation model.

Only calibrated (Tier-1) evidence enters these models: each observation is an independent per-batch estimate of
one resource's damping (from cycle-benchmarking or SPAM probes) with the sampling model of its variance. Algorithm
circuits are never used here, because for non-Clifford circuits the damping of an observable is not a known linear
function of the per-gate rates.

Drift is modelled as an Ornstein-Uhlenbeck process: between observations the deviation of a resource from the
population mean shrinks by 2^(-elapsed / half_life) and the variance relaxes to the population variance. The
posterior of a resource is obtained by Kalman filtering its evidence under a two-component (typical plus outlier)
population mixture, so estimates, Thompson samples and reports all come from one generative model. The measurement
variance of each observation is evaluated at the filter's one-step prediction (as in an extended Kalman filter),
never at the observed value, so an estimate's weight does not depend on its own noise.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Mapping

import numpy as np

from .config import CanaryConfig
from .probes import Observation
from .resources import Resource, from_key, kind, pair_damping, readout_damping, to_key

PRUNE_HALF_LIVES = 20.0


class HealthModel:
    """Per-resource damping rates filtered from calibrated evidence under Ornstein-Uhlenbeck drift.

    The population distribution of each resource kind is a two-component Gaussian mixture: a typical component
    (mean and spread from the configuration or, with ``empirical_bayes``, from robust estimates over calibrated
    resources) and a wide outlier component with weight ``outlier_weight`` and standard deviation
    ``outlier_sd``. Device error distributions are heavy-tailed (a few ion pairs are far worse than the median),
    and a single Gaussian prior would shrink a genuinely bad resource towards the median; the mixture lets such a
    resource leave the typical component after one or two probes. Each component is Kalman-filtered separately and
    weighted by its marginal likelihood.
    """

    def __init__(self, config: CanaryConfig | None = None, clock: Callable[[], float] = time.time) -> None:
        self.config = config or CanaryConfig()
        self.clock = clock
        self._evidence: dict[Resource, list[Observation]] = {}
        self._population: dict[str, tuple[float, float]] = {}

    @property
    def resources(self) -> tuple[Resource, ...]:
        """Resources with at least one observation, in a stable order."""
        return tuple(sorted(self._evidence, key=to_key))

    def _now(self, now: float | None) -> float:
        return float(self.clock()) if now is None else float(now)

    def configured_prior(self, resource_kind: str) -> tuple[float, float]:
        """Mean and standard deviation of the typical component from the configuration."""
        c = self.config
        if resource_kind == "pair":
            return pair_damping(c.prior_pair_infidelity), c.prior_pair_sd
        return readout_damping(c.prior_readout_error), c.prior_ion_sd

    def typical(self, resource_kind: str) -> tuple[float, float]:
        """Mean and standard deviation of the typical population component."""
        return self._population.get(resource_kind, self.configured_prior(resource_kind))

    def components(self, resource_kind: str) -> list[tuple[float, float, float]]:
        """Population mixture as (weight, mean, sd) components."""
        mean, sd = self.typical(resource_kind)
        w = self.config.outlier_weight
        comps = [(1.0 - w, mean, sd)]
        if w > 0.0:
            comps.append((w, mean, self.config.outlier_sd))
        return comps

    def prior(self, resource: Resource) -> tuple[float, float]:
        """Moment-matched mean and standard deviation of the population mixture for the resource's kind."""
        comps = self.components(kind(resource))
        mean = sum(w * m for w, m, _ in comps)
        second = sum(w * (s * s + m * m) for w, m, s in comps)
        return mean, math.sqrt(max(second - mean * mean, 0.0))

    def sample_prior(self, resource_kind: str, rng: np.random.Generator, size: Any = None) -> np.ndarray:
        """Draws from the population mixture."""
        comps = self.components(resource_kind)
        weights = np.array([c[0] for c in comps])
        idx = rng.choice(len(comps), size=size, p=weights / weights.sum())
        means = np.array([c[1] for c in comps])[idx]
        sds = np.array([c[2] for c in comps])[idx]
        return rng.normal(means, sds)

    def update(self, resource: Resource, estimate: float, variance: float, t: float | None = None) -> None:
        """Record an external estimate of a resource's damping rate with a fixed variance."""
        self.record(resource, Observation(self._now(t), float(estimate), "fixed", fixed_variance=float(variance)))

    def record(self, resource: Resource, observation: Observation) -> None:
        """Record a calibrated observation of a resource."""
        ev = self._evidence.setdefault(resource, [])
        ev.append(observation)
        ev.sort(key=lambda e: e.t)

    def evidence(self, resource: Resource) -> tuple[Observation, ...]:
        """Recorded observations of a resource, oldest first."""
        return tuple(self._evidence.get(resource, ()))

    def _filter(
        self, evidence: list[Observation], mean: float, var: float, now: float
    ) -> tuple[float, float, float, list[float]]:
        """Kalman filter under OU drift; returns mean, variance, log marginal likelihood, squared innovations."""
        half_life = self.config.half_life_s
        mu, p, t, loglik = mean, var, None, 0.0
        z2: list[float] = []
        for obs in evidence:
            ts, x = obs.t, obs.x
            if ts > now:
                break
            if t is not None and ts > t:
                rho = 0.5 ** ((ts - t) / half_life)
                mu = mean + rho * (mu - mean)
                p = rho * rho * p + (1.0 - rho * rho) * var
            s = p + obs.variance(max(mu, 0.0))
            loglik -= 0.5 * (math.log(2.0 * math.pi * s) + (x - mu) ** 2 / s)
            z2.append((x - mu) ** 2 / s)
            gain = p / s
            mu += gain * (x - mu)
            p *= 1.0 - gain
            t = ts
        if t is not None and now > t:
            rho = 0.5 ** ((now - t) / half_life)
            mu = mean + rho * (mu - mean)
            p = rho * rho * p + (1.0 - rho * rho) * var
        return mu, p, loglik, z2

    def posterior(self, resource: Resource, now: float | None = None) -> list[tuple[float, float, float]]:
        """Posterior mixture of a damping rate as (weight, mean, variance) components."""
        now = self._now(now)
        comps = self.components(kind(resource))
        ev = self._evidence.get(resource)
        if not ev:
            return [(w, m, s * s) for w, m, s in comps]
        filtered = [(w, *self._filter(ev, m, s * s, now)) for w, m, s in comps]
        logw = [math.log(w) + ll for w, _, _, ll, _ in filtered]
        top = max(logw)
        weights = [math.exp(lw - top) for lw in logw]
        total = sum(weights)
        return [(wt / total, mu, p) for wt, (_, mu, p, _, _) in zip(weights, filtered)]

    def estimate(self, resource: Resource, now: float | None = None) -> tuple[float, float]:
        """Posterior mean and standard deviation of a damping rate at time ``now``."""
        post = self.posterior(resource, now)
        mean = sum(w * m for w, m, _ in post)
        second = sum(w * (p + m * m) for w, m, p in post)
        return float(mean), float(math.sqrt(max(second - mean * mean, 0.0)))

    def outlier_probability(self, resource: Resource, now: float | None = None) -> float:
        """Posterior probability that the resource belongs to the wide outlier component."""
        post = self.posterior(resource, now)
        return float(post[1][0]) if len(post) > 1 else 0.0

    def estimates(self, now: float | None = None) -> dict[Resource, tuple[float, float]]:
        """Posterior mean and standard deviation for every resource with evidence."""
        now = self._now(now)
        return {r: self.estimate(r, now) for r in self.resources}

    def sample(self, rng: np.random.Generator, now: float | None = None) -> dict[Resource, float]:
        """Posterior draw for every resource with evidence (resources are independent a posteriori)."""
        now = self._now(now)
        out: dict[Resource, float] = {}
        for r in self.resources:
            post = self.posterior(r, now)
            k = int(rng.choice(len(post), p=np.array([c[0] for c in post])))
            out[r] = float(rng.normal(post[k][1], math.sqrt(max(post[k][2], 0.0))))
        return out

    def diagnostics(self, now: float | None = None) -> dict[str, tuple[float, int]]:
        """Mean squared standardised innovation per resource kind (about 1 if the evidence variances are right)."""
        now = self._now(now)
        sums: dict[str, list[float]] = {}
        for r, ev in self._evidence.items():
            comps = self.components(kind(r))
            best = max(
                (self._filter(ev, m, s * s, now) + (math.log(w),) for w, m, s in comps),
                key=lambda f: f[2] + f[4],
            )
            sums.setdefault(kind(r), []).extend(best[3])
        return {k: (float(np.mean(v)), len(v)) for k, v in sums.items() if v}

    def summary(self, resource: Resource, now: float | None = None) -> tuple[float, float] | None:
        """Time-discounted inverse-variance mean of a resource's evidence and its variance (data only).

        Variances are evaluated at the configured prior mean of the resource kind, so the weights depend only on the
        probe design and the age of each observation.
        """
        now = self._now(now)
        half_life = self.config.half_life_s
        reference = self.configured_prior(kind(resource))[0]
        num = den = 0.0
        for obs in self._evidence.get(resource, ()):
            if obs.t > now:
                continue
            w = 0.5 ** ((now - obs.t) / half_life) / obs.variance(reference)
            num += w * obs.x
            den += w
        if den <= 0.0:
            return None
        return num / den, 1.0 / den

    def refresh_population(self, now: float | None = None, min_resources: int = 5, pseudo_count: float = 5.0) -> None:
        """Empirical-Bayes update of the typical component of each resource kind (median and MAD)."""
        now = self._now(now)
        for resource_kind in ("pair", "ion"):
            xs, vs = [], []
            for r in self._evidence:
                if kind(r) != resource_kind:
                    continue
                s = self.summary(r, now)
                if s is not None:
                    xs.append(s[0])
                    vs.append(s[1])
            if len(xs) < min_resources:
                self._population.pop(resource_kind, None)
                continue
            x = np.array(xs)
            centre = float(np.median(x))
            spread = 1.4826 * float(np.median(np.abs(x - centre)))
            m0, s0 = self.configured_prior(resource_kind)
            spread2 = max(spread * spread - float(np.median(vs)), (0.1 * s0) ** 2)
            n = float(len(xs))
            mean = (pseudo_count * m0 + n * centre) / (pseudo_count + n)
            var = (pseudo_count * s0 * s0 + n * spread2) / (pseudo_count + n)
            self._population[resource_kind] = (mean, math.sqrt(var))

    def prune(self, now: float | None = None) -> None:
        """Drop evidence older than 20 half-lives (its weight is below 1e-6)."""
        cutoff = self._now(now) - PRUNE_HALF_LIVES * self.config.half_life_s
        for r in list(self._evidence):
            kept = [e for e in self._evidence[r] if e.t >= cutoff]
            if kept:
                self._evidence[r] = kept
            else:
                del self._evidence[r]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable state."""
        return {
            "version": 2,
            "evidence": {to_key(r): [e.to_dict() for e in ev] for r, ev in self._evidence.items()},
            "population": {k: list(v) for k, v in self._population.items()},
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        config: CanaryConfig | None = None,
        clock: Callable[[], float] = time.time,
    ) -> "HealthModel":
        """Restore a model saved with to_dict."""
        if data.get("version") != 2:
            raise ValueError(
                "health state saved before the calibrated-evidence redesign cannot be restored: it pooled algorithm "
                "contrasts, which are no longer treated as calibrated evidence"
            )
        model = cls(config, clock)
        for key, ev in data.get("evidence", {}).items():
            model._evidence[from_key(key)] = sorted((Observation.from_dict(e) for e in ev), key=lambda e: e.t)
        model._population = {k: (float(v[0]), float(v[1])) for k, v in data.get("population", {}).items()}
        return model


class CoherentModel:
    """Coherent ZZ over-rotation of each ion pair, delta(theta) = c + k * theta / 0.25 (turns per gate).

    Estimated from phase probes at one or more reference angles by Bayesian linear regression with independent
    N(0, coherent_prior_sd^2) priors on c and k; evidence is down-weighted by 2^(-age / half_life).
    """

    def __init__(self, config: CanaryConfig | None = None, clock: Callable[[], float] = time.time) -> None:
        self.config = config or CanaryConfig()
        self.clock = clock
        self._evidence: dict[Resource, list[tuple[float, float, float, float]]] = {}

    @property
    def resources(self) -> tuple[Resource, ...]:
        """Pairs with phase-probe evidence."""
        return tuple(sorted(self._evidence, key=to_key))

    def _now(self, now: float | None) -> float:
        return float(self.clock()) if now is None else float(now)

    def update(self, resource: Resource, theta: float, delta: float, variance: float, t: float | None = None) -> None:
        """Record a phase-probe estimate of the over-rotation per ZZ(theta) gate."""
        if not (math.isfinite(delta) and math.isfinite(variance) and variance > 0.0):
            raise ValueError("evidence needs a finite estimate and a positive finite variance")
        self._evidence.setdefault(resource, []).append((self._now(t), float(theta), float(delta), float(variance)))

    def _posterior(self, resource: Resource, now: float) -> tuple[np.ndarray, np.ndarray]:
        s = self.config.coherent_prior_sd
        info = np.eye(2) / (s * s)
        rhs = np.zeros(2)
        for ts, theta, d, v in self._evidence.get(resource, ()):
            if ts > now:
                continue
            w = 0.5 ** ((now - ts) / self.config.half_life_s) / v
            h = np.array([1.0, theta / 0.25])
            info += w * np.outer(h, h)
            rhs += w * h * d
        cov = np.linalg.inv(info)
        return cov @ rhs, cov

    def estimate(self, resource: Resource, theta: float, now: float | None = None) -> tuple[float, float]:
        """Posterior mean and standard deviation of the over-rotation per ZZ(theta) gate, in turns."""
        mean, cov = self._posterior(resource, self._now(now))
        h = np.array([1.0, float(theta) / 0.25])
        return float(h @ mean), float(math.sqrt(max(h @ cov @ h, 0.0)))

    def correction(self, resource: Resource, theta: float, now: float | None = None) -> float:
        """Angle correction (turns) to add to ZZ(theta): minus the over-rotation if resolved, else 0."""
        if resource not in self._evidence:
            return 0.0
        mean, sd = self.estimate(resource, theta, now)
        if abs(mean) < self.config.angle_resolution or abs(mean) < self.config.correction_z * sd:
            return 0.0
        return -mean

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable state."""
        return {"evidence": {to_key(r): [list(e) for e in ev] for r, ev in self._evidence.items()}}

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        config: CanaryConfig | None = None,
        clock: Callable[[], float] = time.time,
    ) -> "CoherentModel":
        """Restore a model saved with to_dict."""
        model = cls(config, clock)
        for key, ev in data.get("evidence", {}).items():
            model._evidence[from_key(key)] = [tuple(float(v) for v in e) for e in ev]
        return model
