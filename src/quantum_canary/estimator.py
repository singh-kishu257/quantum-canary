"""Gaussian health model over per-ion and per-pair damping rates."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .config import CanaryConfig
from .resources import Resource, from_key, is_pair, pair_damping, readout_damping, to_key

Exposure = Mapping[Resource, float]


class HealthModel:
    """Linear-Gaussian model log|<O>| = log|ideal| - sum_r c_r x_r with exponential forgetting.

    Evidence from algorithm contrasts and from return probes is pooled for estimation; probe
    evidence is also kept separately because only it is on a calibrated absolute scale.
    """

    def __init__(self, config: CanaryConfig | None = None, clock: Callable[[], float] = time.time) -> None:
        self.config = config or CanaryConfig()
        self.clock = clock
        self._keys: list[Resource] = []
        self._index: dict[Resource, int] = {}
        self._a = np.zeros((0, 0))
        self._b = np.zeros(0)
        self._a_probe = np.zeros((0, 0))
        self._b_probe = np.zeros(0)
        self._t: float | None = None
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    @property
    def resources(self) -> tuple[Resource, ...]:
        """Resources with at least one observation."""
        return tuple(self._keys)

    def prior(self, resource: Resource) -> tuple[float, float]:
        """Prior mean and standard deviation of a resource damping rate."""
        c = self.config
        if is_pair(resource):
            return pair_damping(c.prior_pair_infidelity), c.prior_pair_sd
        return readout_damping(c.prior_readout_error), c.prior_ion_sd

    def _ensure(self, resources: Iterable[Resource]) -> None:
        new = [r for r in resources if r not in self._index]
        if not new:
            return
        n0 = len(self._keys)
        for r in new:
            self._index[r] = len(self._keys)
            self._keys.append(r)
        n = len(self._keys)
        self._a, self._b = self._grow(self._a, self._b, n0, n)
        self._a_probe, self._b_probe = self._grow(self._a_probe, self._b_probe, n0, n)
        self._cache = {}

    @staticmethod
    def _grow(a0: np.ndarray, b0: np.ndarray, n0: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        a = np.zeros((n, n))
        a[:n0, :n0] = a0
        b = np.zeros(n)
        b[:n0] = b0
        return a, b

    def decay(self, now: float | None = None) -> None:
        """Discount past evidence by 2^(-elapsed / half_life)."""
        now = self.clock() if now is None else float(now)
        if self._t is not None and now > self._t:
            f = 0.5 ** ((now - self._t) / self.config.half_life_s)
            self._a *= f
            self._b *= f
            self._a_probe *= f
            self._b_probe *= f
            self._cache = {}
        if self._t is None or now > self._t:
            self._t = now

    def _stamp(self) -> None:
        if self._t is None:
            self._t = float(self.clock())

    def add_relative(self, observations: Sequence[tuple[Exposure, float, float]]) -> None:
        """Add one group of (exposure, log|m|, variance) sharing an unknown ideal value."""
        if len(observations) < 2:
            return
        self._stamp()
        keys = sorted({r for e, _, _ in observations for r in e}, key=to_key)
        if not keys:
            return
        self._ensure(keys)
        idx = np.array([self._index[r] for r in keys])
        c = np.array([[float(e.get(r, 0.0)) for r in keys] for e, _, _ in observations])
        y = np.array([o[1] for o in observations], dtype=float)
        w = 1.0 / np.array([o[2] for o in observations], dtype=float)
        wsum = w.sum()
        cc = c - (w @ c) / wsum
        t = -(y - (w @ y) / wsum)
        self._a[np.ix_(idx, idx)] += cc.T @ (w[:, None] * cc)
        self._b[idx] += cc.T @ (w * t)
        self._cache = {}

    def add_absolute(self, observations: Sequence[tuple[Exposure, float, float, float]]) -> None:
        """Add (exposure, log|m|, log|ideal|, variance) return-probe observations with known ideal values."""
        if not observations:
            return
        self._stamp()
        keys = sorted({r for e, _, _, _ in observations for r in e}, key=to_key)
        if not keys:
            return
        self._ensure(keys)
        idx = np.array([self._index[r] for r in keys])
        c = np.array([[float(e.get(r, 0.0)) for r in keys] for e, _, _, _ in observations])
        t = np.array([o[2] - o[1] for o in observations], dtype=float)
        w = 1.0 / np.array([o[3] for o in observations], dtype=float)
        block = c.T @ (w[:, None] * c)
        rhs = c.T @ (w * t)
        self._a[np.ix_(idx, idx)] += block
        self._b[idx] += rhs
        self._a_probe[np.ix_(idx, idx)] += block
        self._b_probe[idx] += rhs
        self._cache = {}

    def _posterior(self, source: str = "all") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if source not in ("all", "probe"):
            raise ValueError("source must be 'all' or 'probe'")
        if source not in self._cache:
            n = len(self._keys)
            if n == 0:
                self._cache[source] = (np.zeros(0), np.zeros((0, 0)), np.zeros((0, 0)))
            else:
                a, b = (self._a, self._b) if source == "all" else (self._a_probe, self._b_probe)
                priors = np.array([self.prior(r) for r in self._keys])
                prec0 = 1.0 / priors[:, 1] ** 2
                p = a + np.diag(prec0)
                p = 0.5 * (p + p.T)
                chol = np.linalg.cholesky(p)
                inv_chol = np.linalg.inv(chol)
                cov = inv_chol.T @ inv_chol
                mean = cov @ (b + prec0 * priors[:, 0])
                self._cache[source] = (mean, cov, chol)
        return self._cache[source]

    def estimate(self, resource: Resource) -> tuple[float, float]:
        """Posterior mean and standard deviation of a damping rate."""
        i = self._index.get(resource)
        if i is None:
            return self.prior(resource)
        mean, cov, _ = self._posterior()
        return float(mean[i]), float(math.sqrt(max(cov[i, i], 0.0)))

    def estimates(self) -> dict[Resource, tuple[float, float]]:
        """Posterior mean and standard deviation for every observed resource."""
        mean, cov, _ = self._posterior()
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        return {r: (float(mean[i]), float(sd[i])) for i, r in enumerate(self._keys)}

    def data_marginal(self, resource: Resource, source: str = "all") -> tuple[float, float]:
        """Approximate data-only estimate and precision of a damping rate ('all' or 'probe' evidence)."""
        i = self._index.get(resource)
        if i is None:
            return self.prior(resource)[0], 0.0
        mean, cov, _ = self._posterior(source)
        mu0, sd0 = self.prior(resource)
        p_post = 1.0 / max(cov[i, i], 1e-300)
        p_data = p_post - 1.0 / sd0**2
        if p_data <= 1e-12:
            return mu0, 0.0
        return float((mean[i] * p_post - mu0 / sd0**2) / p_data), float(p_data)

    def probe_information(self, resource: Resource) -> float:
        """Accumulated (decayed) Fisher information on a resource from return probes alone."""
        i = self._index.get(resource)
        return 0.0 if i is None else float(self._a_probe[i, i])

    def sample(self, rng: np.random.Generator) -> dict[Resource, float]:
        """Joint posterior draw for every observed resource."""
        mean, _, chol = self._posterior()
        if mean.size == 0:
            return {}
        draw = mean + np.linalg.solve(chol.T, rng.standard_normal(mean.size))
        return {r: float(draw[i]) for i, r in enumerate(self._keys)}

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable state."""
        return {
            "keys": [to_key(r) for r in self._keys],
            "a": self._a.tolist(),
            "b": self._b.tolist(),
            "a_probe": self._a_probe.tolist(),
            "b_probe": self._b_probe.tolist(),
            "t": self._t,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        config: CanaryConfig | None = None,
        clock: Callable[[], float] = time.time,
    ) -> "HealthModel":
        """Restore a model saved with to_dict."""
        model = cls(config, clock)
        model._keys = [from_key(k) for k in data["keys"]]
        model._index = {r: i for i, r in enumerate(model._keys)}
        n = len(model._keys)
        model._a = np.array(data["a"], dtype=float).reshape(n, n)
        model._b = np.array(data["b"], dtype=float).reshape(n)
        model._a_probe = np.array(data.get("a_probe", np.zeros((n, n))), dtype=float).reshape(n, n)
        model._b_probe = np.array(data.get("b_probe", np.zeros(n)), dtype=float).reshape(n)
        model._t = data.get("t")
        return model
