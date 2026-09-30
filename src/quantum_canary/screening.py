"""Screening: relative per-resource damping learned from algorithm (Tier-2) contrasts, used only for guidance.

Within one batch an observable O has the same ideal value under every assignment a, so

    log|m_a| = log|ideal_O| - sum_r c_{a,r} x_r,    c_{a,r} = LogicalProgram.exposure(O, a, angle_exponent),

and centering each (program, observable, batch) group with its own weights eliminates the unknown ideal value (the
profile likelihood of a free intercept). Groups add to a weighted least-squares information matrix that is forgotten
with ``half_life_s``; a ridge prior N(0, outlier_sd^2) per resource fixes the directions that contrasts cannot
identify, such as a common shift of every pair.

The estimates are diluted, because lightcone exposure counts every gate as a full exposure (an upper bound on the
true sensitivity), and they depend on the circuit's state, so they are only compared with their own population. The
suspicion of a resource is its estimate minus the median estimate of the resources of its kind with screening
evidence, in standard errors of its difference from their mean. Suspicion guides placement tie-breaks, probe
targeting and trust; it never enters the health model, its e-values or the verdicts.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .config import CanaryConfig
from .program import LogicalProgram
from .resources import Resource, from_key, kind, to_key
from .results import expectation

MIN_ABS = 0.05

Row = tuple[Mapping[Resource, float], float, float]


class ScreeningModel:
    """Weighted least squares over within-group contrasts of algorithm observables, with forgetting and a ridge."""

    def __init__(self, config: CanaryConfig | None = None, clock: Callable[[], float] = time.time) -> None:
        self.config = config or CanaryConfig()
        self.clock = clock
        self._index: dict[Resource, int] = {}
        self._info = np.zeros((0, 0))
        self._rhs = np.zeros(0)
        self._t: float | None = None

    @property
    def resources(self) -> tuple[Resource, ...]:
        """Resources that appeared in a used group, in a stable order."""
        return tuple(sorted(self._index, key=to_key))

    def _now(self, now: float | None) -> float:
        return float(self.clock()) if now is None else float(now)

    def _grow(self, resources: Iterable[Resource]) -> None:
        new = [r for r in resources if r not in self._index]
        if not new:
            return
        n = len(self._index)
        for k, r in enumerate(new):
            self._index[r] = n + k
        info = np.zeros((n + len(new), n + len(new)))
        info[:n, :n] = self._info
        self._info = info
        self._rhs = np.concatenate([self._rhs, np.zeros(len(new))])

    def _usable(self, rows: Sequence[Row]) -> bool:
        if len(rows) < 2 or len({m > 0 for _, m, _ in rows}) != 1:
            return False
        z = self.config.screen_min_signal_z
        for _, m, n in rows:
            se = math.sqrt(max(1.0 - m * m, 1.0 / n) / n)
            if abs(m) <= max(MIN_ABS, z * se):
                return False
        return True

    def add_group(self, rows: Sequence[Row], t: float | None = None) -> bool:
        """Add one group of (exposure, expectation, shots) rows that share an ideal value; returns whether it was used.

        A group is used only if it has at least two rows, every |m| exceeds max(0.05, screen_min_signal_z standard
        errors) and all signs agree.
        """
        rows = [(c, float(m), max(float(n), 1.0)) for c, m, n in rows]
        if not self._usable(rows):
            return False
        t = self._now(t)
        weight = 1.0
        if self._t is None or t >= self._t:
            if self._t is not None:
                decay = 0.5 ** ((t - self._t) / self.config.half_life_s)
                self._info *= decay
                self._rhs *= decay
            self._t = t
        else:
            weight = 0.5 ** ((self._t - t) / self.config.half_life_s)
        resources = sorted({r for c, _, _ in rows for r in c}, key=to_key)
        self._grow(resources)
        h = np.array([[-float(c.get(r, 0.0)) for r in resources] for c, _, _ in rows])
        y = np.array([math.log(abs(m)) for _, m, _ in rows])
        v = np.array([max(1.0 - m * m, 1.0 / n) / (n * m * m) for _, m, n in rows]) + self.config.screen_model_sd**2
        w = weight / v
        hc = h - (w @ h) / w.sum()
        yc = y - (w @ y) / w.sum()
        idx = [self._index[r] for r in resources]
        self._info[np.ix_(idx, idx)] += hc.T @ (w[:, None] * hc)
        self._rhs[idx] += hc.T @ (w * yc)
        return True

    def update(
        self,
        program: LogicalProgram,
        assignments: Sequence[Sequence[int]],
        results: Sequence[tuple[Mapping[str, float], float]],
        extra: Iterable[Sequence[int]] = (),
        t: float | None = None,
    ) -> int:
        """Add one batch of one program ((frequencies, shots) per assignment); returns the number of groups used."""
        gamma = self.config.angle_exponent
        used = 0
        for obs in program.observables(extra):
            rows = [
                (program.exposure(obs, a, gamma), expectation(f, obs.clbits), n) for a, (f, n) in zip(assignments, results)
            ]
            used += self.add_group(rows, t)
        return used

    def estimates(self, now: float | None = None) -> dict[Resource, tuple[float, float]]:
        """Ridge estimate and standard deviation of each resource's effective (diluted) damping."""
        mean, cov, _ = self._solve(self._now(now))
        return {r: (float(mean[i]), float(math.sqrt(max(cov[i, i], 0.0)))) for r, i in self._index.items()}

    def _solve(self, now: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k = len(self._index)
        scale = 1.0 if self._t is None or now <= self._t else 0.5 ** ((now - self._t) / self.config.half_life_s)
        info = self._info * scale
        cov = np.linalg.inv(info + np.eye(k) / self.config.outlier_sd**2)
        return cov @ (self._rhs * scale), cov, np.diag(info)

    def suspicion(self, now: float | None = None) -> dict[Resource, float]:
        """z-like suspicion of every resource with screening evidence, relative to the others of its kind.

        A resource has screening evidence when its forgotten data information is at least the ridge precision.
        """
        if not self._index:
            return {}
        mean, cov, diag = self._solve(self._now(now))
        floor = 1.0 / self.config.outlier_sd**2
        out: dict[Resource, float] = {}
        for resource_kind in ("pair", "ion"):
            members = [(r, i) for r, i in self._index.items() if kind(r) == resource_kind and diag[i] >= floor]
            if len(members) < 2:
                continue
            idx = [i for _, i in members]
            centre = float(np.median(mean[idx]))
            for r, i in members:
                c = np.zeros(len(mean))
                c[idx] = -1.0 / len(idx)
                c[i] += 1.0
                out[r] = (float(mean[i]) - centre) / math.sqrt(max(float(c @ cov @ c), 1e-300))
        return out

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable state."""
        order = sorted(self._index, key=self._index.get)
        return {
            "resources": [to_key(r) for r in order],
            "info": self._info.tolist(),
            "rhs": self._rhs.tolist(),
            "t": self._t,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        config: CanaryConfig | None = None,
        clock: Callable[[], float] = time.time,
    ) -> "ScreeningModel":
        """Restore a model saved with to_dict."""
        model = cls(config, clock)
        keys = [from_key(k) for k in data.get("resources", [])]
        model._index = {r: i for i, r in enumerate(keys)}
        model._info = np.array(data.get("info", []), dtype=float).reshape(len(keys), len(keys))
        model._rhs = np.array(data.get("rhs", []), dtype=float).reshape(len(keys))
        model._t = None if data.get("t") is None else float(data["t"])
        return model
