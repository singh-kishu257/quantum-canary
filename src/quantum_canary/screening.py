"""Screening: relative per-resource damping learned from algorithm (Tier-2) contrasts, used only for guidance.

Within one batch an observable O has the same ideal value under every assignment a, so

    log|m_a| = log|ideal_O| - sum_r c_{a,r} x_r,    c_{a,r} = LogicalProgram.exposure(O, a, angle_exponent),

and centering each (program, observable, batch) group with its own weights eliminates the unknown ideal value (the
profile likelihood of a free intercept). Groups add to a weighted least-squares information matrix, one per program,
that is forgotten with ``half_life_s``; a ridge prior N(0, outlier_sd^2) per resource fixes the directions that
contrasts cannot identify, such as a common shift of every pair.

The estimates are diluted, because lightcone exposure counts every gate as a full exposure (an upper bound on the
true sensitivity), and they depend on the circuit's state, so they are only compared with their own population. The
suspicion of a resource is its estimate minus the median estimate of the resources of its kind with screening
evidence, in standard errors of its difference from their mean. ``calibrated`` rescales a program's estimates by a
dilution factor fitted to the pairs that also have Tier-1 estimates. Suspicion and calibrated estimates guide
placement, probe targeting and trust; they never enter the health model, its e-values or the verdicts.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .config import CanaryConfig
from .program import LogicalProgram
from .resources import Resource, from_key, kind, to_key
from .results import expectation

MIN_ABS = 0.05
MIN_CALIBRATION_PAIRS = 3
DILUTION_RANGE = (0.05, 1.0)

Row = tuple[Mapping[Resource, float], float, float]


def program_key(program: LogicalProgram) -> str:
    """Stable identifier of a program's gates and measurements (screening is kept per program)."""
    text = repr((program.num_qubits, program.num_clbits, program.gates, program.measurements))
    return hashlib.sha1(text.encode()).hexdigest()[:16]


class _Contrasts:
    """Forgotten information matrix and right-hand side of one program's centred contrasts."""

    def __init__(self) -> None:
        self.index: dict[Resource, int] = {}
        self.info = np.zeros((0, 0))
        self.rhs = np.zeros(0)
        self.t: float | None = None

    def grow(self, resources: Iterable[Resource]) -> None:
        new = [r for r in resources if r not in self.index]
        if not new:
            return
        n = len(self.index)
        for k, r in enumerate(new):
            self.index[r] = n + k
        info = np.zeros((n + len(new), n + len(new)))
        info[:n, :n] = self.info
        self.info = info
        self.rhs = np.concatenate([self.rhs, np.zeros(len(new))])

    def add(self, rows: Sequence[Row], t: float, half_life: float, model_sd: float) -> None:
        weight = 1.0
        if self.t is None or t >= self.t:
            if self.t is not None:
                decay = 0.5 ** ((t - self.t) / half_life)
                self.info *= decay
                self.rhs *= decay
            self.t = t
        else:
            weight = 0.5 ** ((self.t - t) / half_life)
        resources = sorted({r for c, _, _ in rows for r in c}, key=to_key)
        self.grow(resources)
        h = np.array([[-float(c.get(r, 0.0)) for r in resources] for c, _, _ in rows])
        y = np.array([math.log(abs(m)) for _, m, _ in rows])
        v = np.array([max(1.0 - m * m, 1.0 / n) / (n * m * m) for _, m, n in rows]) + model_sd**2
        w = weight / v
        hc = h - (w @ h) / w.sum()
        yc = y - (w @ y) / w.sum()
        idx = [self.index[r] for r in resources]
        self.info[np.ix_(idx, idx)] += hc.T @ (w[:, None] * hc)
        self.rhs[idx] += hc.T @ (w * yc)

    def solve(self, now: float, half_life: float, prior_sd: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k = len(self.index)
        scale = 1.0 if self.t is None or now <= self.t else 0.5 ** ((now - self.t) / half_life)
        info = self.info * scale
        cov = np.linalg.inv(info + np.eye(k) / prior_sd**2)
        return cov @ (self.rhs * scale), cov, np.diag(info)

    def relative(
        self, now: float, half_life: float, prior_sd: float, resource_kind: str
    ) -> dict[Resource, tuple[float, float, float]]:
        """(estimate, median of its kind, standard error of its difference from the kind's mean) per evidenced resource.

        A resource has screening evidence when its forgotten data information is at least the ridge precision.
        """
        if not self.index:
            return {}
        mean, cov, diag = self.solve(now, half_life, prior_sd)
        members = [(r, i) for r, i in self.index.items() if kind(r) == resource_kind and diag[i] >= 1.0 / prior_sd**2]
        if len(members) < 2:
            return {}
        idx = [i for _, i in members]
        centre = float(np.median(mean[idx]))
        out = {}
        for r, i in members:
            c = np.zeros(len(mean))
            c[idx] = -1.0 / len(idx)
            c[i] += 1.0
            out[r] = (float(mean[i]), centre, math.sqrt(max(float(c @ cov @ c), 1e-300)))
        return out

    def to_dict(self) -> dict[str, Any]:
        order = sorted(self.index, key=self.index.get)
        return {
            "resources": [to_key(r) for r in order],
            "info": self.info.tolist(),
            "rhs": self.rhs.tolist(),
            "t": self.t,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "_Contrasts":
        out = cls()
        keys = [from_key(k) for k in data.get("resources", [])]
        out.index = {r: i for i, r in enumerate(keys)}
        out.info = np.array(data.get("info", []), dtype=float).reshape(len(keys), len(keys))
        out.rhs = np.array(data.get("rhs", []), dtype=float).reshape(len(keys))
        out.t = None if data.get("t") is None else float(data["t"])
        return out


class ScreeningModel:
    """Weighted least squares over within-group contrasts of algorithm observables, per program, for guidance."""

    def __init__(self, config: CanaryConfig | None = None, clock: Callable[[], float] = time.time) -> None:
        self.config = config or CanaryConfig()
        self.clock = clock
        self._programs: dict[str, _Contrasts] = {}

    @property
    def resources(self) -> tuple[Resource, ...]:
        """Resources that appeared in a used group of any program, in a stable order."""
        return tuple(sorted({r for c in self._programs.values() for r in c.index}, key=to_key))

    @property
    def programs(self) -> tuple[str, ...]:
        """Keys of the programs with screening data."""
        return tuple(sorted(self._programs))

    def _now(self, now: float | None) -> float:
        return float(self.clock()) if now is None else float(now)

    def _usable(self, rows: Sequence[Row]) -> bool:
        if len(rows) < 2 or len({m > 0 for _, m, _ in rows}) != 1:
            return False
        z = self.config.screen_min_signal_z
        for _, m, n in rows:
            se = math.sqrt(max(1.0 - m * m, 1.0 / n) / n)
            if abs(m) <= max(MIN_ABS, z * se):
                return False
        return True

    def add_group(self, rows: Sequence[Row], t: float | None = None, program: str = "") -> bool:
        """Add one group of (exposure, expectation, shots) rows that share an ideal value; returns whether it was used.

        A group is used only if it has at least two rows, every |m| exceeds max(0.05, screen_min_signal_z standard
        errors) and all signs agree.
        """
        rows = [(c, float(m), max(float(n), 1.0)) for c, m, n in rows]
        if not self._usable(rows):
            return False
        contrasts = self._programs.setdefault(program, _Contrasts())
        contrasts.add(rows, self._now(t), self.config.half_life_s, self.config.screen_model_sd)
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
        key = program_key(program)
        used = 0
        for obs in program.observables(extra):
            rows = [
                (program.exposure(obs, a, gamma), expectation(f, obs.clbits), n)
                for a, (f, n) in zip(assignments, results)
            ]
            used += self.add_group(rows, t, key)
        return used

    def estimates(self, now: float | None = None, program: str = "") -> dict[Resource, tuple[float, float]]:
        """Ridge estimate and standard deviation of each resource's effective (diluted) damping in one program."""
        contrasts = self._programs.get(program)
        if contrasts is None or not contrasts.index:
            return {}
        mean, cov, _ = contrasts.solve(self._now(now), self.config.half_life_s, self.config.outlier_sd)
        return {r: (float(mean[i]), float(math.sqrt(max(cov[i, i], 0.0)))) for r, i in contrasts.index.items()}

    def suspicion(self, now: float | None = None, program: str | None = None) -> dict[Resource, float]:
        """z-like suspicion of every resource with screening evidence, relative to the others of its kind.

        A resource has screening evidence when its forgotten data information is at least the ridge precision. With
        ``program=None`` the largest suspicion over all programs is returned.
        """
        now = self._now(now)
        keys = self._programs if program is None else [program] if program in self._programs else []
        out: dict[Resource, float] = {}
        for key in keys:
            for resource_kind in ("pair", "ion"):
                rel = self._programs[key].relative(now, self.config.half_life_s, self.config.outlier_sd, resource_kind)
                for r, (x, centre, se) in rel.items():
                    out[r] = max(out.get(r, -math.inf), (x - centre) / se)
        return out

    def calibrated(
        self, program: str, tier1: Mapping[Resource, tuple[float, float]], now: float | None = None
    ) -> tuple[float | None, dict[Resource, tuple[float, float]]]:
        """Dilution factor of one program and its rescaled pair estimates (mean, sd) in damping units.

        ``tier1`` maps pairs to Tier-1 (cycle-benchmarking) estimates and variances. On the pairs with both kinds of
        estimate, the centred screening estimates are regressed on the centred Tier-1 estimates through the origin with
        weights 1 / (screening variance + Tier-1 variance); the slope, clipped to [0.05, 1], is the dilution factor s.
        Every pair with screening evidence is then mapped to c_tier1 + (x - c_screening) / s with standard deviation
        se / s, where the c are the weighted means over the calibration pairs. Returns (None, {}) with fewer than three
        calibration pairs or no spread in their Tier-1 estimates.
        """
        contrasts = self._programs.get(program)
        if contrasts is None:
            return None, {}
        rel = contrasts.relative(self._now(now), self.config.half_life_s, self.config.outlier_sd, "pair")
        both = [r for r in rel if r in tier1]
        if len(both) < MIN_CALIBRATION_PAIRS:
            return None, {}
        xs = np.array([rel[r][0] for r in both])
        xc = np.array([tier1[r][0] for r in both])
        w = 1.0 / np.array([rel[r][2] ** 2 + tier1[r][1] for r in both])
        cs, cc = float(w @ xs / w.sum()), float(w @ xc / w.sum())
        spread = float(w @ (xc - cc) ** 2)
        if spread <= 0.0:
            return None, {}
        s = float(np.clip(w @ ((xs - cs) * (xc - cc)) / spread, *DILUTION_RANGE))
        return s, {r: (cc + (x - cs) / s, se / s) for r, (x, _, se) in rel.items()}

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable state."""
        return {"programs": {k: c.to_dict() for k, c in self._programs.items()}}

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        config: CanaryConfig | None = None,
        clock: Callable[[], float] = time.time,
    ) -> "ScreeningModel":
        """Restore a model saved with to_dict."""
        model = cls(config, clock)
        model._programs = {k: _Contrasts.from_dict(v) for k, v in data.get("programs", {}).items()}
        return model
