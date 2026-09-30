"""Choose logical-to-ion assignments for each batch."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from .assignment import AssignmentSolver
from .config import CanaryConfig
from .decision import Decision, Status
from .estimator import HealthModel
from .program import LogicalProgram
from .resources import Resource, ion, is_pair, pair


class BatchDesigner:
    """One pessimistic (exploit) assignment plus Thompson-sampled (explore) assignments.

    Costs are calibrated damping rates: the posterior mean plus ``pessimism`` standard deviations for the exploit
    assignment, and a posterior draw for each exploratory assignment (resources without evidence are drawn from the
    typical population component with ``explore_typical``, else from the population mixture, whose wide outlier
    component makes exploratory seatings nearly random). Degraded resources carry a large additive penalty. Resources without Tier-1 evidence and a
    positive screening suspicion z carry ``screen_scale * z`` more in the pessimistic and sampled costs, which only
    breaks ties among uncalibrated resources.
    """

    def __init__(self, config: CanaryConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.solver = AssignmentSolver(config.anneal_steps, rng)

    def cost_tables(
        self,
        model: HealthModel,
        decisions: Mapping[Resource, Decision],
        ions: Sequence[int],
        mode: str,
        now: float | None = None,
        suspicion: Mapping[Resource, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-ion and per-pair damping scores ('pessimistic', 'mean' or 'sample')."""
        if mode not in ("pessimistic", "mean", "sample"):
            raise ValueError(f"unknown mode {mode!r}")
        m = len(ions)
        pos = {int(x): k for k, x in enumerate(ions)}
        kappa = self.config.pessimism if mode == "pessimistic" else 0.0
        im, isd = model.prior(ion(0))
        pm, psd = model.prior(pair(0, 1))
        if mode == "sample":
            if self.config.explore_typical:
                iv = self.rng.normal(*model.typical("ion"), m)
                pv = self.rng.normal(*model.typical("pair"), (m, m))
            else:
                iv = model.sample_prior("ion", self.rng, m)
                pv = model.sample_prior("pair", self.rng, (m, m))
            pv = np.triu(pv, 1)
            pv = pv + pv.T
            known = model.sample(self.rng, now)
        else:
            iv = np.full(m, im + kappa * isd)
            pv = np.full((m, m), pm + kappa * psd)
            known = {r: mu + kappa * sd for r, (mu, sd) in model.estimates(now).items()}
        for r, value in known.items():
            if is_pair(r):
                a, b = pos.get(r[1]), pos.get(r[2])
                if a is not None and b is not None:
                    pv[a, b] = pv[b, a] = value
            else:
                a = pos.get(r[1])
                if a is not None:
                    iv[a] = value
        if suspicion and mode != "mean":
            scale = self.config.screen_scale
            for r, z in suspicion.items():
                if z <= 0.0 or r in known:
                    continue
                if is_pair(r):
                    a, b = pos.get(r[1]), pos.get(r[2])
                    if a is not None and b is not None:
                        pv[a, b] += scale * z
                        pv[b, a] += scale * z
                else:
                    a = pos.get(r[1])
                    if a is not None:
                        iv[a] += scale * z
        iv = np.clip(iv, 0.0, None)
        pv = np.clip(pv, 0.0, None)
        penalty = self.config.degraded_penalty
        for r, d in decisions.items():
            if d.status is not Status.DEGRADED:
                continue
            if is_pair(r):
                a, b = pos.get(r[1]), pos.get(r[2])
                if a is not None and b is not None:
                    pv[a, b] += penalty
                    pv[b, a] += penalty
            else:
                a = pos.get(r[1])
                if a is not None:
                    iv[a] += penalty
        np.fill_diagonal(pv, 0.0)
        return iv, pv

    def best(
        self,
        program: LogicalProgram,
        model: HealthModel,
        decisions: Mapping[Resource, Decision],
        ions: Sequence[int],
        mode: str = "pessimistic",
        now: float | None = None,
        suspicion: Mapping[Resource, float] | None = None,
    ) -> tuple[int, ...]:
        """Single best assignment under the chosen scoring mode."""
        wp, wi = program.weights(self.config.angle_exponent)
        iv, pv = self.cost_tables(model, decisions, ions, mode, now, suspicion)
        perm = self.solver.solve(wp, wi, pv, iv)
        return tuple(int(ions[k]) for k in perm)

    def design(
        self,
        program: LogicalProgram,
        model: HealthModel,
        decisions: Mapping[Resource, Decision],
        ions: Sequence[int],
        k: int,
        now: float | None = None,
        suspicion: Mapping[Resource, float] | None = None,
    ) -> list[tuple[int, ...]]:
        """Return up to k distinct assignments; the first is the exploit choice."""
        n, m = program.num_qubits, len(ions)
        if n > m:
            raise ValueError(f"program needs {n} qubits but only {m} ions are available")
        k = min(int(k), math.perm(m, n))
        wp, wi = program.weights(self.config.angle_exponent)
        out: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()

        def add(perm: np.ndarray) -> bool:
            a = tuple(int(ions[x]) for x in perm)
            if a in seen:
                return False
            seen.add(a)
            out.append(a)
            return True

        iv, pv = self.cost_tables(model, decisions, ions, "pessimistic", now, suspicion)
        base = self.solver.solve(wp, wi, pv, iv)
        add(base)
        tries = 0
        while len(out) < k and tries < 4 * k:
            tries += 1
            iv, pv = self.cost_tables(model, decisions, ions, "sample", now, suspicion)
            add(self.solver.solve(wp, wi, pv, iv))
        walk = base
        while len(out) < k:
            walk = self._perturb(walk, m)
            add(walk)
        return out

    def _perturb(self, perm: np.ndarray, m: int) -> np.ndarray:
        out = perm.copy()
        n = out.size
        if n < m and (n < 2 or self.rng.random() < 0.5):
            free = np.setdiff1d(np.arange(m), out)
            out[int(self.rng.integers(n))] = int(self.rng.choice(free))
        elif n >= 2:
            a, c = self.rng.choice(n, size=2, replace=False)
            out[a], out[c] = out[c], out[a]
        return out
