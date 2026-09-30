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
    """One pessimistic (exploit) assignment plus exploratory assignments.

    With ``explore_mode='local'`` each exploratory assignment is the exploit with one ion moved to a free ion or two
    ions swapped, changing at most three used pairs, so its contrast with the exploit isolates those pairs. Moves are
    chosen greedily by information: uncalibrated pairs they remove (weighted by 1 + suspicion) or add, each pair
    counted once per batch, with ties going to the lowest cost under a posterior draw; moves onto degraded pairs or
    uncalibrated pairs with suspicion of at least ``trust_suspicion_z`` are skipped. With ``'thompson'`` each is the
    optimum of an independent posterior draw.

    Costs are calibrated damping rates: the posterior mean plus ``pessimism`` standard deviations for the exploit
    assignment, and a posterior draw for each exploratory assignment (resources without evidence are drawn from the
    typical population component with ``explore_typical``, else from the population mixture, whose wide outlier
    component makes exploratory seatings nearly random). Degraded resources carry a large additive penalty.
    Resources without Tier-1 evidence and a positive screening suspicion z carry ``screen_scale * z`` more in the
    pessimistic and sampled costs, which only breaks ties among uncalibrated resources. Pairs without Tier-1 evidence
    that have a dilution-calibrated screening estimate instead use the population mixture updated with that estimate
    as a Gaussian reading (moment-matched mean plus ``pessimism`` sd; draws come from the more probable component
    with ``explore_typical``). None of this reaches the health model.
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
        screened: Mapping[Resource, tuple[float, float]] | None = None,
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
        screened = {r: v for r, v in (screened or {}).items() if is_pair(r) and r not in known}
        for r, (x, sd) in screened.items():
            a, b = pos.get(r[1]), pos.get(r[2])
            if a is not None and b is not None:
                pv[a, b] = pv[b, a] = self._screened_cost(model, x, sd, mode, kappa)
        if suspicion and mode != "mean":
            scale = self.config.screen_scale
            for r, z in suspicion.items():
                if z <= 0.0 or r in known or r in screened:
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

    def _screened_cost(self, model: HealthModel, x: float, sd: float, mode: str, kappa: float) -> float:
        """Pair cost from the population mixture updated with a calibrated screening reading x +- sd."""
        post = []
        for w, m, s in model.components("pair"):
            total = s * s + sd * sd
            loglik = -0.5 * (math.log(2.0 * math.pi * total) + (x - m) ** 2 / total)
            post.append((math.log(w) + loglik, (m * sd * sd + x * s * s) / total, math.sqrt(s * s * sd * sd / total)))
        top = max(p[0] for p in post)
        weights = np.array([math.exp(p[0] - top) for p in post])
        weights /= weights.sum()
        if mode == "sample":
            k = int(np.argmax(weights)) if self.config.explore_typical else int(self.rng.choice(len(post), p=weights))
            return float(self.rng.normal(post[k][1], post[k][2]))
        mean = float(sum(wt * p[1] for wt, p in zip(weights, post)))
        second = float(sum(wt * (p[2] ** 2 + p[1] ** 2) for wt, p in zip(weights, post)))
        return mean + kappa * math.sqrt(max(second - mean * mean, 0.0))

    def best(
        self,
        program: LogicalProgram,
        model: HealthModel,
        decisions: Mapping[Resource, Decision],
        ions: Sequence[int],
        mode: str = "pessimistic",
        now: float | None = None,
        suspicion: Mapping[Resource, float] | None = None,
        screened: Mapping[Resource, tuple[float, float]] | None = None,
    ) -> tuple[int, ...]:
        """Single best assignment under the chosen scoring mode."""
        wp, wi = program.weights(self.config.angle_exponent)
        iv, pv = self.cost_tables(model, decisions, ions, mode, now, suspicion, screened)
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
        screened: Mapping[Resource, tuple[float, float]] | None = None,
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

        iv, pv = self.cost_tables(model, decisions, ions, "pessimistic", now, suspicion, screened)
        base = self.solver.solve(wp, wi, pv, iv)
        add(base)
        if self.config.explore_mode == "local":
            for perm in self._local(wp, wi, base, model, decisions, ions, k - 1, now, suspicion, screened):
                add(perm)
        tries = 0
        while self.config.explore_mode == "thompson" and len(out) < k and tries < 4 * k:
            tries += 1
            iv, pv = self.cost_tables(model, decisions, ions, "sample", now, suspicion, screened)
            add(self.solver.solve(wp, wi, pv, iv))
        walk = base
        while len(out) < k:
            walk = self._perturb(walk, m)
            add(walk)
        return out

    def _local(
        self,
        wp: np.ndarray,
        wi: np.ndarray,
        base: np.ndarray,
        model: HealthModel,
        decisions: Mapping[Resource, Decision],
        ions: Sequence[int],
        count: int,
        now: float | None,
        suspicion: Mapping[Resource, float] | None,
        screened: Mapping[Resource, tuple[float, float]] | None = None,
    ) -> list[np.ndarray]:
        """Up to ``count`` distinct exploit neighbours chosen for information (see the class docstring)."""
        n, m = base.size, len(ions)
        suspicion = suspicion or {}
        vetted = set(model.resources)
        degraded = {r for r, d in decisions.items() if d.status is Status.DEGRADED}
        links = [(a, b) for a in range(n) for b in range(a + 1, n) if wp[a, b] > 0]

        def seated(perm: np.ndarray) -> dict[Resource, float]:
            return {pair(ions[perm[a]], ions[perm[b]]): float(wp[a, b]) for a, b in links}

        old = seated(base)
        taken = set(base.tolist())
        neighbours = []
        for q in range(n):
            for u in (u for u in range(m) if u not in taken):
                nb = base.copy()
                nb[q] = u
                neighbours.append(nb)
        for p in range(n):
            for q in range(p + 1, n):
                nb = base.copy()
                nb[p], nb[q] = nb[q], nb[p]
                neighbours.append(nb)
        moves = []
        z_trust = self.config.trust_suspicion_z
        for nb in neighbours:
            new = seated(nb)
            added = {r: w for r, w in new.items() if r not in old}
            suspect = any(r not in vetted and suspicion.get(r, 0.0) >= z_trust for r in added)
            if not added or suspect or any(r in degraded for r in added):
                continue
            removed = {r: w for r, w in old.items() if r not in new}
            moves.append((nb, added, removed))
        if not moves:
            return []
        limit = max(3, min(len(a) for _, a, _ in moves))
        moves = [mv for mv in moves if len(mv[1]) <= limit]
        covered: set[Resource] = set()
        chosen: list[np.ndarray] = []
        while len(chosen) < count and moves:
            iv, pv = self.cost_tables(model, decisions, ions, "sample", now, suspicion, screened)

            def score(mv: tuple[np.ndarray, dict[Resource, float], dict[Resource, float]]) -> tuple[float, float]:
                nb, added, removed = mv
                new = [(r, w) for r, w in removed.items() if r not in vetted and r not in covered]
                info = sum(w * (1.0 + max(0.0, suspicion.get(r, 0.0))) for r, w in new)
                info += sum(w for r, w in added.items() if r not in vetted and r not in covered)
                return info, -AssignmentSolver.cost(wp, wi, pv, iv, nb)

            best = max(moves, key=score)
            moves = [mv for mv in moves if mv is not best]
            chosen.append(best[0])
            covered.update(r for r in list(best[1]) + list(best[2]) if r not in vetted)
        return chosen

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
