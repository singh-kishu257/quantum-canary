"""Quadratic assignment of logical qubits to ions by simulated annealing."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


class AssignmentSolver:
    """Minimise sum_ab Wp[a,b] P[pi(a),pi(b)] / 2 + sum_a Wi[a] I[pi(a)] over injective maps."""

    def __init__(self, steps: int = 20000, rng: np.random.Generator | None = None) -> None:
        self.steps = int(steps)
        self.rng = rng if rng is not None else np.random.default_rng()

    @staticmethod
    def cost(wp: np.ndarray, wi: np.ndarray, p: np.ndarray, i: np.ndarray, perm: np.ndarray) -> float:
        """Objective value of a permutation given as ion positions."""
        return float(0.5 * np.sum(wp * p[np.ix_(perm, perm)]) + wi @ i[perm])

    def _greedy(self, wp: np.ndarray, wi: np.ndarray, p: np.ndarray, i: np.ndarray) -> np.ndarray:
        n, m = wp.shape[0], i.shape[0]
        order = np.argsort(-(wp.sum(axis=1) + wi + 1e-9 * self.rng.random(n)))
        perm = -np.ones(n, dtype=int)
        free = np.ones(m, dtype=bool)
        placed: list[int] = []
        for a in order:
            cand = np.flatnonzero(free)
            score = wi[a] * i[cand]
            if placed:
                score = score + p[np.ix_(cand, perm[placed])] @ wp[a, placed]
            score = score + 1e-12 * self.rng.random(cand.size)
            perm[a] = cand[int(np.argmin(score))]
            free[perm[a]] = False
            placed.append(int(a))
        return perm

    def solve(
        self,
        wp: np.ndarray,
        wi: np.ndarray,
        p: np.ndarray,
        i: np.ndarray,
        initial: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Return ion positions (indices into I) for each logical qubit."""
        wp = np.asarray(wp, dtype=float)
        wi = np.asarray(wi, dtype=float)
        p = np.array(p, dtype=float)
        i = np.asarray(i, dtype=float)
        np.fill_diagonal(p, 0.0)
        n, m = wp.shape[0], i.shape[0]
        if n > m:
            raise ValueError(f"{n} logical qubits do not fit on {m} ions")
        perm = np.array(initial, dtype=int) if initial is not None else self._greedy(wp, wi, p, i)
        if perm.shape != (n,) or len(set(perm.tolist())) != n or perm.min() < 0 or perm.max() >= m:
            raise ValueError("invalid initial assignment")
        used = np.zeros(m, dtype=bool)
        used[perm] = True
        current = self.cost(wp, wi, p, i, perm)
        best, best_cost = perm.copy(), current
        if self.steps == 0 or (n < 2 and m == n):
            return best
        deltas = [abs(self._propose(wp, wi, p, i, perm, used)[0]) for _ in range(min(64, self.steps))]
        scale = float(np.median([d for d in deltas if d > 0])) if any(d > 0 for d in deltas) else 0.0
        if scale == 0.0:
            return best
        t0, t1 = scale, scale * 1e-3
        for step in range(self.steps):
            temp = t0 * (t1 / t0) ** (step / max(self.steps - 1, 1))
            delta, move = self._propose(wp, wi, p, i, perm, used)
            if delta <= 0 or self.rng.random() < math.exp(-delta / temp):
                self._apply(perm, used, move)
                current += delta
                if current < best_cost - 1e-12:
                    best, best_cost = perm.copy(), current
        return best

    def _propose(
        self,
        wp: np.ndarray,
        wi: np.ndarray,
        p: np.ndarray,
        i: np.ndarray,
        perm: np.ndarray,
        used: np.ndarray,
    ) -> tuple[float, tuple[str, int, int]]:
        n, m = perm.size, used.size
        if n < m and (n < 2 or self.rng.random() < 0.5):
            a = int(self.rng.integers(n))
            free = np.flatnonzero(~used)
            u = int(free[self.rng.integers(free.size)])
            vals = p[u, perm] - p[perm[a], perm]
            vals[a] = 0.0
            delta = float(wp[a] @ vals + wi[a] * (i[u] - i[perm[a]]))
            return delta, ("move", a, u)
        a, c = (int(x) for x in self.rng.choice(n, size=2, replace=False))
        va = p[perm[c], perm] - p[perm[a], perm]
        vc = p[perm[a], perm] - p[perm[c], perm]
        va[a] = va[c] = 0.0
        vc[a] = vc[c] = 0.0
        delta = float(
            wp[a] @ va + wp[c] @ vc + (wi[a] - wi[c]) * (i[perm[c]] - i[perm[a]])
        )
        return delta, ("swap", a, c)

    @staticmethod
    def _apply(perm: np.ndarray, used: np.ndarray, move: tuple[str, int, int]) -> None:
        kind, a, b = move
        if kind == "move":
            used[perm[a]] = False
            perm[a] = b
            used[b] = True
        else:
            perm[a], perm[b] = perm[b], perm[a]
