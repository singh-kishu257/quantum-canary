import itertools

import numpy as np

from quantum_canary import AssignmentSolver


def _instance(rng, n, m):
    wp = rng.integers(0, 4, (n, n)).astype(float)
    wp = np.triu(wp, 1)
    wp = wp + wp.T
    wi = rng.random(n)
    p = rng.random((m, m))
    p = p + p.T
    np.fill_diagonal(p, 0.0)
    return wp, wi, p, rng.random(m)


def test_incremental_deltas_match_full_cost(rng):
    for _ in range(40):
        n = int(rng.integers(1, 6))
        m = max(int(rng.integers(n, 9)), 2)
        wp, wi, p, i = _instance(rng, n, m)
        solver = AssignmentSolver(0, np.random.default_rng(1))
        perm = solver._greedy(wp, wi, p, i)
        used = np.zeros(m, dtype=bool)
        used[perm] = True
        if n < 2 and m == n:
            continue
        for _ in range(100):
            c0 = solver.cost(wp, wi, p, i, perm)
            delta, move = solver._propose(wp, wi, p, i, perm, used)
            solver._apply(perm, used, move)
            assert abs(solver.cost(wp, wi, p, i, perm) - c0 - delta) < 1e-9
            assert len(set(perm.tolist())) == n and used.sum() == n


def test_annealing_finds_brute_force_optimum(rng):
    for trial in range(15):
        wp, wi, p, i = _instance(rng, 3, 5)
        best = min(AssignmentSolver.cost(wp, wi, p, i, np.array(x)) for x in itertools.permutations(range(5), 3))
        got = AssignmentSolver(3000, np.random.default_rng(trial)).solve(wp, wi, p, i)
        assert AssignmentSolver.cost(wp, wi, p, i, got) <= best + 1e-9
