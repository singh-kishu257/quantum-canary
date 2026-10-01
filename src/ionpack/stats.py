"""Statistics for packed data: within-shot moments, linear estimates, design effect, Welch TOST."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy import stats as _st

from .pack import PackedCircuit
from .unpack import _check_clbits, _extract, _normalize


@dataclass(frozen=True)
class BlockMoments:
    """Per-shot mean and covariance of one scalar per block."""

    mean: np.ndarray  # shape (k,)
    cov: np.ndarray  # shape (k, k), per-shot covariance, denominator S - 1
    shots: int  # S = total integer count


@dataclass(frozen=True)
class TostResult:
    """Result of ``welch_tost``; [ci_low, ci_high] is the (1 - 2 alpha) interval for mean(a) - mean(b)."""

    diff: float
    se: float
    df: float
    ci_low: float
    ci_high: float
    equivalent: bool


def indicator(target: str) -> Callable[[str], float]:
    """f(x) = 1.0 if the block key equals ``target`` else 0.0."""
    return lambda x: 1.0 if x == target else 0.0


def z_parity(local_clbits: Sequence[int]) -> Callable[[str], float]:
    """f(x) = (-1)^(sum of the bits of x at the given block-local clbits; clbit 0 is rightmost)."""
    clbits = list(local_clbits)
    if any(isinstance(j, bool) or not isinstance(j, Integral) or j < 0 for j in clbits):
        raise ValueError(f"local clbits must be non-negative ints, got {clbits!r}")

    def f(x: str) -> float:
        _check_clbits(clbits, len(x))
        return -1.0 if sum(x[len(x) - 1 - j] == "1" for j in clbits) % 2 else 1.0

    return f


def block_moments(
    counts: Mapping[str, int],
    packed: PackedCircuit,
    funcs: Callable[[str], float] | Sequence[Callable[[str], float]],
) -> BlockMoments:
    """Mean and per-shot covariance of y(x) = (f_1(x^(1)), ..., f_k(x^(k))) over shots.

    mu = (1/S) sum_x n_x y(x);  cov = (1/(S-1)) sum_x n_x (y(x) - mu)(y(x) - mu)^T.
    Counts must be non-negative integers with S >= 2 (probabilities are rejected).
    """
    k = len(packed.block_clbits)
    fs = [funcs] * k if callable(funcs) else list(funcs)
    if len(fs) != k or not all(callable(f) for f in fs):
        raise ValueError(f"funcs must be one callable or exactly {k} callables")
    n = packed.circuit.num_clbits
    weights, rows = [], []
    for key, value in counts.items():
        if not float(value).is_integer() or value < 0:
            raise ValueError(f"count {value!r} for {key!r} is not a non-negative integer")
        x = _normalize(key, n)
        weights.append(float(value))
        rows.append([float(f(_extract(x, cl))) for f, cl in zip(fs, packed.block_clbits)])
    w = np.asarray(weights, dtype=float)
    y = np.asarray(rows, dtype=float).reshape(len(rows), k)
    shots = float(w.sum())
    if shots < 2:
        raise ValueError(f"need at least 2 shots, got {shots:g}")
    mu = w @ y / shots
    d = y - mu
    cov = (d.T * w) @ d / (shots - 1)
    return BlockMoments(mu, cov, int(shots))


def linear_estimate(m: BlockMoments, weights: Sequence[float]) -> tuple[float, float]:
    """theta = w^T mu and SE = sqrt(w^T cov w / S).

    Exact for i.i.d. shots and includes within-shot correlation between blocks. It does NOT
    include job-to-job drift: for that, compare job-level estimates with ``welch_tost``.
    """
    w = np.asarray(weights, dtype=float)
    if w.shape != m.mean.shape:
        raise ValueError(f"weights must have length {m.mean.shape[0]}")
    var = float(w @ m.cov @ w) / m.shots
    return float(w @ m.mean), math.sqrt(max(var, 0.0))


def design_effect(cov: np.ndarray) -> float:
    """(1^T cov 1) / tr(cov); equals 1 + (k-1) * mean pairwise correlation for equal variances.

    For k identical copies, Var(mean of copies) = (tr cov / k) * design_effect / (k S), so the
    effective number of shots per copy is k S / design_effect.
    """
    c = np.asarray(cov, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be a square matrix")
    tr = float(np.trace(c))
    if tr == 0:
        raise ValueError("tr(cov) is zero")
    return float(c.sum()) / tr


def welch_tost(a: Sequence[float], b: Sequence[float], margin: float, alpha: float = 0.05) -> TostResult:
    """Two one-sided tests (TOST) for equivalence of means, with Welch's unequal-variance SE.

    ``a`` and ``b`` are per-job estimates (e.g. packed-job and solo-job means), so drift between
    jobs is included. Equivalent iff the (1 - 2 alpha) interval lies inside (-margin, margin).
    Reference: Schuirmann, J. Pharmacokinet. Biopharm. (1987).
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) < 2 or len(y) < 2:
        raise ValueError("a and b must be 1-D with at least 2 values each")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("a and b must be finite")
    if not margin > 0:
        raise ValueError("margin must be > 0")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    diff = float(x.mean() - y.mean())
    va, vb = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = float(math.sqrt(va + vb))
    if se == 0:
        df, lo, hi = math.inf, diff, diff
    else:
        df = float((va + vb) ** 2 / (va**2 / (len(x) - 1) + vb**2 / (len(y) - 1)))
        t = float(_st.t.ppf(1 - alpha, df))
        lo, hi = diff - t * se, diff + t * se
    return TostResult(diff, se, df, lo, hi, bool(lo > -margin and hi < margin))
