import math

import numpy as np
import pytest
import scipy.stats
from qiskit import QuantumCircuit

from ionpack import (
    BlockMoments,
    block_moments,
    design_effect,
    indicator,
    linear_estimate,
    make_layout,
    pack,
    welch_tost,
    z_parity,
)

T9_COUNTS = {"00": 40, "11": 40, "01": 10, "10": 10}


def two_single_qubit_blocks():
    circs = []
    for _ in range(2):
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        circs.append(qc)
    p = pack(circs, make_layout([1, 1], 36, 0))
    assert p.block_clbits == ((0,), (1,)) and p.sentinel_clbits == ()
    return p


# ---------------------------------------------------------------- T9
def test_t9_block_moments_exact():
    m = block_moments(T9_COUNTS, two_single_qubit_blocks(), indicator("1"))
    np.testing.assert_allclose(m.mean, [0.5, 0.5], rtol=0, atol=1e-12)
    np.testing.assert_allclose(m.cov, [[25 / 99, 15 / 99], [15 / 99, 25 / 99]], rtol=0, atol=1e-12)
    assert m.shots == 100
    assert design_effect(m.cov) == pytest.approx(1.6, abs=1e-12)
    est, se = linear_estimate(m, [0.5, 0.5])
    assert est == pytest.approx(0.5, abs=1e-12)
    assert se == pytest.approx(0.044946657497549475, abs=1e-12)


def test_t9_per_block_funcs_and_errors():
    p = two_single_qubit_blocks()
    m = block_moments(T9_COUNTS, p, [indicator("1"), z_parity([0])])
    np.testing.assert_allclose(m.mean, [0.5, 0.0], atol=1e-12)
    with pytest.raises(ValueError):
        block_moments(T9_COUNTS, p, [indicator("1")])
    with pytest.raises(ValueError):  # float (probability) counts
        block_moments({"00": 0.4, "11": 0.4, "01": 0.1, "10": 0.1}, p, indicator("1"))
    with pytest.raises(ValueError):  # S < 2
        block_moments({"00": 1}, p, indicator("1"))
    with pytest.raises(ValueError):
        linear_estimate(BlockMoments(np.zeros(2), np.eye(2), 10), [1.0])


def test_t9_design_effect_equicorrelated():
    k, var, rho = 7, 0.09, 0.07
    cov = np.full((k, k), rho * var)
    np.fill_diagonal(cov, var)
    assert design_effect(cov) == pytest.approx(1.42, abs=1e-12)
    with pytest.raises(ValueError):
        design_effect(np.zeros((2, 2)))


def test_linear_estimate_matches_direct_per_shot_computation():
    rng = np.random.default_rng(1)
    p = two_single_qubit_blocks()
    keys = ["00", "01", "10", "11"]
    counts = dict(zip(keys, rng.integers(1, 50, size=4).tolist()))
    w = [0.3, -1.2]
    est, se = linear_estimate(block_moments(counts, p, z_parity([0])), w)
    shots = np.array([[(-1) ** int(k[1]), (-1) ** int(k[0])] for k in keys for _ in range(counts[k])])
    theta = shots @ np.array(w)
    assert est == pytest.approx(theta.mean(), abs=1e-12)
    assert se == pytest.approx(theta.std(ddof=1) / math.sqrt(len(theta)), abs=1e-12)


# ---------------------------------------------------------------- T10
def test_t10_z_parity():
    assert z_parity([0])("1") == -1
    assert z_parity([0, 1])("11") == 1
    assert z_parity([1])("10") == -1
    assert z_parity([])("101") == 1
    with pytest.raises(ValueError):
        z_parity([2])("10")
    with pytest.raises(ValueError):
        z_parity([-1])


# ---------------------------------------------------------------- T11
A = [0.90, 0.91, 0.89, 0.905]
B = [0.90, 0.895, 0.91, 0.90, 0.90]


def test_t11_welch_tost_exact():
    r = welch_tost(A, B, margin=0.02, alpha=0.05)
    assert r.diff == pytest.approx(0.00025, abs=1e-9)
    assert r.se == pytest.approx(0.004922313141874124, abs=1e-9)
    assert r.df == pytest.approx(4.9016009267569665, abs=1e-9)
    assert r.df == pytest.approx(scipy.stats.ttest_ind(A, B, equal_var=False).df, abs=1e-9)
    assert r.ci_low == pytest.approx(-0.009712853421595861, abs=1e-9)
    assert r.ci_high == pytest.approx(0.010212853421596028, abs=1e-9)
    assert r.equivalent is True
    assert welch_tost(A, B, margin=0.01).equivalent is False


def test_t11_zero_se_and_validation():
    r = welch_tost([1.0, 1.0], [0.5, 0.5, 0.5], margin=1.0)
    assert r.se == 0 and r.df == math.inf
    assert (r.ci_low, r.ci_high) == (r.diff, r.diff) == (0.5, 0.5)
    assert r.equivalent is True
    for args in [([1.0], B, 0.1), (A, B, 0.0), (A, B, 0.1, 0.5), (A, B, 0.1, 0.0), (A, [1.0, math.nan], 0.1)]:
        with pytest.raises(ValueError):
            welch_tost(*args)
