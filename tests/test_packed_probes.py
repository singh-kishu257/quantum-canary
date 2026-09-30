import math

import numpy as np
import pytest

from quantum_canary import Canary, CanaryConfig, compile_twirled, pair
from quantum_canary.probes import BASIS_PAIRS, cb_estimate, cb_packed_program, cb_probe_program
from quantum_canary.resources import pair_damping, pair_infidelity
from quantum_canary.results import marginal_frequencies

from conftest import distribution


def _random_pack(rng, k):
    bases = [BASIS_PAIRS[int(b)] for b in rng.integers(len(BASIS_PAIRS), size=k)]
    frames = [tuple(float(a) for a in rng.uniform(0.0, 2.0 * math.pi, 2)) for _ in range(k)]
    return bases, frames


@pytest.mark.parametrize("k", range(1, 7))
def test_packed_program_matches_single_pair_programs(rng, k):
    bases, frames = _random_pack(rng, k)
    prog = cb_packed_program(bases, 8, frames)
    assert prog.num_qubits == prog.num_clbits == 2 * k
    for i in range(k):
        own = [g for g in prog.gates if set(g.qubits) <= {2 * i, 2 * i + 1}]
        shifted = [(g.name, tuple(q - 2 * i for q in g.qubits), g.params) for g in own]
        single = cb_probe_program(bases[i], 8, frames[i])
        assert shifted == [(g.name, g.qubits, g.params) for g in single.gates]
    if k == 1:
        assert prog.gates == cb_probe_program(bases[0], 8, frames[0]).gates


@pytest.mark.parametrize("k", range(1, 7))
@pytest.mark.parametrize("depth", [4, 8])
def test_packed_program_ideal_outcome_is_all_zeros(rng, k, depth):
    for _ in range(3):
        bases, frames = _random_pack(rng, k)
        ions = [int(x) for x in rng.permutation(2 * k)]
        flips = [c for c in range(2 * k) if rng.random() < 0.5]
        comp = compile_twirled(cb_packed_program(bases, depth, frames), ions, rng, "rc", flips)
        dist = comp.untwirl(distribution(comp.circuit))
        assert abs(dist.get("0" * 2 * k, 0.0) - 1.0) < 1e-9


def _pair_distribution(rng, damping, depth):
    f = math.exp(-damping * depth)
    ma, mb = f * rng.uniform(0.95, 1.0), f * rng.uniform(0.95, 1.0)
    mab = ma * mb + 0.01 * rng.uniform(-1.0, 1.0)
    return {
        "00": 0.25 * (1 + ma + mb + mab),
        "01": 0.25 * (1 + ma - mb - mab),
        "10": 0.25 * (1 - ma + mb - mab),
        "11": 0.25 * (1 - ma - mb + mab),
    }


def _product(parts):
    out = {"": 1.0}
    for d in parts:
        out = {kb + ka: pa * pb for ka, pa in out.items() for kb, pb in d.items()}
    return out


def test_marginals_reproduce_single_pair_cb_estimates(rng):
    k = 4
    rates = rng.uniform(0.002, 0.05, k)
    per_depth = {d: [_pair_distribution(rng, x, d) for x in rates] for d in (4, 20)}
    joint = {d: _product(per_depth[d]) for d in (4, 20)}
    for i in range(k):
        single = cb_estimate([(d, per_depth[d][i], 250.0) for d in (4, 20)])
        packed = cb_estimate([(d, marginal_frequencies(joint[d], (2 * i, 2 * i + 1)), 250.0) for d in (4, 20)])
        assert abs(single - packed) < 1e-12
    assert marginal_frequencies({"0110": 0.5, "1001": 0.5}, (1, 3)) == {"01": 0.5, "10": 0.5}


def test_packed_estimates_agree_with_single_pair_estimates():
    pytest.importorskip("qiskit_aer")
    from quantum_canary.emulator import forte_like_emulator

    ex, truth = forte_like_emulator(ions=8, seed=21, bad_pairs=[(0, 1)], bad_pair_error=0.05)
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    est = {}
    for pack in (4, 1):
        cfg = CanaryConfig(seed=5, probe_pack=pack, model_sd=0.0)
        canary = Canary(ex, cfg, clock=lambda: 0.0)
        runs = canary.probe(pairs, shots=2000, bases=3)
        assert len([r for r in runs if r.kind == "cb"]) == 3 * 2 * (1 if pack == 4 else 4)
        est[pack] = {p: canary.model.evidence(pair(*p))[-1] for p in pairs}
    bad = est[4][(0, 1)]
    assert abs(pair_infidelity(bad.x) - truth.pair_infidelity(0, 1)) < 0.2 * truth.pair_infidelity(0, 1)
    for p in pairs:
        a, b = est[4][p], est[1][p]
        se = math.sqrt(a.variance(a.x) + b.variance(b.x))
        assert abs(a.x - b.x) < 3.0 * se
    assert pair_damping(0.02) < bad.x
