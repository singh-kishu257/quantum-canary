import itertools
from collections import Counter

import pytest
from qiskit import QuantumCircuit

from quantum_canary import BackendExecutor, Canary, CanaryConfig, pair

aer = pytest.importorskip("qiskit_aer")


def _chain(n=6, steps=4):
    qc = QuantumCircuit(n)
    for _ in range(steps):
        for q in range(n - 1):
            qc.rzz(0.6, q, q + 1)
        qc.rx(0.4, range(n))
    qc.measure_all()
    return qc


def _clock():
    t = [0.0]

    def clock():
        t[0] += 600.0
        return t[0]

    return clock


def _cb_pairs(runs, depth):
    out = []
    for r in runs:
        if r.kind == "cb" and r.depth == depth:
            ions = r.ions
            pack = [(min(ions[k], ions[k + 1]), max(ions[k], ions[k + 1])) for k in range(0, len(ions), 2)]
            assert len(set(ions)) == len(ions)
            out.append(pack)
    return out


@pytest.mark.parametrize("assignments", [1, 4])
def test_coverage_measures_more_pairs_at_equal_calibration_cost(assignments):
    from quantum_canary.emulator import forte_like_emulator

    measured, cb_circuits = {}, {}
    for weight in (0.0, 1.0):
        ex, _ = forte_like_emulator(ions=10, seed=4, bad_pairs=[(1, 2)])
        cfg = CanaryConfig(
            seed=2,
            anneal_steps=1000,
            coverage_weight=weight,
            assignments_per_batch=assignments,
            twirls_per_assignment=8 // assignments,
        )
        canary = Canary(ex, cfg, clock=_clock())
        count = 0
        for _ in range(8):
            res = canary.run(_chain(), shots=1600)
            count += sum(1 for p in res.probes if p.kind == "cb")
        measured[weight] = {r for r in canary.model.resources if r[0] == "pair"}
        cb_circuits[weight] = count
    assert cb_circuits[0.0] == cb_circuits[1.0] == 16
    assert len(measured[1.0]) > len(measured[0.0])


@pytest.mark.parametrize("n", [6, 7])
def test_sweep_probes_every_pair_once(n):
    ex = BackendExecutor(aer.AerSimulator(seed_simulator=1), ions=range(n))
    canary = Canary(ex, CanaryConfig(seed=1, probe_pack=2), clock=_clock())
    all_pairs = set(itertools.combinations(range(n), 2))
    for call in (1, 2):
        runs = canary.sweep(shots=20)
        for depth in canary.config.probe_depths:
            packs = _cb_pairs(runs, depth)
            assert all(len(p) <= 2 for p in packs)
            counts = Counter(p for pack in packs for p in pack)
            assert set(counts) == all_pairs and set(counts.values()) == {1}
        assert all(len(canary.model.evidence(pair(*p))) == call for p in all_pairs)


def test_probe_priority_puts_exploit_suspects_first():
    from quantum_canary import LogicalProgram
    from quantum_canary.resources import pair_damping

    ex = BackendExecutor(aer.AerSimulator(seed_simulator=1), ions=range(10))
    canary = Canary(ex, CanaryConfig(seed=1, probe_pack=4), clock=lambda: 0.0)
    canary.model.update(pair(4, 5), pair_damping(0.004), 1e-6, t=0.0)
    program = LogicalProgram.from_circuit(_chain())
    designs = [[(0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 8)]]
    suspicion = {pair(4, 5): 9.0, pair(2, 3): 3.0, pair(6, 7): 5.0, pair(7, 9): 4.0, pair(0, 1): 1.0}
    packs = canary._select_probes([program], designs, 8, 200, 0.0, suspicion)
    assert len(packs) == 1
    assert packs[0][:2] == [(2, 3), (6, 7)]
    assert (4, 5) not in packs[0] and (7, 9) not in packs[0]
