import numpy as np
from qiskit import QuantumCircuit

from quantum_canary import BatchDesigner, CanaryConfig, Decision, HealthModel, LogicalProgram, Status, pair
from quantum_canary.resources import pair_damping


def _chain_program(n=6):
    qc = QuantumCircuit(n)
    for _ in range(3):
        for q in range(n - 1):
            qc.rzz(0.5, q, q + 1)
        qc.rx(0.1, range(n))
    qc.measure_all()
    return LogicalProgram.from_circuit(qc)


def _seated(program, a):
    return {pair(a[u], a[v]) for u, v in program.interaction_weights}


def test_local_explorers_are_small_moves_of_the_exploit():
    cfg = CanaryConfig(seed=0, anneal_steps=500)
    program = _chain_program()
    model = HealthModel(cfg, clock=lambda: 0.0)
    designer = BatchDesigner(cfg, np.random.default_rng(1))
    out = designer.design(program, model, {}, range(10), 4, 0.0)
    assert len(out) == 4 and len(set(out)) == 4
    exploit = out[0]
    for a in out[1:]:
        assert 1 <= sum(x != y for x, y in zip(a, exploit)) <= 2
        assert 1 <= len(_seated(program, a) - _seated(program, exploit)) <= 3


def test_local_exploration_tests_suspects_and_avoids_bad_pairs():
    cfg = CanaryConfig(seed=0, anneal_steps=500)
    program = _chain_program()
    wp, wi = program.weights()
    ions = list(range(10))
    base = np.arange(6)
    model = HealthModel(cfg, clock=lambda: 0.0)
    for r in _seated(program, base) - {pair(2, 3)}:
        model.update(r, pair_damping(0.004), 1e-6, t=0.0)
    degraded = pair(1, 6)
    model.update(degraded, pair_damping(0.08), 1e-6, t=0.0)
    decisions = {degraded: Decision(degraded, Status.DEGRADED, 10.0, -10.0, pair_damping(0.08), 0.001, 1)}
    suspicion = {pair(2, 3): 3.0, pair(3, 7): 4.0, pair(2, 7): 4.0}
    designer = BatchDesigner(cfg, np.random.default_rng(2))
    chosen = designer._local(wp, wi, base, model, decisions, ions, 3, 0.0, suspicion)
    assert len(chosen) == 3
    first = _seated(program, chosen[0])
    assert pair(2, 3) not in first
    for perm in chosen:
        added = _seated(program, perm) - _seated(program, base)
        assert degraded not in added and not added & {pair(3, 7), pair(2, 7)}
