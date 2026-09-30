import copy

import numpy as np
import pytest
from qiskit import QuantumCircuit

from quantum_canary import AssignmentRun, BatchResult, Canary, CanaryConfig, ion, pair
from quantum_canary.resources import pair_damping

aer = pytest.importorskip("qiskit_aer")


def _executor(n=6):
    from quantum_canary import BackendExecutor

    return BackendExecutor(aer.AerSimulator(seed_simulator=3), ions=range(n))


def test_unvetted_suspicious_pair_is_untrusted():
    canary = Canary(_executor(), CanaryConfig(seed=0), clock=lambda: 0.0)
    canary.model.update(pair(0, 1), pair_damping(0.004), 1e-6, t=0.0)
    used = {pair(0, 1), pair(1, 2), ion(0), ion(1), ion(2)}
    assert not canary._trusted(used, set(), {pair(1, 2): 3.0})
    assert canary._trusted(used, set(), {pair(1, 2): 1.0})
    assert canary._trusted(used, set(), {pair(0, 1): 9.0})
    assert not canary._trusted(used, {pair(0, 1)}, {})
    relaxed = Canary(_executor(), CanaryConfig(seed=0, trust_requires_vetting=False), clock=lambda: 0.0)
    assert relaxed._trusted(used, set(), {pair(1, 2): 3.0})


def test_fallback_returns_exploit_run_only():
    runs = [
        AssignmentRun((0, 1), {"00": 1.0}, 400, trusted=False),
        AssignmentRun((2, 3), {"11": 1.0}, 200, trusted=False),
    ]
    batch = BatchResult("c", runs, (0, 1))
    assert batch.frequencies() == {"00": 1.0}
    assert batch.counts() == {"00": 400.0}
    assert batch.frequencies(trusted_only=False) == pytest.approx({"00": 2 / 3, "11": 1 / 3})
    runs[1].trusted = True
    assert batch.frequencies() == {"11": 1.0}


def test_plan_is_side_effect_free_after_learning():
    t = [0.0]
    canary = Canary(_executor(), CanaryConfig(seed=4, anneal_steps=300), clock=lambda: t[0])
    qc = QuantumCircuit(4)
    qc.x([0, 2])
    for _ in range(3):
        for q in range(3):
            qc.rzz(0.5, q, q + 1)
        qc.rx(0.1, range(4))
    qc.measure_all()
    for b in range(2):
        t[0] = 600.0 * (b + 1)
        canary.run(qc, shots=800)
    t[0] = 1800.0

    def snapshot():
        return (
            copy.deepcopy(canary.rng.bit_generator.state),
            canary._group,
            canary.model.to_dict(),
            canary.screening.to_dict(),
            canary.coherent.to_dict(),
            dict(canary._tokens),
            dict(canary._phase_last),
            dict(canary.decisions),
            set(canary.flagged_ions),
        )

    before = snapshot()
    plan = canary.plan(qc, shots=800)
    assert plan.algorithm_circuits > 0 and plan.probe_circuits > 0
    after = snapshot()
    assert before[0] == after[0] and before[1:] == after[1:]
    assert np.isfinite(plan.probe_shot_fraction)
