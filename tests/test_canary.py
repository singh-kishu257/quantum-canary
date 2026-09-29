import json

import pytest
from qiskit import QuantumCircuit

from quantum_canary import BackendExecutor, Canary, CanaryConfig, Status, pair

aer = pytest.importorskip("qiskit_aer")
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error


def _ising(n=4, steps=6):
    qc = QuantumCircuit(n)
    for _ in range(steps):
        for q in range(n - 1):
            qc.rzz(0.5, q, q + 1)
        qc.rx(0.45, range(n))
    qc.measure_all()
    return qc


def _executor(bad_pair=(2, 3), bad_error=0.10, ions=8, seed=11):
    nm = NoiseModel()
    for i in range(ions):
        for j in range(ions):
            if i != j:
                e = bad_error if {i, j} == set(bad_pair) else 0.004
                nm.add_quantum_error(depolarizing_error(16 * e / 15, 2), "rzz", [i, j])
        nm.add_readout_error(ReadoutError([[0.995, 0.005], [0.005, 0.995]]), [i])
    return BackendExecutor(aer.AerSimulator(noise_model=nm, seed_simulator=seed), ions=range(ions))


def _clock():
    t = [0.0]

    def clock():
        t[0] += 600.0
        return t[0]

    return clock


def test_end_to_end_avoids_bad_pair(tmp_path):
    canary = Canary(_executor(), CanaryConfig(seed=3, anneal_steps=3000), clock=_clock())
    circ = _ising()
    for _ in range(6):
        result = canary.run(circ, shots=2000)
    rec = result.recommended
    prog = canary.prepare(circ)
    assert pair(2, 3) not in prog.used_resources(rec)
    report = {row["resource"]: row for row in canary.report()}
    assert all(0.0 <= row["error"] < 0.9375 for row in report.values())
    path = tmp_path / "state.json"
    canary.save(path)
    restored = Canary.load(path, canary.executor, clock=_clock())
    assert restored.model.resources == canary.model.resources
    json.loads(path.read_text())


def test_plan_is_side_effect_free():
    canary = Canary(_executor(), CanaryConfig(seed=5, anneal_steps=500), clock=_clock())
    state = canary.rng.bit_generator.state
    plan = canary.plan(_ising(), shots=1600)
    assert canary.rng.bit_generator.state == state
    assert plan.shots_per_circuit == 200
    assert plan.algorithm_circuits == 8
    assert plan.two_qubit_gate_shots > 0 and plan.single_qubit_gate_shots > 0
    assert canary.model.resources == ()


def test_no_healthy_verdict_without_probes():
    cfg = CanaryConfig(seed=7, anneal_steps=500, probe_fraction=0.0)
    canary = Canary(_executor(), cfg, clock=_clock())
    for _ in range(4):
        canary.run(_ising(), shots=2000)
    assert all(d.status is not Status.HEALTHY for d in canary.decisions.values())


def test_run_fixed_and_single_qubit_circuit():
    canary = Canary(_executor(), CanaryConfig(seed=1, anneal_steps=200), clock=_clock())
    res = canary.run_fixed(_ising(), (0, 1, 4, 5), 1000, twirls=2)
    assert abs(sum(res.frequencies().values()) - 1.0) < 1e-9
    one = QuantumCircuit(1)
    one.h(0)
    one.measure_all()
    out = canary.run(one, shots=400)
    assert len(out.runs) >= 1
