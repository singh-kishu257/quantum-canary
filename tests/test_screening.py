import json
import math

import numpy as np
import pytest
from qiskit import QuantumCircuit

from quantum_canary import BatchDesigner, Canary, CanaryConfig, Executor, HealthModel, ScreeningModel, pair
from quantum_canary.resources import pair_damping

aer = pytest.importorskip("qiskit_aer")


def _trotter(n=6, steps=3):
    qc = QuantumCircuit(n)
    for q in range(0, n, 2):
        qc.x(q)
    for _ in range(steps):
        for q in range(n - 1):
            qc.rzz(0.5, q, q + 1)
        qc.rx(0.1, range(n))
    qc.measure_all()
    return qc


class _Deferred(Executor):
    """Executor whose device is built after planning, so a fault can be planted where Canary will look first."""

    def __init__(self, n):
        self.inner = None
        self._ions = tuple(range(n))

    @property
    def ions(self):
        return self._ions

    def run(self, circuits, shots):
        return self.inner.run(circuits, shots)


def test_ideal_cancels_in_relative_groups(rng):
    truth = {pair(0, 1): 0.02, pair(1, 2): 0.10, pair(2, 3): 0.02, pair(0, 3): 0.02}
    model = ScreeningModel(CanaryConfig(screen_model_sd=1e-4), clock=lambda: 0.0)
    keys = list(truth)
    for _ in range(200):
        ideal = rng.uniform(-1.0, 0.0)
        rows = []
        for k in rng.permutation(len(keys))[:3]:
            r = keys[int(k)]
            rows.append(({r: 5.0}, math.exp(ideal - 5.0 * truth[r]), 1e8))
        assert model.add_group(rows)
    est = {r: v[0] for r, v in model.estimates().items()}
    diffs = [est[r] - est[pair(0, 1)] for r in truth]
    true_diffs = [truth[r] - truth[pair(0, 1)] for r in truth]
    assert np.allclose(diffs, true_diffs, atol=1e-3)
    assert max(model.suspicion(), key=model.suspicion().get) == pair(1, 2)


def test_groups_need_signal_and_agreeing_signs():
    model = ScreeningModel(CanaryConfig(), clock=lambda: 0.0)
    c = {pair(0, 1): 3.0}
    assert not model.add_group([(c, 0.9, 1000)])
    assert not model.add_group([(c, 0.9, 1000), (c, -0.9, 1000)])
    assert not model.add_group([(c, 0.9, 1000), (c, 0.04, 1000)])
    assert not model.add_group([(c, 0.9, 1000), (c, 0.1, 100)])
    assert model.add_group([(c, 0.9, 1000), ({pair(2, 3): 3.0}, 0.8, 1000)])


def test_screening_penalises_only_unvetted_resources():
    cfg = CanaryConfig(seed=0)
    model = HealthModel(cfg, clock=lambda: 0.0)
    model.update(pair(0, 1), pair_damping(0.004), 1e-6, t=0.0)
    designer = BatchDesigner(cfg, np.random.default_rng(0))
    ions = range(4)
    suspicion = {pair(0, 1): 5.0, pair(2, 3): 5.0, pair(1, 2): -3.0}
    _, base = designer.cost_tables(model, {}, ions, "pessimistic", 0.0)
    _, with_s = designer.cost_tables(model, {}, ions, "pessimistic", 0.0, suspicion)
    assert with_s[0, 1] == base[0, 1]
    assert with_s[1, 2] == base[1, 2]
    assert math.isclose(with_s[2, 3] - base[2, 3], 5.0 * cfg.screen_scale)


def _clock(t):
    return lambda: t[0]


def test_screening_never_changes_tier1_evidence_or_verdicts(monkeypatch):
    from quantum_canary.emulator import forte_like_emulator

    canaries, times = {}, {}
    for on in (True, False):
        ex, _ = forte_like_emulator(ions=8, seed=7, bad_pairs=[(1, 5), (2, 6)], bad_pair_error=0.06)
        times[on] = [0.0]
        cfg = CanaryConfig(seed=11, anneal_steps=500, probe_fraction=0.0, screening=on)
        canaries[on] = Canary(ex, cfg, clock=_clock(times[on]))
        canaries[on].probe([(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3)], shots=200)
    on, off = canaries[True], canaries[False]
    circuit = _trotter()
    designs = {True: [], False: []}
    for b in range(4):
        for flag, canary in canaries.items():
            times[flag][0] = 600.0 * (b + 1)
            if b == 2:
                canary.model.update(pair(0, 3), pair_damping(0.03), 1e-5, t=times[flag][0])
            designs[flag].append(tuple(r.assignment for r in canary.run(circuit, shots=1600).runs))
        assert on.model.resources == off.model.resources
        assert all(on.model.evidence(r) == off.model.evidence(r) for r in on.model.resources)
        assert on.decisions == off.decisions
    assert on.screening.suspicion() and not off.screening.resources
    assert on._guidance(on.prepare(circuit), times[True][0])[1]
    assert designs[True] != designs[False]

    def forbidden(*args, **kwargs):
        raise AssertionError("algorithm data reached the health model")

    monkeypatch.setattr(HealthModel, "record", forbidden)
    monkeypatch.setattr(HealthModel, "update", forbidden)
    before = dict(on.decisions)
    times[True][0] += 600.0
    on.run(circuit, shots=1600)
    rows = [({pair(0, 3): 9.0}, 0.9, 1e6), ({pair(4, 7): 9.0}, 0.1, 1e6)]
    for _ in range(50):
        on.screening.add_group(rows, times[True][0])
    on._after_calibration(times[True][0])
    after = on.decisions
    assert set(before) == set(after)
    off_now = off.tester.decide(on.model, on.family_size, times[True][0])
    assert after == off_now


def test_screening_ranks_planted_bad_pair_from_algorithm_data():
    from quantum_canary.emulator import forte_like_emulator

    ex = _Deferred(8)
    t = [0.0]
    canary = Canary(ex, CanaryConfig(seed=0, anneal_steps=1000, probe_fraction=0.0), clock=_clock(t))
    prog = canary.prepare(_trotter())
    first = canary.plan(prog, 1600).assignments[0][0]
    bad = pair(first[2], first[3])
    ex.inner, _ = forte_like_emulator(ions=8, seed=0, bad_pairs=[bad[1:]], bad_pair_error=0.08)
    for b in range(3):
        t[0] = 600.0 * (b + 1)
        canary.run(prog, shots=1600)
    assert canary.model.resources == ()
    suspicion = canary.screening.suspicion()
    ranked = sorted((r for r in suspicion if r[0] == "pair"), key=lambda r: -suspicion[r])
    assert bad in ranked[:3]


def test_screening_state_round_trips(tmp_path):
    model = ScreeningModel(CanaryConfig(), clock=lambda: 0.0)
    model.add_group([({pair(0, 1): 3.0}, 0.9, 1000), ({pair(2, 3): 3.0}, 0.6, 1000), ({pair(1, 2): 3.0}, 0.8, 1000)])
    clone = ScreeningModel.from_dict(json.loads(json.dumps(model.to_dict())), CanaryConfig(), clock=lambda: 0.0)
    assert clone.suspicion() == model.suspicion()
    ex = _Deferred(4)
    canary = Canary(ex, CanaryConfig(seed=1), screening=model, clock=lambda: 0.0)
    path = tmp_path / "state.json"
    canary.save(path)
    assert Canary.load(path, ex, clock=lambda: 0.0).screening.suspicion() == model.suspicion()
    data = json.loads(path.read_text())
    data["version"] = 2
    del data["screening"]
    path.write_text(json.dumps(data))
    assert Canary.load(path, ex, clock=lambda: 0.0).screening.resources == ()


def _diluted_groups(rng, model, truth, dilution):
    keys = list(truth)
    for _ in range(300):
        ideal = rng.uniform(-1.0, 0.0)
        rows = []
        for k in rng.permutation(len(keys))[:3]:
            r = keys[int(k)]
            rows.append(({r: 5.0}, math.exp(ideal - 5.0 * dilution * truth[r]), 1e9))
        model.add_group(rows)


@pytest.mark.parametrize("dilution, expected", [(0.3, 0.3), (2.0, 1.0), (-0.5, 0.05)])
def test_dilution_calibration_recovers_factor(rng, dilution, expected):
    truth = {pair(0, k): x for k, x in zip(range(1, 8), (0.004, 0.006, 0.003, 0.05, 0.005, 0.02, 0.008))}
    model = ScreeningModel(CanaryConfig(screen_model_sd=1e-4), clock=lambda: 0.0)
    _diluted_groups(rng, model, truth, dilution)
    vetted = [pair(0, 1), pair(0, 2), pair(0, 4), pair(0, 6)]
    tier1 = {r: (truth[r], 1e-8) for r in vetted}
    factor, calibrated = model.calibrated("", tier1)
    assert factor == pytest.approx(expected, abs=0.01)
    if dilution == 0.3:
        for r in (pair(0, 3), pair(0, 5), pair(0, 7)):
            assert calibrated[r][0] == pytest.approx(truth[r], abs=2e-3)
    assert model.calibrated("", dict(list(tier1.items())[:2])) == (None, {})
    assert model.calibrated("other", tier1) == (None, {})


def test_calibrated_costs_follow_the_reading():
    cfg = CanaryConfig(seed=0)
    model = HealthModel(cfg, clock=lambda: 0.0)
    designer = BatchDesigner(cfg, np.random.default_rng(0))
    ions = range(4)
    _, prior = designer.cost_tables(model, {}, ions, "pessimistic", 0.0)
    bad = {pair(0, 1): (pair_damping(0.05), 0.005)}
    good = {pair(0, 1): (pair_damping(0.003), 0.001)}
    vague = {pair(0, 1): (pair_damping(0.003), 1.0)}
    _, with_bad = designer.cost_tables(model, {}, ions, "pessimistic", 0.0, {pair(0, 1): 9.0}, bad)
    _, with_good = designer.cost_tables(model, {}, ions, "pessimistic", 0.0, None, good)
    _, with_vague = designer.cost_tables(model, {}, ions, "pessimistic", 0.0, None, vague)
    assert with_bad[0, 1] > pair_damping(0.04) > prior[0, 1]
    assert with_good[0, 1] < pair_damping(0.006) < prior[0, 1]
    assert with_vague[0, 1] == pytest.approx(prior[0, 1], rel=1e-2)
    assert with_bad[2, 3] == prior[2, 3]
    model.update(pair(0, 1), pair_damping(0.004), 1e-6, t=0.0)
    _, vetted = designer.cost_tables(model, {}, ions, "pessimistic", 0.0, None, bad)
    assert vetted[0, 1] < pair_damping(0.006)
