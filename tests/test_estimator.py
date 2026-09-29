import numpy as np

from quantum_canary import CanaryConfig, HealthModel, SequentialTest, Status, ion, pair
from quantum_canary.resources import pair_damping, readout_damping


def test_ideal_cancels_in_relative_groups(rng):
    truth = {pair(0, 1): 0.02, pair(1, 2): 0.10, pair(2, 3): 0.02, pair(0, 3): 0.02}
    model = HealthModel(CanaryConfig(prior_pair_sd=0.5, model_sd=0.0), clock=lambda: 0.0)
    for _ in range(200):
        ideal = rng.uniform(-3, 0)
        group = []
        keys = list(truth)
        for k in rng.permutation(len(keys))[:3]:
            r = keys[int(k)]
            c = {r: 5.0}
            group.append((c, ideal - 5.0 * truth[r], 1e-4))
        model.add_relative(group)
    est = {r: model.estimate(r)[0] for r in truth}
    diffs = [est[r] - est[pair(0, 1)] for r in truth]
    true_diffs = [truth[r] - truth[pair(0, 1)] for r in truth]
    assert np.allclose(diffs, true_diffs, atol=1e-3)


def test_absolute_probes_recover_rates():
    x_pair, x_i, x_j = pair_damping(0.03), readout_damping(0.01), readout_damping(0.02)
    model = HealthModel(CanaryConfig(prior_pair_sd=0.5, prior_ion_sd=0.5, model_sd=0.0), clock=lambda: 0.0)
    obs = []
    for depth in (4, 12):
        obs.append(({ion(0): 1.0, pair(0, 1): depth}, -(x_i + depth * x_pair), 0.0, 1e-6))
        obs.append(({ion(1): 1.0, pair(0, 1): depth}, -(x_j + depth * x_pair), 0.0, 1e-6))
    model.add_absolute(obs)
    assert abs(model.estimate(pair(0, 1))[0] - x_pair) < 1e-4
    assert abs(model.estimate(ion(0))[0] - x_i) < 1e-3


def test_healthy_requires_probe_evidence():
    cfg = CanaryConfig(model_sd=0.0)
    model = HealthModel(cfg, clock=lambda: 0.0)
    r_good, r_ref = pair(0, 1), pair(2, 3)
    for _ in range(500):
        model.add_relative([({r_good: 1.0}, -0.001, 1e-4), ({r_ref: 1.0}, -0.001, 1e-4)])
    test = SequentialTest(cfg)
    assert test.test(model, r_good).status is not Status.HEALTHY
    model.add_absolute([({r_good: 12.0}, -12 * pair_damping(0.003), 0.0, 1e-6)])
    assert test.test(model, r_good).status is Status.HEALTHY


def test_degraded_from_relative_evidence():
    cfg = CanaryConfig(model_sd=0.0, prior_pair_sd=0.5)
    model = HealthModel(cfg, clock=lambda: 0.0)
    bad, ok = pair(0, 1), pair(2, 3)
    for _ in range(50):
        model.add_relative([({bad: 10.0}, -10 * pair_damping(0.08), 1e-4), ({ok: 10.0}, -10 * pair_damping(0.004), 1e-4)])
    model.add_absolute([({ok: 12.0}, -12 * pair_damping(0.004), 0.0, 1e-6)])
    assert SequentialTest(cfg).test(model, bad).status is Status.DEGRADED


def test_forgetting_and_serialisation():
    t = [0.0]
    cfg = CanaryConfig(half_life_s=100.0)
    model = HealthModel(cfg, clock=lambda: t[0])
    model.add_absolute([({pair(0, 1): 8.0}, -0.1, 0.0, 1e-4)])
    info0 = model.probe_information(pair(0, 1))
    t[0] = 100.0
    model.decay()
    assert abs(model.probe_information(pair(0, 1)) - info0 / 2) < 1e-9
    clone = HealthModel.from_dict(model.to_dict(), cfg)
    assert np.allclose(clone.estimate(pair(0, 1)), model.estimate(pair(0, 1)))
    assert clone.probe_information(pair(0, 1)) == model.probe_information(pair(0, 1))
