import numpy as np

from quantum_canary import CanaryConfig, HealthModel, HealthTest, Observation, Status, ion, pair
from quantum_canary.resources import pair_damping, readout_damping


def test_absolute_probes_recover_rates():
    x_pair, x_i, x_j = pair_damping(0.03), readout_damping(0.01), readout_damping(0.02)
    model = HealthModel(CanaryConfig(prior_pair_sd=0.5, prior_ion_sd=0.5, model_sd=0.0), clock=lambda: 0.0)
    model.update(pair(0, 1), x_pair, 1e-8, t=0.0)
    model.update(ion(0), x_i, 1e-8, t=0.0)
    model.update(ion(1), x_j, 1e-8, t=0.0)
    assert abs(model.estimate(pair(0, 1))[0] - x_pair) < 1e-4
    assert abs(model.estimate(ion(0))[0] - x_i) < 1e-4
    assert abs(model.estimate(ion(1))[0] - x_j) < 1e-4


def test_healthy_requires_probe_evidence():
    cfg = CanaryConfig(model_sd=0.0)
    model = HealthModel(cfg, clock=lambda: 0.0)
    r_good = pair(0, 1)
    test = HealthTest(cfg)
    assert r_good not in test.decide(model, now=0.0)
    obs = Observation(0.0, pair_damping(0.003), "cb", 2000.0, (4, 20), (1.0, 1.0, 1.0), 3)
    model.record(r_good, obs)
    assert test.decide(model, now=0.0)[r_good].status is Status.HEALTHY


def test_degraded_from_probe_evidence():
    cfg = CanaryConfig(model_sd=0.0, prior_pair_sd=0.5)
    model = HealthModel(cfg, clock=lambda: 0.0)
    bad, ok = pair(0, 1), pair(2, 3)
    model.record(bad, Observation(0.0, pair_damping(0.08), "cb", 500.0, (4, 20)))
    model.record(ok, Observation(0.0, pair_damping(0.004), "cb", 500.0, (4, 20)))
    decisions = HealthTest(cfg).decide(model, now=0.0)
    assert decisions[bad].status is Status.DEGRADED
    assert decisions[ok].status is not Status.DEGRADED


def test_forgetting_and_serialisation():
    t = [0.0]
    cfg = CanaryConfig(half_life_s=100.0)
    model = HealthModel(cfg, clock=lambda: t[0])
    model.record(pair(0, 1), Observation(0.0, 0.012, "cb", 500.0, (4, 20)))
    test = HealthTest(cfg)
    e0 = test.log_e_values(model, pair(0, 1), 0.0)
    v0 = model.summary(pair(0, 1), 0.0)[1]
    e1 = test.log_e_values(model, pair(0, 1), 100.0)
    v1 = model.summary(pair(0, 1), 100.0)[1]
    assert np.allclose(e1, (0.5 * e0[0], 0.5 * e0[1]))
    assert abs(v1 - 2.0 * v0) < 1e-12
    clone = HealthModel.from_dict(model.to_dict(), cfg, clock=lambda: t[0])
    assert np.allclose(clone.estimate(pair(0, 1), 50.0), model.estimate(pair(0, 1), 50.0))
    assert clone.evidence(pair(0, 1)) == model.evidence(pair(0, 1))
