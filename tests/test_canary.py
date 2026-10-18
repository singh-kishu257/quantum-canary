"""Checks of quantum_canary (no hardware, about a minute). Run: pytest"""
import contextlib
import io

import numpy as np
import pytest

import quantum_canary as canary
from quantum_canary import api, inversion as inv, ionq, runner


@pytest.mark.parametrize("arch, truth", [
    ("superconducting", {"T1": 180e-6, "T2": 140e-6, "T2s": 90e-6, "dw": 2 * np.pi * 6e3, "eps": 3e-4}),
    ("superconducting", {"T1": 60e-6, "T2": 30e-6, "T2s": 25e-6, "dw": -2 * np.pi * 15e3, "eps": 1e-3}),
    ("trapped_ion", {"T1": 50.0, "T2": 1.0, "T2s": 0.6, "dw": 2 * np.pi * 3.0, "eps": 2e-4}),
])
def test_exact_data_inverts_exactly(arch, truth):
    base = inv.BackendProfile.from_architecture(arch)
    constants = dict(base.constants)
    prof = inv.BackendProfile(architecture=arch, T1_prior_s=truth["T1"] * 1.2,
                              T2_prior_s=truth["T2"] * 0.85, dt_ns=base.dt_ns,
                              custom_arch=constants, prior_confidence="prior_informed")
    _assert_exact_inversion(prof, truth)


@pytest.mark.parametrize("arch, truth", [
    ("superconducting", {"T1": 180e-6, "T2": 140e-6, "T2s": 90e-6, "dw": 2 * np.pi * 6e3, "eps": 3e-4}),
    ("trapped_ion", {"T1": 50.0, "T2": 1.0, "T2s": 0.6, "dw": 2 * np.pi * 0.3, "eps": 2e-4}),
])
def test_exact_data_inverts_exactly_on_log_spaced_grid(arch, truth):
    _assert_exact_inversion(inv.BackendProfile.from_architecture(arch), truth)


def _assert_exact_inversion(prof, truth):
    constants = prof.constants
    _, meta = inv.build_probe_circuits(prof)
    a, b, shots = constants["p0_given_1"], constants["p1_given_0"], 10 ** 8
    z = inv._gate_z_star(truth["eps"], truth["T1"], truth["T2"], constants["gate_time_ns"] * 1e-9)
    p1 = ([inv.forward_t1(d, truth["T1"], a, b) for d in meta["t1_delays_s"]]
          + [p for d in meta["ramsey_delays_s"] for p in inv.forward_ramsey_xy(d, truth["T2s"], truth["dw"], a, b)]
          + [1 - inv.forward_gate_affine(np.array([n]), truth["eps"], z, a, b)[0] for n in meta["gate_rep_N"]]
          + [inv.forward_echo(d, truth["T2"], a, b) for d in meta["echo_delays_s"]])
    counts = [{"1": round(p * shots), "0": shots - round(p * shots)} for p in p1]
    with contextlib.redirect_stdout(io.StringIO()):
        r = inv.lindblad_inversion(counts, meta, prof, shots_t1=shots, shots_ramsey=shots,
                                   shots_gate=shots, shots_echo=shots)
    got = {"T1": r.T1_s, "T2": r.T2_s, "T2s": r.T2_ramsey_s, "dw": r.delta_omega, "eps": r.epsilon_sx}
    for key, value in truth.items():
        assert got[key] == pytest.approx(value, rel=1e-6), key


def test_probe_groups_at_most_two_and_no_neighbours():
    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
    fez = FakeProviderForBackendV2().backend("fake_fez")
    edges = {tuple(sorted(e)) for e in fez.coupling_map.get_edges()}
    rng = np.random.default_rng(1)
    for _ in range(100):
        qubits = [int(q) for q in rng.choice(fez.num_qubits, int(rng.integers(1, 60)), replace=False)]
        groups = runner.probe_groups(fez, qubits)
        assert len(groups) <= 2
        assert sorted(q for g in groups for q in g) == sorted(qubits)
        for g in groups:
            assert not any(tuple(sorted((x, y))) in edges for i, x in enumerate(g) for y in g[i + 1:])


def test_uniform_pubs_keep_every_circuits_shots():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    shots = [30, 100, 50, 60]
    pubs = runner.uniform_pubs([qc] * len(shots), shots)
    assert len({p[2] for p in pubs}) == 1
    assert [runner.pub_executions(p) for p in pubs] == shots


def test_repeated_check_threshold():
    assert api.repeated_threshold(1) == pytest.approx(2.0)
    assert 2.8 < api.repeated_threshold(10) < 2.9


def test_report_save_load_and_compare(tmp_path):
    r = canary.simulate(2, regime="ideal", seed=7)
    path = tmp_path / "check.json"
    r.save(path)
    loaded = canary.load(path)
    for a, b in zip(r, loaded):
        assert (a.t1, a.t2, a.t2_star, a.delta_omega, a.eps, a.health) == \
               (b.t1, b.t2, b.t2_star, b.delta_omega, b.eps, b.health)
    drift = canary.compare(r, loaded)
    assert drift.stable and drift.changed == []
    assert "fit" in r.render() and "health" in r.render()


def test_cli_arguments():
    from quantum_canary.cli import build_parser
    args = build_parser().parse_args(["check", "fake_fez", "13,14", "--json", "x.json"])
    assert (args.backend, args.qubits, args.json) == ("fake_fez", [13, 14], "x.json")
    args = build_parser().parse_args(["compare", "a.json", "b.json", "c.json"])
    assert args.checks == ["a.json", "b.json", "c.json"]


def test_ionq_priors_from_characterization():
    from qiskit_ionq.ionq_client import Characterization

    class Backend:
        def calibration(self):
            return Characterization({"id": "x", "backend": "qpu.forte-1", "date": "2026-10-14T12:00:00Z",
                                     "qubits": 36, "fidelity": {"spam": {"median": 0.995},
                                                                "1q": {"median": 0.9998}}})
    p = ionq.ionq_priors(Backend())
    assert p["eps_prior"] == pytest.approx(2e-4)
    assert sum(p["spam"]) / 2 == pytest.approx(0.005)
    assert p["spam"][0] / p["spam"][1] == pytest.approx(inv._TI_SPAM_RATIO_P0_OVER_P1)
    assert ionq.ionq_priors(object()) is None


def test_frozen_engine_files_unchanged():
    import hashlib
    import pathlib
    src = pathlib.Path(canary.__file__).parent
    expected = {"inversion.py": "88798959d97358169c659de5aeae5699d167c786800bd9da814cae0cc90f5d51",
                "simulation.py": "0d1c216345eb31e3c4d61810234a9fbd9e4f0eacb9c1111f6628556987f6e5f8"}
    for name, digest in expected.items():
        data = (src / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == digest, \
            f"{name} changed: it is frozen; update this hash only for a deliberate change"


def test_frequency_advice_and_bound_hits():
    from quantum_canary import workflow
    prof = inv.BackendProfile.from_architecture("superconducting", T1_prior_s=100e-6, T2_prior_s=80e-6)
    result = inv.InversionResult(backend_name="x", qubit_id=0, timestamp="", architecture="superconducting",
                                 T1_s=100e-6, T1_sigma_s=5e-6, T2_s=200e-6, T2_sigma_s=10e-6,
                                 delta_omega=2 * np.pi * 20e3, delta_omega_sigma=100.0,
                                 epsilon_sx=3e-4, epsilon_sx_sigma=2e-5,
                                 T2_ramsey_s=190e-6, T2_ramsey_sigma_s=9e-6,
                                 T2_echo_s=200e-6, T2_echo_sigma_s=10e-6)
    q = api.QubitResult.from_qubit_report(runner.QubitReport(qubit=0, result=result, profile=prof,
                                                             health="GREEN"))
    assert q.dd_advice == [] and len(q.frame_advice) == 1
    assert q.at_bound == ["T2"]
    assert workflow._dd_reasons(runner.QubitReport(qubit=0, result=result, profile=prof,
                                                   health="GREEN")) == []


def test_raw_counts_saved_and_loaded(tmp_path):
    raw = canary.runner.run_simulation(n_qubits=1, regime="ideal", seed=3)
    raw.block_counts = {0: [{"circuit": 0, "block": 0, "counts": {"0": 12, "1": 18}}]}
    path = tmp_path / "check.json"
    canary.Report(raw).save(path)
    loaded = canary.load(path)
    assert loaded.raw.block_counts == raw.block_counts
    loaded.save(tmp_path / "again.json")
    assert canary.load(tmp_path / "again.json").raw.block_counts == raw.block_counts


def _ion_report(eps, sigmas, chi2s):
    results = [ionq.IonResult(ion=q, epsilon_sx=e, epsilon_sx_sigma=s, chi2_dof=c, chi2_p=0.5,
                              gate_rep_n=[312, 625, 1250], p0=[0.9, 0.8, 0.7], shots=[500] * 3)
               for q, e, s, c in zip([0, 3, 6], eps, sigmas, chi2s)]
    raw = [{"job_id": f"j{q}", "batch": [q], "N": n, "counts": {"0": 400, "1": 100}, "probabilities": None}
           for q in (0, 3, 6) for n in (312, 625, 1250)]
    return ionq.IonQReport(backend="ionq_qpu.forte-1", ions=[0, 3, 6], shots=500,
                           gate_rep_n=[312, 625, 1250], eps_prior=2e-4, spam=(0.006, 0.004),
                           batches=[[0], [3], [6]], job_ids=["j0", "j3", "j6"],
                           ionq_estimates=[None] * 3, results=results, raw=raw)


def test_ionq_works_like_ibm(tmp_path):
    before = canary.Report(ionq.to_canary_report(
        _ion_report([2e-4, 3e-4, 2e-4], [1e-5, 1e-5, float("inf")], [1.0, 6.0, 1.0])))
    assert [q.health for q in before] == ["GREEN", "YELLOW", "RED"]
    assert before[6].health_reason == "gate fit failed" and np.isnan(before[6].eps)
    assert before[3].fit and np.isnan(before[0].t1) and before[0].dd_advice == []
    assert before.raw.shots_per_qubit == 1500 and len(before.raw.block_counts[0]) == 3
    assert "trapped ions" in before.render() and "n/a" in before.render()
    path = tmp_path / "ions.json"
    before.save(path)
    loaded = canary.load(path)
    assert [(q.eps, q.health) for q in loaded][:2] == [(q.eps, q.health) for q in before][:2]
    after = canary.Report(ionq.to_canary_report(
        _ion_report([4e-4, 3e-4, 2e-4], [1e-5, 1e-5, 1e-5], [1.0, 1.0, 1.0])))
    drift = canary.compare(loaded, after)
    assert drift.changed == [0] and np.isnan(drift.z_scores[6]["epsilon_sx"])
    assert np.isnan(drift.z_scores[0]["T1_s"]) and drift.z_scores[0]["epsilon_sx"] > 10


def test_ionq_backend_detection():
    from qiskit_ionq import IonQProvider
    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
    assert ionq.is_ionq(IonQProvider("offline-test").get_backend("simulator", gateset="native"))
    assert not ionq.is_ionq(FakeProviderForBackendV2().backend("fake_fez"))
