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
    gate = []
    for m, k in zip(meta["rb_lengths"], meta["rb_sequences"]):
        keep = 0.5 + 0.5 * (1 - 2 * inv.RB_PULSES_PER_CLIFFORD * truth["eps"]) ** m
        for j in range(k):
            s0 = keep if j % 2 == 0 else 1 - keep
            gate.append(1 - (s0 * (1 - b) + (1 - s0) * a))
    p1 = ([inv.forward_t1(d, truth["T1"], a, b) for d in meta["t1_delays_s"]]
          + [p for d in meta["ramsey_delays_s"] for p in inv.forward_ramsey_xy(d, truth["T2s"], truth["dw"], a, b)]
          + gate
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
    expected = {"inversion.py": "202c814b2a000943904452a311147b675c95ecffce4f4600865c011bb9c207d1",
                "simulation.py": "fb6b0588d1e5e980d0f5f83d9b55138ed562ec83462dca2d24c240c1f1cd89e2"}
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
                              rb_lengths=[94, 750, 1500], p0=[0.9] * 60, shots=[24] * 60)
               for q, e, s, c in zip([0, 3, 6], eps, sigmas, chi2s)]
    raw = [{"job_id": f"j{q}", "batch": [q], "circuit": k, "counts": {"0": 20, "1": 4}, "probabilities": None}
           for q in (0, 3, 6) for k in range(60)]
    return ionq.IonQReport(backend="ionq_qpu.forte-1", ions=[0, 3, 6], shots=24,
                           rb_lengths=[94, 750, 1500], eps_prior=2e-4, spam=(0.006, 0.004),
                           batches=[[0], [3], [6]], job_ids=["j0", "j3", "j6"],
                           ionq_estimates=[None] * 3, results=results, raw=raw, rb_seed=1)


def test_ionq_works_like_ibm(tmp_path):
    before = canary.Report(ionq.to_canary_report(
        _ion_report([2e-4, 3e-4, 2e-4], [1e-5, 1e-5, float("inf")], [1.0, 8.0, 1.0])))
    assert [q.health for q in before] == ["GREEN", "YELLOW", "RED"]
    assert before[6].health_reason == "gate fit failed" and np.isnan(before[6].eps)
    assert before[3].fit and np.isnan(before[0].t1) and before[0].dd_advice == []
    assert before.raw.shots_per_qubit == 1440 and len(before.raw.block_counts[0]) == 60
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


def test_probe_block_defaults():
    import inspect
    assert runner.DEFAULT_PROBE_BLOCKS == {"superconducting": 1, "trapped_ion": 10}
    for fn in (runner.run_ibm, runner.run_simulation, api.check, api.simulate):
        assert inspect.signature(fn).parameters["probe_blocks"].default is None
    assert inspect.signature(ionq.run_ionq).parameters["probe_blocks"].default == 1


class _FakeIonQ:
    """Records every job; circuit k of job j returns z = (k + j) % 25 zeros out of its shots (<= 24)."""
    name = "fake_ionq"

    def __init__(self):
        self.jobs = []

    def run(self, circuits, **options):
        j, shots = len(self.jobs), options["shots"]
        self.jobs.append((len(circuits), shots))

        class Result:
            def data(self, k):
                return {}

            def get_counts(self, k):
                z = min((k + j) % 25, shots)
                return {"0": z, "1": shots - z}

        class Job:
            def job_id(self):
                return f"job{j}"

            def result(self):
                return Result()
        return Job()


@pytest.mark.parametrize("blocks", [1, 3])
def test_run_ionq_blocks_split_shots_and_sum_counts(blocks):
    backend = _FakeIonQ()
    report = ionq.run_ionq(backend, [5], eps_prior=2e-4, spam=(0.01, 0.01), seed=1, probe_blocks=blocks)
    assert backend.jobs == [(60, 24 // blocks)] * blocks
    assert report.probe_blocks == blocks and report.job_ids == [f"job{b}" for b in range(blocks)]
    assert sorted({e["block"] for e in report.raw}) == list(range(blocks))
    per = 24 // blocks
    position = {k: i for i, k in enumerate(ionq._mixed_order(inv._RB_SEQUENCES))}
    expected = [sum(min((position[k] + b) % 25, per) for b in range(blocks)) / 24 for k in range(60)]
    assert np.allclose(report.results[0].p0, expected)
    assert ionq.to_canary_report(report).probe_blocks == blocks


class _IonQRateCard:
    """IonQ's /jobs/estimate reply as of October 2026: estimated_total_cost from gates x shots at the rate card,
    which also carries the per-job minimum that IonQ applies to each circuit (child job) of a job."""
    def estimate_job(self, oneq_gates, shots, **kwargs):
        from qiskit_ionq.ionq_client import JobEstimate
        return JobEstimate({"estimated_unit": "usd", "estimated_total_cost": round(oneq_gates * shots * 0.000128, 2),
                            "estimated_execution_time": 1000, "rate_card": {"rates": [
                                {"sku": "forte_on_demand", "cost_1q_gate": 0.000128, "job_cost_minimum": 20.0836}]}})


def test_ionq_estimate_prices_each_circuit_with_its_minimum():
    backend = type("B", (), {"name": "qpu.forte-1", "client": _IonQRateCard()})()
    _, one_ion = ionq.ionq_circuits([0], [94, 750, 1500], parallel=False, seed=0)
    e = ionq.ionq_estimate(backend, one_ion, 24)
    assert e["cost_unit"] == "usd" and round(e["cost"], 2) == 1205.02
    _, ten_ions = ionq.ionq_circuits([0, 3, 6, 9, 12, 14, 18, 21, 27, 33], [94, 750, 1500], seed=0)
    e10 = ionq.ionq_estimate(backend, ten_ions, 24)
    assert len(ten_ions) == 60 and 1600 < e10["cost"] < 1900 and min(e10["per_circuit_cost"]) == 20.0836


def test_run_ionq_splits_large_batches_into_jobs_without_changing_the_fit():
    class ByName:
        """Counts that depend only on the circuit (its name), not on how circuits are grouped into jobs."""
        def __init__(self):
            self.jobs = []

        def run(self, circuits, **options):
            self.jobs.append(len(circuits))
            names = [qc.name for qc in circuits]

            class Result:
                def data(self, i):
                    return {}

                def get_counts(self, i):
                    z = 6 + hash(names[i]) % 13
                    return {"0": z, "1": options["shots"] - z}
            return type("J", (), {"job_id": lambda s: f"j{len(names)}", "result": lambda s: Result()})()

    ions = [0, 3, 6, 9, 12, 14, 18, 21, 27, 33]
    one, split = ByName(), ByName()
    a = ionq.run_ionq(one, ions, eps_prior=2e-4, spam=(0.004, 0.003), seed=1, max_job_gates=10 ** 9)
    b = ionq.run_ionq(split, ions, eps_prior=2e-4, spam=(0.004, 0.003), seed=1)
    assert one.jobs == [60] and len(split.jobs) > 1 and sum(split.jobs) == 60
    assert [r.p0 for r in a.results] == [r.p0 for r in b.results]
    assert [r.epsilon_sx for r in a.results] == [r.epsilon_sx for r in b.results]
    _, circuits = ionq.ionq_circuits(ions, [94, 750, 1500], seed=1)
    order = ionq._mixed_order(inv._RB_SEQUENCES)
    assert sorted(order) == list(range(60))
    chunks = ionq._job_chunks([circuits[k] for k in order], ionq.DEFAULT_MAX_JOB_GATES)
    assert all(sum(v for qc in chunk for k, v in qc.count_ops().items() if k.startswith("gpi")) <= ionq.DEFAULT_MAX_JOB_GATES
               for _, chunk in chunks)
    group = lambda k: 0 if k < 10 else 1 if k < 30 else 2
    for start, chunk in chunks:
        share = np.bincount([group(order[start + i]) for i in range(len(chunk))], minlength=3) / len(chunk)
        assert np.allclose(share, [1 / 6, 2 / 6, 3 / 6], atol=0.15)


class _QueuedIonQ:
    """A backend whose jobs finish only when told, are retrievable by id, and return their circuit names."""
    name = "qpu.fake"

    def __init__(self):
        self.jobs = {}

    def run(self, circuits, **options):
        names, shots, jid = [qc.name for qc in circuits], options["shots"], f"job{len(self.jobs)}"

        class Result:
            def to_dict(self):
                return {"results": [{"header": {"name": n}} for n in names]}

            def data(self, i):
                return {}

            def get_counts(self, i):
                z = 4 + sum(map(ord, names[i])) % 17
                return {"0": z, "1": shots - z}

        job = type("Job", (), {"finished": False, "job_id": lambda s: jid, "result": lambda s: Result(),
                               "done": lambda s: s.finished})()
        self.jobs[jid] = job
        return job

    def retrieve_job(self, jid):
        return self.jobs[jid]


def test_submit_then_collect_later_matches_run_ionq():
    import json
    ions, kw = [0, 3, 6, 9, 12, 14, 18, 21, 27, 33], dict(eps_prior=2e-4, spam=(0.004, 0.003), seed=3)
    direct = ionq.run_ionq(_QueuedIonQ(), ions, **kw)
    backend, saved = _QueuedIonQ(), []
    plan = ionq.submit_ionq(backend, ions, on_job=lambda p: saved.append(len(p["jobs"])), **kw)
    assert saved == [1, 2, 3, 4] and len(backend.jobs) == 4
    on_disk = json.loads(json.dumps({**plan, "jobs": [{k: v for k, v in r.items() if k != "_job"} for r in plan["jobs"]]}))
    assert ionq.collect_ionq(backend, on_disk, wait=False) is None
    for job in backend.jobs.values():
        job.finished = True
    later = ionq.collect_ionq(backend, on_disk, wait=False)
    assert [r.p0 for r in later.results] == [r.p0 for r in direct.results]
    assert [r.epsilon_sx for r in later.results] == [r.epsilon_sx for r in direct.results]
    assert later.job_ids == ["job0", "job1", "job2", "job3"] and later.rb_seed == 3
    on_disk["jobs"][0]["names"] = list(reversed(on_disk["jobs"][0]["names"]))
    with pytest.raises(RuntimeError, match="returned circuits"):
        ionq.collect_ionq(backend, on_disk)


def test_ionq_shares_circuits_by_default():
    import inspect
    assert inspect.signature(ionq.run_ionq).parameters["parallel"].default is True
    batches, circuits = ionq.ionq_circuits([0, 3, 6], [94, 750, 1500], seed=0)
    assert batches == [[0, 3, 6]] and len(circuits) == 60


def test_run_ionq_rejects_blocks_not_dividing_shots():
    with pytest.raises(ValueError, match="dividing shots"):
        ionq.run_ionq(_FakeIonQ(), [5], eps_prior=2e-4, spam=(0.01, 0.01), probe_blocks=5)


def test_ionq_backend_detection():
    from qiskit_ionq import IonQProvider
    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
    assert ionq.is_ionq(IonQProvider("offline-test").get_backend("simulator", gateset="native"))
    assert not ionq.is_ionq(FakeProviderForBackendV2().backend("fake_fez"))


def test_credentials_read_files_safely(tmp_path):
    from quantum_canary import credentials
    good = tmp_path / "token.txt"
    good.write_bytes(b"\xef\xbb\xbfABC-123-SECRET \r\n")          # BOM and trailing whitespace
    assert credentials.read_secret(good) == "ABC-123-SECRET"
    for name, content in (("empty.txt", " \n"), ("two.txt", "AAA-SECRET BBB-SECRET")):
        bad = tmp_path / name
        bad.write_text(content)
        with pytest.raises(ValueError) as err:
            credentials.read_secret(bad)
        assert "SECRET" not in str(err.value) and name in str(err.value)
    with pytest.raises(ValueError) as err:
        credentials.read_secret(tmp_path / "missing.txt")
    assert "missing.txt" in str(err.value)


def test_credentials_passed_explicitly_to_vendors(tmp_path, monkeypatch):
    import qiskit_ibm_runtime
    import qiskit_ionq.helpers
    from quantum_canary import credentials
    (tmp_path / "t.txt").write_text("IBM-TOKEN\n")
    (tmp_path / "i.txt").write_text("crn:v1:bluemix:public:quantum-computing:us-east:a/x::\n")
    (tmp_path / "ionq.txt").write_text("IONQ-TOKEN\n")
    calls = []
    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", lambda **kw: calls.append(kw) or "service")
    assert credentials.ibm_service() == "service" and calls[-1] == {}
    credentials.ibm_service(tmp_path / "t.txt", tmp_path / "i.txt")
    assert calls[-1] == {"channel": "ibm_quantum_platform", "token": "IBM-TOKEN",
                         "instance": "crn:v1:bluemix:public:quantum-computing:us-east:a/x::"}
    credentials.ibm_service(instance="my-plan")
    assert calls[-1] == {"instance": "my-plan"}

    for var in ("QISKIT_IONQ_API_TOKEN", "IONQ_API_KEY", "IONQ_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(qiskit_ionq.helpers, "dotenv_values", lambda *a, **k: {})
    assert credentials.ionq_provider(tmp_path / "ionq.txt").credentials["token"] == "IONQ-TOKEN"
    with pytest.raises(ValueError, match="no IonQ API key"):
        credentials.ionq_provider()
    monkeypatch.setenv("IONQ_API_KEY", "FROM-ENV")
    assert credentials.ionq_provider().credentials["token"] == "FROM-ENV"


def test_cli_credential_options_and_no_secret_in_saved_files(tmp_path):
    from quantum_canary.cli import build_parser
    args = build_parser().parse_args(["check", "ibm_kingston", "13", "--ibm-token-file", "t.txt",
                                      "--ibm-instance-file", "i.txt", "--ionq-token-file", "k.txt"])
    assert (args.ibm_token_file, args.ibm_instance_file, args.ionq_token_file) == ("t.txt", "i.txt", "k.txt")
    r = canary.simulate(1, regime="ideal", seed=2)
    path = tmp_path / "saved.json"
    r.save(path)
    text = path.read_text(encoding="utf-8").lower()
    assert "token" not in text and "api_key" not in text and "crn:" not in text
