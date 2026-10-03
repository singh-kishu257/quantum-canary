"""Does checking qubits with Canary first make an error-mitigated result more accurate?

The task is the usual way to validate error mitigation: a small circuit whose
exact answer is known, here a 6-qubit Trotterised transverse-field Ising chain
measured for its average magnetisation, run with zero-noise extrapolation
(CZ folding x1, x3, x5, linear fit) and readout correction.

The device is a simulated 16-qubit line with every noise mechanism of the
simulator switched on, crosstalk between neighbours included, plus a CZ noise
layer built from IBM's ibm_fez calibration snapshot (gate time 84 ns, error
drawn from the snapshot's measured CZ errors, plus T1/T2 relaxation of both
qubits during the gate). Its calibration record is one hour old.

On the same device, starting at the same moment and with the same random
numbers, the task is placed on six neighbouring qubits chosen by:
  calibration         the stale calibration record
  calibration_budget  the same window, with the shots Canary would use added
                      to the task
  canary              Canary's check of all 16 qubits, RED qubits excluded
  canary_dd           as canary, with XX dynamical decoupling on the qubits
                      Canary flags
  oracle              the true parameters (a reference, not a method)
Every arm ranks windows with the same estimated success probability and the
same two-qubit error record; only the single-qubit numbers differ. The truth
is used to score results and to build the oracle, never by an arm. After the
canary job, Canary checks the window again and compare() says whether it
drifted; that verdict is scored against the true change.

The primary endpoint is the paired difference in |ZNE estimate - exact|
between canary and calibration. Known limits: the simulator's decay models
are the ones Canary fits, measurement takes no time, and CZ gates carry no
detuning phase. Output is one CSV.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import logging
import multiprocessing as mp
import pathlib
import sys
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_SRC = SCRIPT_DIR.parent / "src"
if REPO_SRC.is_dir():
    sys.path.insert(0, str(REPO_SRC))

import quantum_canary as canary
from quantum_canary import inversion as inv
from quantum_canary import simulation as sim
from quantum_canary.runner import CanaryReport, QubitReport, _health
from quantum_canary.workflow import _dd_reasons

ARCH = "superconducting"
N_QUBITS = 16
WIDTH = 6
J_DT, H_DT, STEPS = 0.25, 0.35, 4
SCALES = (1, 3, 5)
SHOTS = 4000
T_CALIB_S = 0.0
T_CHECK_S = 3600.0
CZ_NS = 84.0
REP_DELAY_S = 250e-6
PROBE_BLOCKS = 10
DRIFT_TRUTH_REL = 0.20
DT_NS = inv.ARCH_DEFAULTS[ARCH]["dt_ns"]
CANARY_SHOTS_PER_QUBIT = sum(n * canary.DEFAULT_SHOTS[k] for k, n in
                             (("t1", 3), ("ramsey", 8), ("gate", 3), ("echo", 3)))
ARMS = ("calibration", "calibration_budget", "canary", "canary_dd", "oracle")
PAIRS = [(i, i + 1) for i in range(N_QUBITS - 1)]
WINDOWS = [tuple(range(s, s + WIDTH)) for s in range(N_QUBITS - WIDTH + 1)]

CSV_COLUMNS = [
    "instance_id", "seed", "arm", "window", "window_start", "same_as_calibration",
    "exact", "raw_estimate", "zne_estimate", "raw_abs_error", "zne_abs_error",
    "esp_true", "dd_applied", "dd_qubits", "red_qubits", "fallback",
    "shots_task", "shots_canary", "job_start_s", "job_duration_s",
    "drift_flagged", "drift_true", "drift_qubits_flagged", "drift_qubits_true",
    "cz_count", "perturbation_frac", "qiskit_ibm_runtime", "status", "timestamp_utc",
]
NAN = float("nan")

_CZ_ERRORS = None


def _fez_cz_errors() -> np.ndarray:
    """Measured CZ errors of ibm_fez, from qiskit-ibm-runtime's snapshot."""
    global _CZ_ERRORS
    if _CZ_ERRORS is None:
        from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            target = FakeProviderForBackendV2().backend("fake_fez").target
        errs = np.array([p.error for p in target["cz"].values()
                         if p is not None and p.error is not None], dtype=float)
        _CZ_ERRORS = np.sort(errs[(errs > 0) & (errs < 1)])
    return _CZ_ERRORS


def _cz_channel(truth, a: int, b: int, t_s: float):
    """CZ noise on (a, b): thermal relaxation of both qubits over the gate,
    then two-qubit depolarizing with the pair's average infidelity."""
    from qiskit_aer.noise import depolarizing_error, thermal_relaxation_error
    ta, tb = sim.evaluate_truth_at(truth, a, t_s), sim.evaluate_truth_at(truth, b, t_s)
    t_g = CZ_NS * 1e-9
    relax_a = thermal_relaxation_error(ta["T1_s"], min(ta["T2_s"], 2 * ta["T1_s"]), t_g)
    relax_b = thermal_relaxation_error(tb["T1_s"], min(tb["T2_s"], 2 * tb["T1_s"]), t_g)
    e_dep = truth.architecture_specific["cz_dep_error"][(min(a, b), max(a, b))]
    return relax_b.tensor(relax_a).compose(depolarizing_error(4.0 * e_dep / 3.0, 2))


_base_noise_model = sim.build_noise_model


def _noise_model_with_cz(truth, t_s: float = 0.0, native_gate_names=None, circuit=None,
                         dt_ns=None):
    nm = _base_noise_model(truth, t_s=t_s, native_gate_names=native_gate_names,
                           circuit=circuit, dt_ns=dt_ns)
    if "cz_dep_error" in truth.architecture_specific and circuit is not None and any(
            inst.operation.name == "cz" for inst in circuit.data):
        for a, b in PAIRS:
            nm.add_quantum_error(_cz_channel(truth, a, b, t_s), ["cz"], [a, b])
            nm.add_quantum_error(_cz_channel(truth, b, a, t_s), ["cz"], [b, a])
    return nm


sim.build_noise_model = _noise_model_with_cz


def _true_cz_error(truth, a: int, b: int, t_s: float) -> float:
    from qiskit.quantum_info import average_gate_fidelity
    return float(1.0 - average_gate_fidelity(_cz_channel(truth, a, b, t_s).to_quantumchannel()))


def _make_device(seed: int):
    truth = sim.generate_device(ARCH, N_QUBITS, seed=seed, regime="nisq",
                                detuning_1_over_f=True, t1_telegraph=True,
                                spam_spread_frac=0.5, dw_population="literature",
                                quasi_static_dephasing=True, detuning_during_gates=True)
    errs = _fez_cz_errors()
    rng = sim._sub_rng(seed, "online_utility_cz")
    truth.architecture_specific["cz_dep_error"] = {
        p: float(errs[rng.integers(0, len(errs))]) for p in PAIRS}
    return truth


def _calibration_record(truth, seed: int, frac: float):
    priors = {q: sim.generate_calibration_prior(truth, q, T_CALIB_S, perturbation_frac=frac,
                                                spam_perturbation_frac=frac)
              for q in range(N_QUBITS)}
    rng = sim._sub_rng(seed, "online_utility_cz_record")
    cz_record = {p: _true_cz_error(truth, *p, T_CALIB_S) * rng.uniform(1 - frac, 1 + frac)
                 for p in PAIRS}
    return priors, cz_record


def _logical_circuit():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(WIDTH)
    for _ in range(STEPS):
        for i in range(WIDTH - 1):
            qc.rzz(2.0 * J_DT, i, i + 1)
        for i in range(WIDTH):
            qc.rx(2.0 * H_DT, i)
    return qc


def _exact_magnetisation() -> float:
    from qiskit.quantum_info import Statevector
    probs = Statevector(_logical_circuit()).probabilities_dict()
    return float(np.mean([sum(p * (1 - 2 * int(bits[WIDTH - 1 - i])) for bits, p in probs.items())
                          for i in range(WIDTH)]))


def _durations():
    from qiskit.transpiler import InstructionDurations
    one_q = int(round(sim.get_architecture_profile(ARCH).gate_duration_ns / DT_NS))
    return InstructionDurations([("rz", None, 0), ("sx", None, one_q), ("x", None, one_q),
                                 ("cz", None, int(round(CZ_NS / DT_NS))), ("measure", None, 0),
                                 ("reset", None, 0)], dt=DT_NS * 1e-9)


def _physical_circuits(window: Sequence[int], dd_qubits: Sequence[int] = ()):
    """The task on `window`, one circuit per noise scale, scheduled with
    explicit idle delays (and XX decoupling on dd_qubits)."""
    from qiskit import transpile
    from qiskit.circuit.library import XGate
    from qiskit.transpiler import CouplingMap, PassManager
    from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay, PadDynamicalDecoupling
    logical = _logical_circuit()
    logical.measure_all()
    base = transpile(logical, coupling_map=CouplingMap.from_line(N_QUBITS),
                     basis_gates=["rz", "sx", "x", "cz"], initial_layout=list(window),
                     optimization_level=1, seed_transpiler=0)
    durations = _durations()
    out = []
    for scale in SCALES:
        folded = base.copy_empty_like()
        for inst in base.data:
            folded.append(inst)
            if inst.operation.name == "cz":
                for _ in range(scale - 1):
                    folded.append(inst)
        if dd_qubits:
            pad = PadDynamicalDecoupling(durations=durations, dd_sequence=[XGate(), XGate()],
                                         qubits=list(dd_qubits))
        else:
            pad = PadDelay(durations=durations)
        scheduled = PassManager([ALAPScheduleAnalysis(durations), pad]).run(folded)
        keep = set(window)
        trimmed = scheduled.copy_empty_like()
        for inst in scheduled.data:
            qs = {scheduled.find_bit(q).index for q in inst.qubits}
            if inst.operation.name == "delay" and not qs <= keep:
                continue
            trimmed.append(inst)
        trimmed.name = f"task_x{scale}"
        out.append(trimmed)
    return out


def _window_bits(circ, window: Sequence[int]) -> List[int]:
    """Classical bit measuring each window qubit, in window order."""
    where = {}
    for inst in circ.data:
        if inst.operation.name == "measure":
            where[circ.find_bit(inst.qubits[0]).index] = circ.find_bit(inst.clbits[0]).index
    return [where[q] for q in window]


def _verify_circuits(circuits, window, exact: float) -> None:
    from qiskit.quantum_info import Statevector
    for c in circuits:
        bare = c.copy_empty_like()
        for inst in c.data:
            if inst.operation.name not in ("measure", "barrier", "delay"):
                bare.append(inst)
        probs = Statevector(bare).probabilities_dict(qargs=list(window))
        m = np.mean([sum(p * (1 - 2 * int(bits[WIDTH - 1 - i])) for bits, p in probs.items())
                     for i in range(WIDTH)])
        if abs(m - exact) > 1e-9:
            raise RuntimeError(f"{c.name} on {window} does not reproduce the exact answer")


def _run_task(circuits, truth, shots: int, t_s: float, seed: int):
    """Run circuits back to back from t_s with the simulator's time-resolved
    execution, keeping full bitstrings. Returns (counts list, duration)."""
    shots_list = [int(shots)] * len(circuits)
    schedule = sim._probe_schedule(circuits, truth, shots_list, t_s, DT_NS, REP_DELAY_S,
                                   sim._DEFAULT_MAX_CHUNKS_PER_CIRCUIT)
    counts_list = []
    for i, qc in enumerate(circuits):
        circ_seed = seed + i
        raw: Dict[str, int] = {}
        for k, chunk in enumerate(schedule[i]["chunks"]):
            ch_seed = circ_seed if k == 0 else int(
                sim._sub_rng(circ_seed, "aer_chunk", k).integers(0, 2**31 - 1))
            snap, _ = sim._truth_snapshot(truth, chunk["t_mid_s"])
            part = sim.simulate_circuit(qc, snap, shots=chunk["shots"], t_s=chunk["t_mid_s"],
                                        seed=ch_seed, dt_ns=DT_NS)
            for key, v in part.items():
                raw[key.replace(" ", "")] = raw.get(key.replace(" ", ""), 0) + v
        counts_list.append(raw)
    return counts_list, float(schedule[-1]["t_end_s"] - t_s)


def _magnetisation(counts: Dict[str, int], bits: Sequence[int], spam: Sequence[Tuple[float, float]]):
    """Average <Z> over the window, each qubit corrected for its readout error."""
    total = sum(counts.values())
    zs = []
    for bit, (p01, p10) in zip(bits, spam):
        p1 = sum(v for key, v in counts.items() if key[-1 - bit] == "1") / total
        p1_true = (p1 - p10) / (1.0 - p01 - p10)
        zs.append(1.0 - 2.0 * p1_true)
    return float(np.mean(zs))


def _probe_qubits(truth, priors, qubits: Sequence[int], t_s: float, seed: int):
    """Canary on each qubit in turn, embedded in the 16-qubit device.
    Returns ({qubit: QubitReport}, {qubit: (t_start, t_end)}, t_end)."""
    from qiskit import QuantumCircuit
    reports, windows = {}, {}
    for q in qubits:
        prior = priors[q]
        constants = dict(inv.ARCH_DEFAULTS[ARCH])
        p01 = float(np.clip(prior.spam_p0_given_1_prior, 0.0, 0.499))
        p10 = float(np.clip(prior.spam_p1_given_0_prior, 0.0, 0.499))
        constants.update(T1_s=prior.T1_prior_s, T2_s=prior.T2_prior_s,
                         eps_typical=prior.epsilon_prior, p0_given_1=p01, p1_given_0=p10)
        profile = inv.BackendProfile(architecture=ARCH, T1_prior_s=prior.T1_prior_s,
                                     T2_prior_s=prior.T2_prior_s, dt_ns=DT_NS,
                                     backend_name="online_utility_sim", custom_arch=constants,
                                     prior_confidence="prior_informed")
        circuits, meta = inv.build_probe_circuits(profile)
        embedded = []
        for c in circuits:
            e = QuantumCircuit(N_QUBITS, 1, name=c.name)
            e.compose(c, qubits=[q], clbits=[0], inplace=True)
            embedded.append(e)
        shots = canary.DEFAULT_SHOTS
        shots_list = ([shots["t1"]] * meta["n_t1"] + [shots["ramsey"]] * meta["n_ramsey"]
                      + [shots["gate"]] * meta["n_gate"] + [shots["echo"]] * meta["n_echo"])
        blocks, block_shots, index = inv.interleave_probe_blocks(embedded, shots_list, PROBE_BLOCKS)
        res = sim.simulate_probe(blocks, truth, shots=block_shots, t_s=t_s, dt_ns=DT_NS,
                                 seed=seed + 1000 * q, time_resolved=True,
                                 rep_delay_s=REP_DELAY_S)
        counts = inv.merge_block_counts(res.counts_list, index, len(embedded))
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = inv.lindblad_inversion(counts, meta, profile, shots_t1=shots["t1"],
                                            shots_ramsey=shots["ramsey"], shots_gate=shots["gate"],
                                            shots_echo=shots["echo"], qubit_id=q)
        reports[q] = QubitReport(qubit=q, result=result, profile=profile, health=_health(q, result))
        windows[q] = (t_s, t_s + res.run_duration_s)
        t_s += res.run_duration_s
    return reports, windows, t_s


_PROFILE_CACHE: Dict[Tuple[int, ...], Tuple[Dict[int, int], Dict[Tuple[int, int], int], float]] = {}


def _window_profile(window: Tuple[int, ...]):
    """Single-qubit gate counts per qubit, CZ counts per pair and the run
    time of the task's x1 circuit on `window` (the same for every arm)."""
    if window not in _PROFILE_CACHE:
        circuit = _physical_circuits(window)[0]
        dt_of = _durations()
        one_q = {q: 0 for q in window}
        cz: Dict[Tuple[int, int], int] = {}
        busy = {q: 0 for q in window}
        for inst in circuit.data:
            qs = [circuit.find_bit(x).index for x in inst.qubits]
            name = inst.operation.name
            if name in ("sx", "x"):
                one_q[qs[0]] += 1
            elif name == "cz":
                key = (min(qs), max(qs))
                cz[key] = cz.get(key, 0) + 1
            if name == "delay":
                length = int(inst.operation.params[0])
            elif name in ("sx", "x", "cz"):
                length = dt_of.get(name, qs)
            else:
                length = 0
            for q in qs:
                if q in busy:
                    busy[q] += length
        _PROFILE_CACHE[window] = (one_q, cz, max(busy.values()) * DT_NS * 1e-9)
    return _PROFILE_CACHE[window]


def _log_esp(window: Tuple[int, ...], single: Dict[int, Dict[str, float]],
             cz_error: Dict[Tuple[int, int], float]) -> float:
    """Log estimated success probability of the task on `window`: CZ and
    single-qubit gate errors, decay over the run time, and readout error.
    Every arm uses this formula; only the numbers plugged in differ."""
    one_q, cz, t_c = _window_profile(window)
    total = sum(n * np.log(max(1.0 - cz_error[pair], 1e-12)) for pair, n in cz.items())
    for q, n in one_q.items():
        s = single[q]
        if not all(np.isfinite([s["T1"], s["T2"], s["eps"]])) or s["T1"] <= 0 or s["T2"] <= 0:
            return -np.inf
        total += n * np.log(max(1.0 - s["eps"], 1e-12))
        total += -t_c / s["T1"] - t_c / s["T2"] + np.log(max(1.0 - s["readout"], 1e-12))
    return float(total)


def _best_window(score) -> Tuple[int, ...]:
    best = max(WINDOWS, key=score)
    return best if np.isfinite(score(best)) else None


def run_one_instance(instance_id: int, seed: int, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not 0 <= seed < 20_000:
        raise ValueError("seed must be in [0, 20000) so Aer seeds stay below 2**31")
    timestamp = datetime.now(timezone.utc).isoformat()
    frac = float(cfg["perturbation_frac"])
    shots = int(cfg["shots"])
    try:
        import qiskit_ibm_runtime
        qir = qiskit_ibm_runtime.__version__
    except Exception:
        qir = ""
    base_row = dict.fromkeys(CSV_COLUMNS, NAN)
    base_row.update(instance_id=instance_id, seed=seed, perturbation_frac=frac,
                    qiskit_ibm_runtime=qir, timestamp_utc=timestamp)
    try:
        truth = _make_device(seed)
        priors, cz_record = _calibration_record(truth, seed, frac)
        exact = _exact_magnetisation()

        cz_count = sum(_window_profile(WINDOWS[0])[1].values())

        def score_with(single, cz_table):
            return lambda window: _log_esp(window, single, cz_table)

        calib_single = {q: {"T1": priors[q].T1_prior_s, "T2": priors[q].T2_prior_s,
                            "eps": priors[q].epsilon_prior,
                            "readout": 0.5 * (priors[q].spam_p0_given_1_prior
                                              + priors[q].spam_p1_given_0_prior)}
                        for q in range(N_QUBITS)}
        w_cal = _best_window(score_with(calib_single, cz_record))

        before, probe_windows, t_job = _probe_qubits(truth, priors, range(N_QUBITS), T_CHECK_S,
                                                     seed * 100_000)
        red = sorted(q for q, r in before.items() if r.health == "RED")
        canary_single = {}
        for q, r in before.items():
            res = r.result
            ok = r.health != "RED"
            canary_single[q] = {"T1": res.T1_s if ok else NAN, "T2": res.T2_s if ok else NAN,
                                "eps": res.epsilon_sx if ok else NAN,
                                "readout": calib_single[q]["readout"]}
        w_can = _best_window(score_with(canary_single, cz_record))
        fallback = w_can is None
        if fallback:
            w_can = w_cal

        true_single = {}
        for q in range(N_QUBITS):
            tv = sim.evaluate_truth_at(truth, q, t_job)
            true_single[q] = {"T1": tv["T1_s"], "T2": tv["T2_s"],
                              "eps": sim.effective_gate_epsilon(truth, q, t_job),
                              "readout": 0.5 * (tv["spam_p0_given_1"] + tv["spam_p1_given_0"])}
        true_cz = {p: _true_cz_error(truth, *p, t_job) for p in PAIRS}

        true_score = score_with(true_single, true_cz)
        w_orc = max(WINDOWS, key=true_score)

        dd_q = [q for q in w_can if _dd_reasons(before[q])]
        plans = {"calibration": (w_cal, (), shots),
                 "calibration_budget": (w_cal, (), shots + int(round(
                     N_QUBITS * CANARY_SHOTS_PER_QUBIT / len(SCALES)))),
                 "canary": (w_can, (), shots),
                 "canary_dd": (w_can, tuple(dd_q), shots),
                 "oracle": (w_orc, (), shots)}

        rows, durations = [], {}
        for arm in ARMS:
            window, dd, n_shots = plans[arm]
            circuits = _physical_circuits(window, dd)
            _verify_circuits(circuits, window, exact)
            counts, duration = _run_task(circuits, truth, n_shots, t_job, seed * 100_000 + 90_000)
            durations[arm] = duration
            spam = [(priors[q].spam_p0_given_1_prior, priors[q].spam_p1_given_0_prior) for q in window]
            values = [_magnetisation(c, _window_bits(circ, window), spam)
                      for c, circ in zip(counts, circuits)]
            zne = float(np.polyval(np.polyfit(SCALES, values, 1), 0.0))
            row = dict(base_row)
            row.update(arm=arm, window="-".join(map(str, window)), window_start=window[0],
                       same_as_calibration=bool(window == w_cal), exact=exact,
                       raw_estimate=values[0], zne_estimate=zne,
                       raw_abs_error=abs(values[0] - exact), zne_abs_error=abs(zne - exact),
                       esp_true=float(np.exp(true_score(window))), dd_applied=bool(dd),
                       dd_qubits="-".join(map(str, dd)), red_qubits="-".join(map(str, red)),
                       fallback=bool(fallback) if arm.startswith("canary") else False,
                       shots_task=n_shots * len(SCALES),
                       shots_canary=N_QUBITS * CANARY_SHOTS_PER_QUBIT if arm.startswith("canary") else 0,
                       job_start_s=t_job, job_duration_s=duration, cz_count=cz_count,
                       status="ok")
            rows.append(row)

        after, after_windows, _ = _probe_qubits(truth, priors, w_can, t_job + durations["canary"],
                                                seed * 100_000 + 50_000)
        before_report = CanaryReport(mode="simulation", backend="online_utility_sim",
                                     architecture=ARCH, shots=dict(canary.DEFAULT_SHOTS),
                                     probe_blocks=PROBE_BLOCKS, circuits_per_qubit=17,
                                     qubits=[before[q] for q in w_can], timestamp=timestamp)
        after_report = CanaryReport(mode="simulation", backend="online_utility_sim",
                                    architecture=ARCH, shots=dict(canary.DEFAULT_SHOTS),
                                    probe_blocks=PROBE_BLOCKS, circuits_per_qubit=17,
                                    qubits=[after[q] for q in w_can], timestamp=timestamp)
        drift = canary.compare(before_report, after_report)
        changed = []
        for q in w_can:
            b0, b1 = probe_windows[q]
            a0, a1 = after_windows[q]
            tb = sim.evaluate_truth_at(truth, q, 0.5 * (b0 + b1))
            ta = sim.evaluate_truth_at(truth, q, 0.5 * (a0 + a1))
            if any(abs(ta[k] - tb[k]) / tb[k] > DRIFT_TRUTH_REL for k in ("T1_s", "T2_s")):
                changed.append(q)
        for row in rows:
            if row["arm"] == "canary":
                row.update(drift_flagged=not drift.stable, drift_true=bool(changed),
                           drift_qubits_flagged="-".join(map(str, drift.drifted_qubits)),
                           drift_qubits_true="-".join(map(str, changed)))
        return rows
    except Exception as exc:
        rows = []
        for arm in ARMS:
            row = dict(base_row)
            row.update(arm=arm, status=f"exception:{type(exc).__name__}:{exc}")
            rows.append(row)
        return rows


def _worker(job):
    warnings.filterwarnings("ignore")
    logging.getLogger("qiskit_aer.noise.noise_model").setLevel(logging.ERROR)
    return run_one_instance(*job)


def run(n_instances: int, instance_start: int, n_workers: int, cfg: Dict[str, Any]):
    jobs = [(i, i, cfg) for i in range(instance_start, instance_start + n_instances)]
    print(f"[online_utility] instances {instance_start}..{instance_start + n_instances - 1}, "
          f"{n_workers} workers", flush=True)
    if n_workers <= 1:
        results = [_worker(j) for j in jobs]
    else:
        with mp.Pool(n_workers) as pool:
            results = list(pool.imap(_worker, jobs, chunksize=1))
    return [row for rows in results for row in rows]


def write_csv(rows, path: pathlib.Path) -> None:
    rank = {a: i for i, a in enumerate(ARMS)}
    rows = sorted(rows, key=lambda r: (r["instance_id"], rank[r["arm"]]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"[online_utility] wrote {len(rows)} rows -> {path}", flush=True)


def read_csv(path: pathlib.Path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(f"{path.name}: unexpected columns")
        rows = []
        for raw in reader:
            row = dict(raw)
            for k in ("instance_id", "seed"):
                row[k] = int(row[k])
            for k in ("exact", "raw_estimate", "zne_estimate", "raw_abs_error", "zne_abs_error",
                      "esp_true", "job_start_s", "job_duration_s", "perturbation_frac",
                      "window_start", "shots_task", "shots_canary", "cz_count"):
                row[k] = float(row[k]) if row[k] not in ("", "nan") else NAN
            for k in ("same_as_calibration", "dd_applied", "fallback", "drift_flagged", "drift_true"):
                row[k] = {"true": True, "false": False}.get(str(row[k]).lower(), NAN)
            rows.append(row)
    return rows


def validate(rows, n_instances: int, instance_start: int = 0) -> List[str]:
    problems = []
    ids = sorted({r["instance_id"] for r in rows})
    if ids != list(range(instance_start, instance_start + n_instances)):
        problems.append("instance ids are not contiguous")
    if len(rows) != n_instances * len(ARMS):
        problems.append(f"{len(rows)} rows, expected {n_instances * len(ARMS)}")
    keys = [(r["instance_id"], r["arm"]) for r in rows]
    if len(set(keys)) != len(keys):
        problems.append("duplicate (instance, arm) rows")
    bad = [r for r in rows if r["status"] != "ok"]
    if bad:
        problems.append(f"{len(bad)} rows failed, e.g. {bad[0]['status'][:120]}")
    return problems


def summarize(rows) -> None:
    from scipy.stats import wilcoxon
    ok = [r for r in rows if r["status"] == "ok"]
    by = {}
    for r in ok:
        by.setdefault(r["instance_id"], {})[r["arm"]] = r
    inst = [v for v in by.values() if len(v) == len(ARMS)]
    print(f"\n[online_utility] {len(inst)} complete instances (failures counted above, none dropped)")
    print(f"  exact answer {inst[0]['calibration']['exact']:+.5f} (same task every instance)")
    print(f"  {'arm':<20}{'median |ZNE err|':>17}{'mean':>10}{'median |raw err|':>18}"
          f"{'median ESP_true':>17}{'task shots':>12}{'canary shots':>14}")
    for arm in ARMS:
        z = np.array([v[arm]["zne_abs_error"] for v in inst])
        raw = np.array([v[arm]["raw_abs_error"] for v in inst])
        esp = np.array([v[arm]["esp_true"] for v in inst])
        print(f"  {arm:<20}{np.median(z):>17.4f}{np.mean(z):>10.4f}{np.median(raw):>18.4f}"
              f"{np.median(esp):>17.4f}{int(np.median([v[arm]['shots_task'] for v in inst])):>12,}"
              f"{int(np.median([v[arm]['shots_canary'] for v in inst])):>14,}")
    rng = np.random.default_rng(0)
    print("\n  paired comparisons of |ZNE error| (negative = first arm more accurate)")
    for a, b in (("canary", "calibration"), ("canary", "calibration_budget"),
                 ("canary_dd", "calibration"), ("canary_dd", "canary"), ("oracle", "calibration")):
        d = np.array([v[a]["zne_abs_error"] - v[b]["zne_abs_error"] for v in inst])
        boot = [np.median(rng.choice(d, len(d))) for _ in range(2000)]
        nz = d[d != 0]
        p = wilcoxon(nz).pvalue if len(nz) > 0 else NAN
        print(f"  {a:>10} - {b:<19} median {np.median(d):+.4f}  95% CI [{np.percentile(boot, 2.5):+.4f},"
              f" {np.percentile(boot, 97.5):+.4f}]  {a} better in {np.mean(d < 0):.0%}, tied "
              f"{np.mean(d == 0):.0%}  Wilcoxon p = {p:.3g}")
    moved = np.mean([not v["canary"]["same_as_calibration"] for v in inst])
    red = np.mean([bool(v["canary"]["red_qubits"]) for v in inst])
    dd = np.mean([bool(v["canary_dd"]["dd_applied"]) for v in inst])
    print(f"\n  Canary chose a different window in {moved:.0%} of instances; found a RED qubit in "
          f"{red:.0%}; applied decoupling in {dd:.0%}")
    flag = np.array([v["canary"]["drift_flagged"] for v in inst], dtype=bool)
    true = np.array([v["canary"]["drift_true"] for v in inst], dtype=bool)
    print(f"  drift after the canary job (truth: T1 or T2 moved > {DRIFT_TRUTH_REL:.0%}): "
          f"flagged&true {np.sum(flag & true)}, flagged&false {np.sum(flag & ~true)}, "
          f"missed {np.sum(~flag & true)}, quiet&stable {np.sum(~flag & ~true)}")
    durs = np.array([v["canary"]["job_duration_s"] for v in inst])
    print(f"  task run length median {np.median(durs):.1f} s (repetition delay {REP_DELAY_S * 1e6:.0f} us)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Canary before an error-mitigated task: does it help?")
    ap.add_argument("--n-instances", type=int, default=100)
    ap.add_argument("--instance-start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--shots", type=int, default=SHOTS, help="shots per task circuit")
    ap.add_argument("--perturbation-frac", type=float, default=None,
                    help="staleness of the calibration record (default: the simulator's 0.25)")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--input", type=pathlib.Path, action="append", default=[])
    args = ap.parse_args()
    if args.combine:
        rows = [r for p in args.input for r in read_csv(p)]
        problems = validate(rows, args.n_instances)
        write_csv(rows, args.out)
        summarize(rows)
        _finish(problems)
        return
    frac = (sim.resolve_calibration_perturbation(ARCH, "nisq")
            if args.perturbation_frac is None else args.perturbation_frac)
    if not 0.0 <= frac < 1.0:
        sys.exit("--perturbation-frac must be in [0, 1)")
    cfg = {"perturbation_frac": frac, "shots": args.shots}
    n_workers = args.workers if args.workers is not None else max(1, mp.cpu_count() - 1)
    rows = run(args.n_instances, args.instance_start, n_workers, cfg)
    problems = validate(rows, args.n_instances, args.instance_start)
    write_csv(rows, args.out)
    if not problems:
        summarize(rows)
    _finish(problems)


def _finish(problems: List[str]) -> None:
    if problems:
        print("\n[online_utility] VALIDATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\n[online_utility] validation passed")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.getLogger("qiskit_aer.noise.noise_model").setLevel(logging.ERROR)
    if sys.platform != "win32":
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()
