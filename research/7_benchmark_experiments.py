import importlib.util
import multiprocessing as mp
import pathlib
import sys
import csv
import time
import warnings
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("inversion", SCRIPT_DIR / "1_inversion.py")
inv = importlib.util.module_from_spec(_spec)
sys.modules["inversion"] = inv
_spec.loader.exec_module(inv)

DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

SEED               = 48
N_INSTANCES        = 200
N_BOOTSTRAP        = 2000
R2_THRESHOLD       = 0.95

CANARY_SHOTS_T1     = 300
CANARY_SHOTS_RAMSEY = 1000
CANARY_SHOTS_GATE   = 500
CANARY_SHOTS_ECHO   = 500
CANARY_TOTAL        = (3 * CANARY_SHOTS_T1 +
                       6 * CANARY_SHOTS_RAMSEY +
                       3 * CANARY_SHOTS_GATE +
                       3 * CANARY_SHOTS_ECHO)

QE_N_DELAYS        = 15
QE_RB_LENGTHS      = [1, 2, 4, 8, 16, 32, 64, 128]
QE_RB_SAMPLES      = 5
QE_SX_PER_CLIFFORD = 1.875

QE_DW_PLAUSIBILITY_MULT  = 2.0
QE_T1_PLAUSIBILITY_MULT  = 5.0
QE_EPS_PLAUSIBILITY_MULT = 5.0

QE_GUARD_REJECTS = defaultdict(lambda: defaultdict(int))
QE_GUARD_TOTAL   = defaultdict(lambda: defaultdict(int))

import os as _os

_ALL_BUDGETS = [1_000, 2_500, 5_000, 9_900, 15_000, 25_000, 50_000]
_budget_env  = _os.environ.get("BUDGET_SUBSET", "").strip()
if _budget_env:
    SHOT_BUDGETS = [int(b.strip()) for b in _budget_env.split(",") if b.strip()]
else:
    SHOT_BUDGETS = _ALL_BUDGETS

ALL_ARCHITECTURES = ["superconducting", "trapped_ion", "neutral_atom"]
ARCH = _os.environ.get("ARCH", "superconducting").strip()
if ARCH not in ALL_ARCHITECTURES:
    sys.exit(f"ARCH env var must be one of {ALL_ARCHITECTURES}, got {ARCH!r}")

JOB_TAG = _os.environ.get("JOB_TAG", f"full{ARCH}")

TRUE_PARAM_RANGES = {
    "superconducting": {
        "T1_s": (80e-6,  400e-6),
        "T2_s": (40e-6,  200e-6),
        "eps":  (1e-4,   2e-3),
    },
    "trapped_ion": {
        "T1_s": (100.0,  10000.0),
        "T2_s": (0.1,    3.0),
        "eps":  (1e-4,   2e-3),
    },
    "neutral_atom": {
        "T1_s": (1.0,    100.0),
        "T2_s": (0.3,    3.0),
        "eps":  (1e-3,   1e-2),
    },
}

PARAMS = [
    ("T1",          0),
    ("T2",          1),
    ("delta_omega", 2),
    ("epsilon_sx",  3),
]

BENCH_COLS = [
    "budget",  "architecture",  "parameter",  "method",
    "r2",  "r2_lo",  "r2_hi",
    "nrmse",  "n_valid",  "n_total",  "mean_time_s",
]

THRESH_COLS = ["architecture", "parameter", "method", "shots_to_R2_095", "shots_to_R2_099"]


def sample_instance(rng, arch_name):
    ranges = TRUE_PARAM_RANGES[arch_name]
    T1_lo, T1_hi = ranges["T1_s"]
    T2_lo, T2_hi = ranges["T2_s"]
    eps_lo, eps_hi = ranges["eps"]
    if T1_hi / T1_lo > 10.0:
        T1 = float(10 ** rng.uniform(np.log10(T1_lo), np.log10(T1_hi)))
    else:
        T1 = float(rng.uniform(T1_lo, T1_hi))
    T_phi = float(rng.uniform(T2_lo, T2_hi))
    T2    = min(1.0 / (1.0 / (2.0 * T1) + 1.0 / T_phi), 2.0 * T1)
    eps   = float(10 ** rng.uniform(np.log10(eps_lo), np.log10(eps_hi)))
    dw_max = inv.BackendProfile.from_architecture(arch_name).dw_max_rad_s
    dw     = float(rng.choice([-1, 1]) * rng.uniform(0.2 * dw_max, dw_max))
    return T1, T2, dw, eps


REALISTIC_NOISE = {
    "superconducting": {
        "sigma_T1_lognormal": 0.15,
        "T2_reduction_max":   0.12,
        "sigma_coherent_rad": 0.010,
        "sigma_SPAM_frac":    0.05,
    },
    "trapped_ion": {
        "sigma_T1_lognormal": 0.00,
        "T2_reduction_max":   0.10,
        "sigma_coherent_rad": 0.005,
        "sigma_SPAM_frac":    0.03,
    },
    "neutral_atom": {
        "sigma_T1_lognormal": 0.03,
        "T2_reduction_max":   0.15,
        "sigma_coherent_rad": 0.015,
        "sigma_SPAM_frac":    0.08,
    },
}


def _perturb_instance(arch_name, T1, T2, eps, p0g1_nom, p1g0_nom, rng):
    """Apply architecture-specific unmodeled hardware physics to one instance.

    Returns effective (T1_eff, T2_eff, eps_eff, p0g1_eff, p1g0_eff) that
    are passed to BOTH run_canary and run_qe, so both methods see identical
    simulated hardware and neither is advantaged or disadvantaged.

    Neither method knows the effective values — Canary uses
    BackendProfile.from_architecture() (arch-typical prior) and QE uses
    arch-typical delay grids. Both are equally blind to the drift.

    Perturbations:
      T1  : log-normal TLS fluctuator (Klimov et al. PRL 2018;
                                        Carroll et al. npj QI 2022)
      T2  : uniform fractional reduction from quasi-static 1/f dephasing
            (same references; also micromotion for trapped-ion,
             laser phase noise for neutral-atom)
      eps : coherent over-rotation delta ~ N(0, sigma_coh), adds
            eps_coh = delta^2/4 to depolarising (Rol et al. PRL 2019,
            first-order Magnus)
      SPAM: calibration drift, multiplicative fractional noise
            (Bultink et al. PRApplied 2018)

    No dw perturbation: sigma_dw interacts with Canary's hard arctan2
    bounds in a way that depends on the per-instance profile dw_max,
    which varies with T2_prior and cannot be safely bounded by a fixed
    sigma. Omitting it is the correct choice — dw recovery is already
    characterised in the ideal Markovian scenario.
    """
    arch = inv.ARCH_DEFAULTS[arch_name]
    cfg  = REALISTIC_NOISE[arch_name]
    if cfg["sigma_T1_lognormal"] > 0.0:
        T1_eff = float(np.clip(
            T1 * np.exp(rng.normal(0.0, cfg["sigma_T1_lognormal"])),
            arch["T1_min_s"], arch["T1_max_s"]))
    else:
        T1_eff = T1
    T2_eff = float(np.clip(
        T2 * (1.0 - rng.uniform(0.0, cfg["T2_reduction_max"])),
        arch["T2_min_s"], min(2.0 * T1_eff, arch["T1_max_s"])))
    delta_coh = rng.normal(0.0, cfg["sigma_coherent_rad"])
    eps_eff   = float(np.clip(eps + delta_coh ** 2 / 4.0, 0.0, arch["eps_max"]))
    p0g1_eff = float(np.clip(
        p0g1_nom * (1.0 + rng.normal(0.0, cfg["sigma_SPAM_frac"])), 0.0, 0.30))
    p1g0_eff = float(np.clip(
        p1g0_nom * (1.0 + rng.normal(0.0, cfg["sigma_SPAM_frac"])), 0.0, 0.30))
    return T1_eff, T2_eff, eps_eff, p0g1_eff, p1g0_eff


def _convert_native_to_unitary(qc, native_gate_names):
    from qiskit.circuit.library import UnitaryGate
    new_qc = qc.copy_empty_like()
    for inst in qc.data:
        op = inst.operation
        if op.name in native_gate_names:
            new_qc.append(UnitaryGate(op.to_matrix(), label=op.name),
                          inst.qubits, inst.clbits)
        else:
            new_qc.append(inst)
    return new_qc


def _make_backend(qc, T1, T2, eps, p0g1, p1g0, gate_time_ns, dt_ns,
                  noise_gate_names=None):
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (NoiseModel, thermal_relaxation_error,
                                  depolarizing_error, ReadoutError)
    nm   = NoiseModel()
    gate_time_s = gate_time_ns * 1e-9
    sx_e = (thermal_relaxation_error(T1, T2, gate_time_s)
            .compose(depolarizing_error(2.0 * eps, 1)))
    names = noise_gate_names or ["sx", "x", "id", "rx", "u1", "p"]
    nm.add_quantum_error(sx_e, names, [0])
    seen = set()
    for inst in qc.data:
        if inst.operation.name == "delay":
            ds = inv._delay_seconds(inst, dt_ns)
            key = round(ds, 15)
            if key not in seen and ds > 1e-12:
                seen.add(key)
                nm.add_quantum_error(
                    thermal_relaxation_error(T1, T2, ds), ["delay"], [0])
    nm.add_readout_error(
        ReadoutError([[1 - p0g1, p0g1], [p1g0, 1 - p1g0]]), [0])
    return AerSimulator(noise_model=nm)


def _run_exp(exp_obj, T1, T2, eps, p0g1, p1g0, shots, gate_time_ns, dt_ns,
             dw_true=0.0, inject_detuning=False, native_target=None,
             native_gate_names=None):
    from qiskit import transpile
    exp_obj.set_run_options(shots=shots)
    circuits = exp_obj.circuits()
    exp_data = exp_obj._initialize_experiment_data()
    for qc in circuits:
        run_qc = (inv._inject_ramsey_detuning(qc, dw_true, dt_ns)
                  if inject_detuning else qc)
        noise_gate_names = None
        if native_target is not None:
            run_qc = transpile(run_qc, target=native_target, optimization_level=1)
            run_qc = _convert_native_to_unitary(run_qc, native_gate_names)
            noise_gate_names = ["unitary"]
        b    = _make_backend(run_qc, T1, T2, eps, p0g1, p1g0, gate_time_ns,
                             dt_ns, noise_gate_names=noise_gate_names)
        tqc  = run_qc if native_target is not None else transpile(run_qc, b, optimization_level=0)
        cnts = b.run(tqc, shots=shots).result().get_counts()
        meta = {**dict(qc.metadata), "count_ops": [(((0,), g), c) for g, c in tqc.count_ops().items()]}
        exp_data.add_data({"counts": cnts, "shots": shots, "metadata": meta})
    exp_obj.analysis.run(exp_data).block_for_results()
    return exp_data


def _extract(exp_data, name):
    try:
        df  = exp_data.analysis_results(dataframe=True)
        row = df[df["name"] == name]
        if row.empty:
            return float("nan"), float("nan")
        val = row["value"].iloc[0]
        nom = float(val.nominal_value)
        std = float(val.std_dev)
        std = std if np.isfinite(std) and std > 0 else float("nan")
        return nom, std
    except Exception:
        return float("nan"), float("nan")


def _ionq_native_target(gate_time_ns: float):
    try:
        from qiskit.transpiler import Target, InstructionProperties
        from qiskit.circuit import Delay, Measure, Parameter
        from qiskit_ionq import GPIGate, GPI2Gate, add_equivalences
    except Exception:
        return None
    add_equivalences()  # idempotent; registers IonQ gate equivalences globally
    target = Target(num_qubits=1)
    theta = Parameter("theta")
    dur = gate_time_ns * 1e-9
    target.add_instruction(GPIGate(theta),  {(0,): InstructionProperties(duration=dur)})
    target.add_instruction(GPI2Gate(theta), {(0,): InstructionProperties(duration=dur)})
    target.add_instruction(Delay(Parameter("t")), {(0,): None})
    target.add_instruction(Measure(), {(0,): InstructionProperties(duration=dur)})
    return target


def _native_gates_per_clifford(rb_circuits, native_target):
    from qiskit import transpile
    from qiskit.transpiler.exceptions import TranspilerError
    weighted_gates = 0.0
    weighted_cliffords = 0.0
    for qc in rb_circuits:
        n_cliff = qc.metadata.get("xval")
        if not n_cliff:
            continue
        try:
            tqc = transpile(qc, target=native_target, optimization_level=1)
        except TranspilerError:
            continue
        counts = tqc.count_ops()
        n_native = counts.get("gpi", 0) + counts.get("gpi2", 0)
        weighted_gates      += n_native
        weighted_cliffords  += n_cliff
    if weighted_cliffords <= 0:
        return float("nan")
    return weighted_gates / weighted_cliffords


def _qe_backend(arch_name: str = "superconducting"):
    try:
        from qiskit.providers.fake_provider import GenericBackendV2
        return GenericBackendV2(
            num_qubits=1,
            basis_gates=["sx", "x", "rz", "measure", "delay"],
            seed=0)
    except Exception:
        from qiskit_ibm_runtime.fake_provider import FakeNairobiV2
        return FakeNairobiV2()


def run_canary(T1_true, T2_true, dw_true, eps_true, seed, budget, arch_name,
               p0g1_eff=None, p1g0_eff=None):
    arch    = inv.ARCH_DEFAULTS[arch_name]
    profile = inv.BackendProfile.from_architecture(arch_name)
    p0g1    = p0g1_eff if p0g1_eff is not None else arch["p0_given_1"]
    p1g0    = p1g0_eff if p1g0_eff is not None else arch["p1_given_0"]
    circuits, meta = inv.build_probe_circuits(profile)
    scale   = budget / CANARY_TOTAL
    sh_t1   = max(10, int(round(CANARY_SHOTS_T1     * scale)))
    sh_ram  = max(10, int(round(CANARY_SHOTS_RAMSEY  * scale)))
    sh_gate = max(10, int(round(CANARY_SHOTS_GATE    * scale)))
    sh_echo = max(10, int(round(CANARY_SHOTS_ECHO    * scale)))
    try:
        counts_list = inv.run_probe_circuits_aer(
            circuits, meta,
            T1_true, T2_true, eps_true,
            p0g1, p1g0,
            profile.dt_ns,
            sh_t1, sh_ram, sh_gate, sh_echo,
            dw_s=dw_true,
        )
        r = inv.lindblad_inversion(
            counts_list,
            meta, profile,
            shots_t1=sh_t1, shots_ramsey=sh_ram,
            shots_gate=sh_gate, shots_echo=sh_echo,
            qubit_id=0,
            timestamp=datetime.now(timezone.utc).isoformat())
        return r.T1_s, r.T2_s, abs(r.delta_omega), r.epsilon_sx
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan")


def run_qe(T1_true, T2_true, dw_true, eps_true, budget, seed, arch_name,
           p0g1_eff=None, p1g0_eff=None, profile_dw_max=None):
    from qiskit_experiments.library import (T1 as QE_T1, T2Ramsey,
                                            T2Hahn, StandardRB)
    arch         = inv.ARCH_DEFAULTS[arch_name]
    p0g1         = p0g1_eff if p0g1_eff is not None else arch["p0_given_1"]
    p1g0         = p1g0_eff if p1g0_eff is not None else arch["p1_given_0"]
    T1_pr        = arch["T1_s"]
    T2_pr        = arch["T2_s"]
    eps_max      = arch["eps_max"]
    gate_time_ns = arch["gate_time_ns"]
    dw_max = (profile_dw_max if profile_dw_max is not None
              else inv.BackendProfile.from_architecture(arch_name).dw_max_rad_s)
    budget_per_exp = budget // 4
    n_rb_circs     = len(QE_RB_LENGTHS) * QE_RB_SAMPLES
    shots_t1  = max(10, budget_per_exp // QE_N_DELAYS)
    shots_t2r = max(10, budget_per_exp // QE_N_DELAYS)
    shots_t2h = max(10, budget_per_exp // QE_N_DELAYS)
    shots_rb  = max(10, budget_per_exp // n_rb_circs)
    delays_t1 = np.linspace(1e-6, 5.0 * T1_pr, QE_N_DELAYS).tolist()
    delays_t2 = np.linspace(1e-6, 3.0 * T2_pr, QE_N_DELAYS).tolist()
    backend = _qe_backend(arch_name)
    dt_ns = (backend.dt * 1e9) if getattr(backend, "dt", None) else 0.2222

    # Native Execution
    native_target     = None
    native_gate_names = None
    if arch_name == "trapped_ion":
        native_target = _ionq_native_target(gate_time_ns)
        if native_target is not None:
            native_gate_names = ["gpi", "gpi2"]

    T1_qe   = float("nan");  T1_std  = float("nan")
    T2r_qe  = float("nan");  T2r_std = float("nan")
    T2h_qe  = float("nan");  T2h_std = float("nan")
    dw_qe   = float("nan")
    eps_qe  = float("nan")
    guard_flags = {"T1": (False, False), "T2": (False, False),
                   "dw": (False, False), "eps": (False, False)}
    T1_true_max = TRUE_PARAM_RANGES[arch_name]["T1_s"][1]
    T1_guard_threshold = QE_T1_PLAUSIBILITY_MULT * T1_true_max

    try:
        data          = _run_exp(QE_T1([0], delays=delays_t1, backend=backend),
                                 T1_true, T2_true, eps_true, p0g1, p1g0, shots_t1,
                                 gate_time_ns, dt_ns,
                                 native_target=native_target,
                                 native_gate_names=native_gate_names)
        T1_candidate, T1_std = _extract(data, "T1")
        if np.isfinite(T1_candidate):
            plausible = (0.0 < T1_candidate <= T1_guard_threshold)
            guard_flags["T1"] = (True, not plausible)
            if plausible:
                T1_qe = T1_candidate
    except Exception:
        pass

    try:
        osc_freq_hz      = arch["dw_typical_khz"] * 1e3
        data             = _run_exp(
            T2Ramsey([0], delays=delays_t2, osc_freq=osc_freq_hz, backend=backend),
            T1_true, T2_true, eps_true, p0g1, p1g0, shots_t2r,
            gate_time_ns, dt_ns,
            dw_true=dw_true, inject_detuning=True,
            native_target=native_target, native_gate_names=native_gate_names)
        T2r_qe, T2r_std  = _extract(data, "T2star")
        freq_hz, _        = _extract(data, "Frequency")
        if np.isfinite(freq_hz):
            dw_candidate = abs(freq_hz - osc_freq_hz) * 2.0 * np.pi
            plausible = dw_candidate <= QE_DW_PLAUSIBILITY_MULT * dw_max
            guard_flags["dw"] = (True, not plausible)
            if plausible:
                dw_qe = dw_candidate
    except Exception:
        pass

    try:
        T2e_true         = min(T2_true, 2.0 * T1_true)
        data             = _run_exp(
            T2Hahn([0], delays=delays_t2, backend=backend),
            T1_true, T2e_true, eps_true, p0g1, p1g0, shots_t2h,
            gate_time_ns, dt_ns,
            native_target=native_target, native_gate_names=native_gate_names)
        T2h_qe, T2h_std  = _extract(data, "T2")
    except Exception:
        pass

    valid = [(v, s) for v, s in [(T2r_qe, T2r_std), (T2h_qe, T2h_std)]
             if np.isfinite(v) and np.isfinite(s) and s > 0]
    if len(valid) == 2:
        w0 = 1.0 / valid[0][1] ** 2
        w1 = 1.0 / valid[1][1] ** 2
        T2_candidate = (w0 * valid[0][0] + w1 * valid[1][0]) / (w0 + w1)
    elif len(valid) == 1:
        T2_candidate = valid[0][0]
    elif np.isfinite(T2r_qe):
        T2_candidate = T2r_qe
    elif np.isfinite(T2h_qe):
        T2_candidate = T2h_qe
    else:
        T2_candidate = float("nan")

    T2_combined = float("nan")
    if np.isfinite(T2_candidate):
        plausible = (0.0 < T2_candidate <= T1_guard_threshold)
        guard_flags["T2"] = (True, not plausible)
        if plausible:
            T2_combined = T2_candidate
    if np.isfinite(T1_qe) and np.isfinite(T2_combined):
        T2_combined = min(T2_combined, 2.0 * T1_qe)

    try:
        exp_rb = StandardRB([0], lengths=QE_RB_LENGTHS,
                            num_samples=QE_RB_SAMPLES,
                            seed=int(seed), backend=backend)
        exp_rb.analysis.set_options(gate_error_ratio=False)
        data       = _run_exp(exp_rb, T1_true, T2_true, eps_true,
                              p0g1, p1g0, shots_rb, gate_time_ns, dt_ns,
                              native_target=native_target,
                              native_gate_names=native_gate_names)
        epc, _     = _extract(data, "EPC")
        if np.isfinite(epc) and epc > 0:
            gates_per_clifford = QE_SX_PER_CLIFFORD
            if arch_name == "trapped_ion":
                native_target = _ionq_native_target(gate_time_ns)
                if native_target is not None:
                    measured = _native_gates_per_clifford(
                        exp_rb.circuits(), native_target)
                    if np.isfinite(measured) and measured > 0:
                        gates_per_clifford = measured
            eps_candidate = epc / gates_per_clifford
            eps_true_max = TRUE_PARAM_RANGES[arch_name]["eps"][1]
            eps_guard_threshold = QE_EPS_PLAUSIBILITY_MULT * eps_true_max
            plausible = (0.0 <= eps_candidate <= eps_guard_threshold)
            guard_flags["eps"] = (True, not plausible)
            if plausible:
                eps_qe = float(np.clip(eps_candidate, 0.0, eps_max))
    except Exception:
        pass

    return float(T1_qe), float(T2_combined), float(dw_qe), float(eps_qe), guard_flags


def _r2_point(true_arr, rec_arr):
    t = np.asarray(true_arr, float)
    r = np.asarray(rec_arr,  float)
    mask = np.isfinite(r) & np.isfinite(t)
    if mask.sum() < 2:
        return float("nan")
    t, r   = t[mask], r[mask]
    ss_res = np.sum((r - t) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def bootstrap_r2(true_vals, rec_vals, n_boot=N_BOOTSTRAP, rng_seed=0):
    t = np.asarray(true_vals, float)
    r = np.asarray(rec_vals,  float)
    mask = np.isfinite(r) & np.isfinite(t)
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    t, r   = t[mask], r[mask]
    n      = len(t)
    ss_res = np.sum((r - t) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    r2_pt  = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    rng    = np.random.default_rng(rng_seed)
    boot   = []
    for _ in range(n_boot):
        idx  = rng.integers(0, n, size=n)
        ss_r = np.sum((r[idx] - t[idx]) ** 2)
        ss_t = np.sum((t[idx] - t[idx].mean()) ** 2)
        boot.append(float(1.0 - ss_r / ss_t) if ss_t > 0 else 0.0)
    b = np.array(boot)
    return r2_pt, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def nrmse(true_vals, rec_vals):
    t = np.asarray(true_vals, float)
    r = np.asarray(rec_vals,  float)
    mask = np.isfinite(r) & np.isfinite(t)
    if mask.sum() < 2:
        return float("nan")
    t, r  = t[mask], r[mask]
    rmse  = float(np.sqrt(np.mean((r - t) ** 2)))
    span  = float(t.max() - t.min())
    return rmse / span if span > 0 else float("nan")


def _worker_canary(args):
    warnings.filterwarnings("ignore")
    idx, (T1, T2, dw, eps), seed, budget, arch_name, p0g1_eff, p1g0_eff = args
    t0  = time.perf_counter()
    res = run_canary(T1, T2, dw, eps, seed, budget, arch_name, p0g1_eff, p1g0_eff)
    return idx, res, time.perf_counter() - t0


def _worker_qe(args):
    warnings.filterwarnings("ignore")
    idx, (T1, T2, dw, eps), budget, seed, arch_name, p0g1_eff, p1g0_eff, profile_dw_max = args
    t0  = time.perf_counter()
    res = run_qe(T1, T2, dw, eps, budget, seed, arch_name, p0g1_eff, p1g0_eff, profile_dw_max)
    return idx, res, time.perf_counter() - t0


def run_sweep(n_workers):
    rng_master  = np.random.default_rng(SEED)
    instances   = [sample_instance(rng_master, ARCH) for _ in range(N_INSTANCES)]
    rng_perturb = np.random.default_rng(SEED + 999)
    arch_def    = inv.ARCH_DEFAULTS[ARCH]
    perturbed   = [
        _perturb_instance(
            ARCH,
            inst[0], inst[1], inst[3],
            arch_def["p0_given_1"], arch_def["p1_given_0"],
            rng_perturb)
        for inst in instances
    ]
    profile_dw_max = inv.BackendProfile.from_architecture(ARCH).dw_max_rad_s
    print(f"  profile_dw_max (dynamic, used for QE's dw guard) = "
          f"{profile_dw_max:.4f} rad/s", flush=True)
    rows = []
    print(f"  ARCH={ARCH}  N={N_INSTANCES}  budgets={SHOT_BUDGETS}  workers={n_workers}", flush=True)
    print(f"  Canary={CANARY_TOTAL} shots | QE=budget/4 per exp | osc_freq=arch-typical | RB lengths={QE_RB_LENGTHS}", flush=True)
    print(f"  Realistic noise: TLS T1 drift, 1/f T2 reduction, coherent over-rotation, SPAM drift", flush=True)
    print(f"  Perturbations applied once per instance; identical effective params fed to both methods.", flush=True)

    for bi, budget in enumerate(SHOT_BUDGETS):
        print(f"\n[{bi+1}/{len(SHOT_BUDGETS)}] budget={budget:,}  arch={ARCH}", flush=True)
        c_seed_base = SEED * 10_000 + bi * 1_000
        q_seed_base = c_seed_base + 500
        c_args = [
            (i,
             (perturbed[i][0], perturbed[i][1], instances[i][2], perturbed[i][2]),
             c_seed_base + i, budget, ARCH,
             perturbed[i][3], perturbed[i][4])
            for i in range(N_INSTANCES)
        ]
        q_args = [
            (i,
             (perturbed[i][0], perturbed[i][1], instances[i][2], perturbed[i][2]),
             budget, q_seed_base + i, ARCH,
             perturbed[i][3], perturbed[i][4], profile_dw_max)
            for i in range(N_INSTANCES)
        ]
        c_res  = [None] * N_INSTANCES;  c_time = [float("nan")] * N_INSTANCES
        q_res  = [None] * N_INSTANCES;  q_time = [float("nan")] * N_INSTANCES
        with mp.Pool(n_workers) as pool:
            for idx, res, t in pool.imap_unordered(_worker_canary, c_args, chunksize=4):
                c_res[idx]  = res;  c_time[idx] = t
        with mp.Pool(n_workers) as pool:
            for idx, res, t in pool.imap_unordered(_worker_qe, q_args, chunksize=4):
                q_res[idx]  = res;  q_time[idx] = t

        for res in q_res:
            gf = res[4]
            for pname, (had_candidate, rejected) in gf.items():
                if had_candidate:
                    QE_GUARD_TOTAL[ARCH][pname]   += 1
                    QE_GUARD_REJECTS[ARCH][pname] += int(rejected)

        true = {
            "T1":          [perturbed[i][0] for i in range(N_INSTANCES)],
            "T2":          [perturbed[i][1] for i in range(N_INSTANCES)],
            "delta_omega": [abs(instances[i][2]) for i in range(N_INSTANCES)],
            "epsilon_sx":  [perturbed[i][2] for i in range(N_INSTANCES)],
        }
        for pname, pidx in PARAMS:
            tv = true[pname]
            cv = [r[pidx] for r in c_res]
            qv = [r[pidx] for r in q_res]
            for method, vals, times in [
                    ("Canary",            cv, c_time),
                    ("QiskitExperiments", qv, q_time)]:
                r2pt, r2lo, r2hi = bootstrap_r2(tv, vals, rng_seed=bi * 100 + pidx)
                nr      = nrmse(tv, vals)
                n_valid = int(sum(1 for v in vals if np.isfinite(v)))
                mean_t  = float(np.nanmean(times))
                rows.append({
                    "budget":      budget,
                    "architecture": ARCH,
                    "parameter":   pname,
                    "method":      method,
                    "r2":          r2pt,
                    "r2_lo":       r2lo,
                    "r2_hi":       r2hi,
                    "nrmse":       nr,
                    "n_valid":     n_valid,
                    "n_total":     N_INSTANCES,
                    "mean_time_s": mean_t,
                })
            rc = _r2_point(tv, cv); nc = sum(1 for v in cv if np.isfinite(v))
            rq = _r2_point(tv, qv); nq = sum(1 for v in qv if np.isfinite(v))
            print(f"  {pname:12s}: Canary R2={rc:+.4f} ({nc}/{N_INSTANCES})"
                  f"  QE R2={rq:+.4f} ({nq}/{N_INSTANCES})", flush=True)

        save_benchmark_csv(rows)
        print(f"  [checkpoint] saved after budget={budget:,} "
              f"({bi+1}/{len(SHOT_BUDGETS)} budgets complete)", flush=True)

    print(f"\n  QE plausibility-guard summary for {ARCH} "
          f"(rejected/had-candidate, across all budgets):", flush=True)
    for pname in ("T1", "T2", "dw", "eps"):
        total    = QE_GUARD_TOTAL[ARCH].get(pname, 0)
        rejected = QE_GUARD_REJECTS[ARCH].get(pname, 0)
        pct      = (100.0 * rejected / total) if total else 0.0
        flag     = "  <-- check this" if pct > 50.0 else ""
        print(f"    {pname:4s}: {rejected:5d}/{total:5d} rejected "
              f"({pct:5.1f}%){flag}", flush=True)
    return rows


def save_benchmark_csv(rows):
    path = DATA_DIR / "qe_benchmark.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BENCH_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {path}  ({len(rows)} rows)", flush=True)
    return path


def save_threshold_csv(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["architecture"], row["parameter"], row["method"])].append(
            (row["budget"], row["r2"]))
    thresh_rows = []
    max_b = max(SHOT_BUDGETS)
    for (arch_name, pname, method), pairs in sorted(groups.items()):
        pairs.sort()
        s95 = next((b for b, r2 in pairs if np.isfinite(r2) and r2 >= 0.95), None)
        s99 = next((b for b, r2 in pairs if np.isfinite(r2) and r2 >= 0.99), None)
        thresh_rows.append({
            "architecture":    arch_name,
            "parameter":       pname,
            "method":          method,
            "shots_to_R2_095": s95 if s95 else f"> {max_b:,}",
            "shots_to_R2_099": s99 if s99 else f"> {max_b:,}",
        })
    path = DATA_DIR / "qe_threshold.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=THRESH_COLS)
        w.writeheader()
        w.writerows(thresh_rows)
    print(f"Saved: {path}", flush=True)
    by_arch_param = defaultdict(dict)
    for r in thresh_rows:
        by_arch_param[(r["architecture"], r["parameter"])][r["method"]] = r["shots_to_R2_095"]
    print(f"\n  Shots to R2>=0.95:")
    print(f"  {'Arch':16s}  {'Parameter':14s}  {'Canary':>10s}  {'QE':>10s}")
    print(f"  {'-'*56}")
    for (arch_name, pname), d in sorted(by_arch_param.items()):
        sc = d.get("Canary", "-")
        sq = d.get("QiskitExperiments", "-")
        print(f"  {arch_name:16s}  {pname:14s}  {str(sc):>10s}  {str(sq):>10s}")
    return path


if __name__ == "__main__":
    import os
    warnings.filterwarnings("ignore")
    if sys.platform != "win32":
        mp.set_start_method("fork", force=True)
    n_workers = int(os.environ.get("N_WORKERS", max(1, mp.cpu_count() - 1)))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"[{ts}] QE benchmark - Canary vs. Qiskit Experiments", flush=True)
    print(f"  Architecture={ARCH}  Seed={SEED}  N={N_INSTANCES}  bootstrap={N_BOOTSTRAP}  workers={n_workers}", flush=True)
    rows = run_sweep(n_workers)
    save_benchmark_csv(rows)
    save_threshold_csv(rows)
    print(f"\n[job JOB_TAG={JOB_TAG}] done. Outputs: data/qe_benchmark.csv, data/qe_threshold.csv", flush=True)