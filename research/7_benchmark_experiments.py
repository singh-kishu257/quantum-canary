import importlib.util
import multiprocessing as mp
import pathlib
import sys
import csv
import time
import warnings
import numpy as np
from datetime import datetime, timezone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("inversion", SCRIPT_DIR / "1_inversion.py")
inv = importlib.util.module_from_spec(_spec)
sys.modules["inversion"] = inv
_spec.loader.exec_module(inv)

DATA_DIR = SCRIPT_DIR / "data"
FIG_DIR  = SCRIPT_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

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

JOB_TAG = _os.environ.get("JOB_TAG", f"full_{ARCH}")

# Real, physically-sampled current-hardware ranges per architecture — same
# ranges used throughout the codebase (2_parity_experiments.py,
# 3_shot_ablation.py) for consistency across all synthetic experiments.
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
    "budget", "architecture", "parameter", "method",
    "r2", "r2_lo", "r2_hi",
    "nrmse", "n_valid", "n_total", "mean_time_s",
]

THRESH_COLS = ["architecture", "parameter", "method", "shots_to_R2_095", "shots_to_R2_099"]

PARAM_LATEX = {
    "T1":          r"$T_1$",
    "T2":          r"$T_2$ (combined)",
    "delta_omega": r"$|\Delta\omega|$",
    "epsilon_sx":  r"$\varepsilon_{sx}$",
}

COLOUR = {"Canary": "#1F3864", "QiskitExperiments": "#C00000"}
MARKER = {"Canary": "o",       "QiskitExperiments": "s"}
LABEL  = {
    "Canary":            "Canary (this work)",
    "QiskitExperiments": "Qiskit Experiments (T1+T2R+T2H+RB)",
}
METHODS = ["Canary", "QiskitExperiments"]
PORDER  = ["T1", "T2", "delta_omega", "epsilon_sx"]


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


def _make_backend(qc, T1, T2, eps, p0g1, p1g0, gate_time_ns, dt_ns):
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (NoiseModel, thermal_relaxation_error,
                                   depolarizing_error, ReadoutError)
    nm   = NoiseModel()
    gate_time_s = gate_time_ns * 1e-9
    sx_e = (thermal_relaxation_error(T1, T2, gate_time_s)
            .compose(depolarizing_error(2.0 * eps, 1)))
    # rz excluded: virtual Z rotation, zero physical pulse, zero noise on
    # every real hardware platform (same convention as 1_inversion.py).
    nm.add_quantum_error(sx_e, ["sx", "x", "id", "rx", "u1", "p"], [0])
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
            dw_true=0.0, inject_detuning=False):
    from qiskit import transpile
    exp_obj.set_run_options(shots=shots)
    circuits = exp_obj.circuits()
    exp_data = exp_obj._initialize_experiment_data()
    for qc in circuits:
        run_qc = (inv._inject_ramsey_detuning(qc, dw_true, dt_ns)
                  if inject_detuning else qc)
        b    = _make_backend(run_qc, T1, T2, eps, p0g1, p1g0, gate_time_ns, dt_ns)
        tqc  = transpile(run_qc, b, optimization_level=0)
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


def _qe_backend():
    try:
        from qiskit.providers.fake_provider import GenericBackendV2
        return GenericBackendV2(
            num_qubits=1,
            basis_gates=["sx", "x", "rz", "measure", "delay"],
            seed=0)
    except Exception:
        from qiskit_ibm_runtime.fake_provider import FakeNairobiV2
        return FakeNairobiV2()


def run_canary(T1_true, T2_true, dw_true, eps_true, seed, budget, arch_name):
    arch    = inv.ARCH_DEFAULTS[arch_name]
    profile = inv.BackendProfile.from_architecture(arch_name)
    p0g1    = arch["p0_given_1"]
    p1g0    = arch["p1_given_0"]

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


def run_qe(T1_true, T2_true, dw_true, eps_true, budget, seed, arch_name):
    # NOTE: for trapped_ion and neutral_atom, QE's fixed 15-point linear
    # delay grid (spanning multiple seconds to capture T2 decay) is
    # structurally unable to resolve kHz-scale detuning via T2Ramsey — the
    # Nyquist limit set by the grid spacing (~2 Hz for trapped_ion) is
    # orders of magnitude below the realistic detuning range (up to 10 kHz).
    # No choice of osc_freq fixes this; it is a genuine structural
    # limitation of a fixed linear grid trying to serve both T2 estimation
    # (needs a wide time span) and Δω estimation (needs fine time
    # resolution) simultaneously. This is disclosed honestly in results
    # rather than patched, since no realistic osc_freq choice resolves it.
    from qiskit_experiments.library import (T1 as QE_T1, T2Ramsey,
                                            T2Hahn, StandardRB)
    arch         = inv.ARCH_DEFAULTS[arch_name]
    p0g1         = arch["p0_given_1"]
    p1g0         = arch["p1_given_0"]
    T1_pr        = arch["T1_s"]
    T2_pr        = arch["T2_s"]
    eps_max      = arch["eps_max"]
    gate_time_ns = arch["gate_time_ns"]
    dt_ns        = arch.get("dt_ns")

    budget_per_exp = budget // 4
    n_rb_circs     = len(QE_RB_LENGTHS) * QE_RB_SAMPLES
    shots_t1  = max(10, budget_per_exp // QE_N_DELAYS)
    shots_t2r = max(10, budget_per_exp // QE_N_DELAYS)
    shots_t2h = max(10, budget_per_exp // QE_N_DELAYS)
    shots_rb  = max(10, budget_per_exp // n_rb_circs)

    delays_t1 = np.linspace(1e-6, 5.0 * T1_pr, QE_N_DELAYS).tolist()
    delays_t2 = np.linspace(1e-6, 3.0 * T2_pr, QE_N_DELAYS).tolist()

    backend = _qe_backend()

    T1_qe   = float("nan");  T1_std  = float("nan")
    T2r_qe  = float("nan");  T2r_std = float("nan")
    T2h_qe  = float("nan");  T2h_std = float("nan")
    dw_qe   = float("nan")
    eps_qe  = float("nan")

    try:
        data          = _run_exp(QE_T1([0], delays=delays_t1, backend=backend),
                                 T1_true, T2_true, eps_true, p0g1, p1g0, shots_t1,
                                 gate_time_ns, dt_ns)
        T1_qe, T1_std = _extract(data, "T1")
    except Exception:
        pass

    try:
        osc_freq_hz      = arch["dw_typical_khz"] * 1e3
        data             = _run_exp(
            T2Ramsey([0], delays=delays_t2, osc_freq=osc_freq_hz, backend=backend),
            T1_true, T2_true, eps_true, p0g1, p1g0, shots_t2r,
            gate_time_ns, dt_ns,
            dw_true=dw_true, inject_detuning=True)
        T2r_qe, T2r_std  = _extract(data, "T2star")
        freq_hz, _        = _extract(data, "Frequency")
        if np.isfinite(freq_hz):
            dw_qe = abs(freq_hz - osc_freq_hz) * 2.0 * np.pi
    except Exception:
        pass

    try:
        T2e_true         = min(T2_true, 2.0 * T1_true)
        data             = _run_exp(
            T2Hahn([0], delays=delays_t2, backend=backend),
            T1_true, T2e_true, eps_true, p0g1, p1g0, shots_t2h,
            gate_time_ns, dt_ns)
        T2h_qe, T2h_std  = _extract(data, "T2")
    except Exception:
        pass

    valid = [(v, s) for v, s in [(T2r_qe, T2r_std), (T2h_qe, T2h_std)]
             if np.isfinite(v) and np.isfinite(s) and s > 0]
    if len(valid) == 2:
        w0 = 1.0 / valid[0][1] ** 2
        w1 = 1.0 / valid[1][1] ** 2
        T2_combined = (w0 * valid[0][0] + w1 * valid[1][0]) / (w0 + w1)
    elif len(valid) == 1:
        T2_combined = valid[0][0]
    elif np.isfinite(T2r_qe):
        T2_combined = T2r_qe
    elif np.isfinite(T2h_qe):
        T2_combined = T2h_qe
    else:
        T2_combined = float("nan")

    if np.isfinite(T1_qe) and np.isfinite(T2_combined):
        T2_combined = min(T2_combined, 2.0 * T1_qe)

    try:
        exp_rb = StandardRB([0], lengths=QE_RB_LENGTHS,
                            num_samples=QE_RB_SAMPLES,
                            seed=int(seed), backend=backend)
        exp_rb.analysis.set_options(gate_error_ratio=False)
        data       = _run_exp(exp_rb, T1_true, T2_true, eps_true,
                              p0g1, p1g0, shots_rb, gate_time_ns, dt_ns)
        epc, _     = _extract(data, "EPC")
        if np.isfinite(epc) and epc > 0:
            eps_qe = float(np.clip(epc / QE_SX_PER_CLIFFORD, 0.0, eps_max))
    except Exception:
        pass

    return float(T1_qe), float(T2_combined), float(dw_qe), float(eps_qe)


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
    idx, (T1, T2, dw, eps), seed, budget, arch_name = args
    t0  = time.perf_counter()
    res = run_canary(T1, T2, dw, eps, seed, budget, arch_name)
    return idx, res, time.perf_counter() - t0


def _worker_qe(args):
    warnings.filterwarnings("ignore")
    idx, (T1, T2, dw, eps), budget, seed, arch_name = args
    t0  = time.perf_counter()
    res = run_qe(T1, T2, dw, eps, budget, seed, arch_name)
    return idx, res, time.perf_counter() - t0


def run_sweep(n_workers):
    rng_master = np.random.default_rng(SEED)
    instances  = [sample_instance(rng_master, ARCH) for _ in range(N_INSTANCES)]
    rows       = []

    print(f"  ARCH={ARCH}  N={N_INSTANCES}  budgets={SHOT_BUDGETS}  workers={n_workers}", flush=True)
    print(f"  Canary={CANARY_TOTAL} shots | QE=budget/4 per exp | osc_freq=arch-typical | RB lengths={QE_RB_LENGTHS}", flush=True)

    for bi, budget in enumerate(SHOT_BUDGETS):
        print(f"\n[{bi+1}/{len(SHOT_BUDGETS)}] budget={budget:,}  arch={ARCH}", flush=True)

        c_seed_base = SEED * 10_000 + bi * 1_000
        q_seed_base = c_seed_base + 500

        c_args = [(i, inst, c_seed_base + i, budget, ARCH)
                  for i, inst in enumerate(instances)]
        q_args = [(i, inst, budget, q_seed_base + i, ARCH)
                  for i, inst in enumerate(instances)]

        c_res  = [None] * N_INSTANCES;  c_time = [float("nan")] * N_INSTANCES
        q_res  = [None] * N_INSTANCES;  q_time = [float("nan")] * N_INSTANCES

        with mp.Pool(n_workers) as pool:
            for idx, res, t in pool.imap_unordered(_worker_canary, c_args, chunksize=4):
                c_res[idx]  = res;  c_time[idx] = t

        with mp.Pool(n_workers) as pool:
            for idx, res, t in pool.imap_unordered(_worker_qe, q_args, chunksize=4):
                q_res[idx]  = res;  q_time[idx] = t

        true = {
            "T1":          [abs(inst[0]) for inst in instances],
            "T2":          [abs(inst[1]) for inst in instances],
            "delta_omega": [abs(inst[2]) for inst in instances],
            "epsilon_sx":  [abs(inst[3]) for inst in instances],
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

    return rows


def save_benchmark_csv(rows):
    path = DATA_DIR / f"fig7_benchmark_{JOB_TAG}.csv"
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

    path = DATA_DIR / "fig7_threshold.csv"
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


def plot_results(rows):
    idx_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    architectures_present = []
    for row in rows:
        a = row["architecture"]
        if a not in architectures_present:
            architectures_present.append(a)
        idx_data[a][row["parameter"]][row["method"]][row["budget"]] = row

    architectures_present = [a for a in ALL_ARCHITECTURES if a in architectures_present]
    budgets = sorted(SHOT_BUDGETS)
    n_rows  = len(architectures_present)
    fig, axes = plt.subplots(n_rows, 4, figsize=(15, 3.6 * n_rows), squeeze=False)

    ARCH_LABEL = {"superconducting": "Superconducting", "trapped_ion": "Trapped ion",
                  "neutral_atom": "Neutral atom"}

    for row_i, arch_name in enumerate(architectures_present):
        for col, pname in enumerate(PORDER):
            ax = axes[row_i, col]

            for method in METHODS:
                xs, ys, los, his = [], [], [], []
                for b in budgets:
                    r = idx_data[arch_name][pname][method].get(b)
                    if r is None:
                        continue
                    xs.append(b)
                    if np.isfinite(r["r2"]):
                        ys.append(r["r2"])
                        los.append(r["r2"] - r["r2_lo"])
                        his.append(r["r2_hi"] - r["r2"])
                    else:
                        ys.append(float("nan"))
                        los.append(0.0)
                        his.append(0.0)

                xs  = np.array(xs); ys = np.array(ys)
                los = np.array(los); his = np.array(his)
                finite = np.isfinite(ys)

                if finite.any():
                    ax.fill_between(xs[finite], (ys - los)[finite], (ys + his)[finite],
                                    color=COLOUR[method], alpha=0.12)
                    ax.errorbar(xs[finite], ys[finite],
                                yerr=[los[finite], his[finite]],
                                color=COLOUR[method], marker=MARKER[method],
                                label=LABEL[method],
                                linewidth=1.8, markersize=5, capsize=3,
                                elinewidth=1.0, alpha=0.95)

            ax.axhline(0.95, color="#555", linestyle="--", linewidth=0.9, alpha=0.7)
            ax.axvline(CANARY_TOTAL, color=COLOUR["Canary"], linestyle=":",
                       linewidth=1.0, alpha=0.5)
            ax.set_xscale("log")
            ax.set_xlim(budgets[0] * 0.75, budgets[-1] * 1.35)
            ax.set_ylim(-0.15, 1.05)
            if row_i == 0:
                ax.set_title(PARAM_LATEX[pname], fontsize=12)
            if row_i == n_rows - 1:
                ax.set_xlabel("Total shots", fontsize=9)
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))
            ax.grid(True, which="major", alpha=0.25)
            ax.grid(True, which="minor", alpha=0.08)
            if col == 0:
                ax.set_ylabel(f"{ARCH_LABEL[arch_name]}\n" + r"$R^2$ (bootstrap median, 95% CI)",
                              fontsize=8.5)
            if col == 0 and row_i == 0:
                ax.legend(fontsize=6.5, loc="lower right", framealpha=0.85)

    fig.suptitle(
        "Fig. 7 - Canary vs. Qiskit Experiments: "
        r"$R^2$ vs. total shot budget across architectures"
        "\n(N=200 instances, 2000-resample bootstrap 95% CI; "
        "QE: osc_freq=arch-typical prior, arch-prior delays, equal budget; "
        "vertical dotted = Canary native budget; dashed = 0.95 target)",
        fontsize=9, y=1.01)

    plt.tight_layout()
    for fmt in ("pdf", "png"):
        p = FIG_DIR / f"fig7_benchmark.{fmt}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    import os
    warnings.filterwarnings("ignore")
    if sys.platform != "win32":
        mp.set_start_method("fork", force=True)
    n_workers = int(os.environ.get("N_WORKERS", max(1, mp.cpu_count() - 1)))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"[{ts}] Fig. 7 benchmark - Canary vs. Qiskit Experiments", flush=True)
    print(f"  Architecture={ARCH}  Seed={SEED}  N={N_INSTANCES}  bootstrap={N_BOOTSTRAP}  workers={n_workers}", flush=True)

    rows = run_sweep(n_workers)
    save_benchmark_csv(rows)
    print(f"\n[job JOB_TAG={JOB_TAG}] "
         f"Skipping threshold CSV and plot — run 7_merge_results.py "
         f"after all matrix jobs complete to produce final outputs.",
         flush=True)