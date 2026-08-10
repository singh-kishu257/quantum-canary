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
ARCH               = "superconducting"
DT_S               = 2.2222e-10
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
JOB_TAG = _os.environ.get("JOB_TAG", "full")

TRUE_PARAM_RANGES = {
    "T1_s": (80e-6,  400e-6),
    "T2_s": (40e-6,  200e-6),
    "eps":  (1e-4,   2e-3),
}

PARAMS = [
    ("T1",          0),
    ("T2",          1),
    ("delta_omega", 2),
    ("epsilon_sx",  3),
]

BENCH_COLS = [
    "budget", "parameter", "method",
    "r2", "r2_lo", "r2_hi",
    "nrmse", "n_valid", "n_total", "mean_time_s",
]

THRESH_COLS = ["parameter", "method", "shots_to_R2_095", "shots_to_R2_099"]

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


def sample_instance(rng):
    T1    = float(rng.uniform(*TRUE_PARAM_RANGES["T1_s"]))
    T_phi = float(rng.uniform(*TRUE_PARAM_RANGES["T2_s"]))
    T2    = min(1.0 / (1.0 / (2.0 * T1) + 1.0 / T_phi), 2.0 * T1)
    eps   = float(10 ** rng.uniform(
                  np.log10(TRUE_PARAM_RANGES["eps"][0]),
                  np.log10(TRUE_PARAM_RANGES["eps"][1])))
    dw_max = inv.BackendProfile.from_architecture(ARCH).dw_max_rad_s
    dw     = float(rng.choice([-1, 1]) * rng.uniform(0.2 * dw_max, dw_max))
    return T1, T2, dw, eps


def _make_backend(qc, T1, T2, eps, p0g1, p1g0):
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (NoiseModel, thermal_relaxation_error,
                                   depolarizing_error, ReadoutError)
    nm   = NoiseModel()
    sx_e = (thermal_relaxation_error(T1, T2, 50e-9)
            .compose(depolarizing_error(2.0 * eps, 1)))
    nm.add_quantum_error(sx_e, ["sx", "x", "id", "rx", "u1", "p"], [0])
    seen = set()
    for inst in qc.data:
        if inst.operation.name == "delay":
            dur = inst.operation.params[0]
            if dur not in seen:
                seen.add(dur)
                ds = dur * DT_S
                if ds > 1e-9:
                    nm.add_quantum_error(
                        thermal_relaxation_error(T1, T2, ds), ["delay"], [0])
    nm.add_readout_error(
        ReadoutError([[1 - p0g1, p0g1], [p1g0, 1 - p1g0]]), [0])
    return AerSimulator(noise_model=nm)


def _run_exp(exp_obj, T1, T2, eps, p0g1, p1g0, shots,
            dw_true=0.0, dt_ns=None, inject_detuning=False):
    from qiskit import transpile
    exp_obj.set_run_options(shots=shots)
    circuits = exp_obj.circuits()
    exp_data = exp_obj._initialize_experiment_data()
    for qc in circuits:
        run_qc = (inv._inject_ramsey_detuning(qc, dw_true, dt_ns)
                  if inject_detuning else qc)
        b    = _make_backend(run_qc, T1, T2, eps, p0g1, p1g0)
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


def run_canary(T1_true, T2_true, dw_true, eps_true, seed, budget):
    arch    = inv.ARCH_DEFAULTS[ARCH]
    profile = inv.BackendProfile.from_architecture(ARCH)
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


def run_qe(T1_true, T2_true, dw_true, eps_true, budget, seed):
    from qiskit_experiments.library import (T1 as QE_T1, T2Ramsey,
                                            T2Hahn, StandardRB)
    arch    = inv.ARCH_DEFAULTS[ARCH]
    p0g1    = arch["p0_given_1"]
    p1g0    = arch["p1_given_0"]
    T1_pr   = arch["T1_s"]
    T2_pr   = arch["T2_s"]
    eps_max = arch["eps_max"]

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
                                 T1_true, T2_true, eps_true, p0g1, p1g0, shots_t1)
        T1_qe, T1_std = _extract(data, "T1")
    except Exception:
        pass

    try:
        osc_freq_hz      = arch["dw_typical_khz"] * 1e3
        dt_ns            = arch.get("dt_ns")
        data             = _run_exp(
            T2Ramsey([0], delays=delays_t2, osc_freq=osc_freq_hz, backend=backend),
            T1_true, T2_true, eps_true, p0g1, p1g0, shots_t2r,
            dw_true=dw_true, dt_ns=dt_ns, inject_detuning=True)
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
            T1_true, T2e_true, eps_true, p0g1, p1g0, shots_t2h)
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
                              p0g1, p1g0, shots_rb)
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
    idx, (T1, T2, dw, eps), seed, budget = args
    t0  = time.perf_counter()
    res = run_canary(T1, T2, dw, eps, seed, budget)
    return idx, res, time.perf_counter() - t0


def _worker_qe(args):
    warnings.filterwarnings("ignore")
    idx, (T1, T2, dw, eps), budget, seed = args
    t0  = time.perf_counter()
    res = run_qe(T1, T2, dw, eps, budget, seed)
    return idx, res, time.perf_counter() - t0


def run_sweep(n_workers):
    rng_master = np.random.default_rng(SEED)
    instances  = [sample_instance(rng_master) for _ in range(N_INSTANCES)]
    rows       = []

    print(f"  N={N_INSTANCES}  budgets={SHOT_BUDGETS}  workers={n_workers}", flush=True)
    print(f"  Canary={CANARY_TOTAL} shots | QE=budget/4 per exp | osc_freq=arch-typical | RB lengths={QE_RB_LENGTHS}", flush=True)

    for bi, budget in enumerate(SHOT_BUDGETS):
        print(f"\n[{bi+1}/{len(SHOT_BUDGETS)}] budget={budget:,}", flush=True)

        c_seed_base = SEED * 10_000 + bi * 1_000
        q_seed_base = c_seed_base + 500

        c_args = [(i, inst, c_seed_base + i, budget)
                  for i, inst in enumerate(instances)]
        q_args = [(i, inst, budget, q_seed_base + i)
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
        groups[(row["parameter"], row["method"])].append(
            (row["budget"], row["r2"]))

    thresh_rows = []
    max_b = max(SHOT_BUDGETS)
    for (pname, method), pairs in sorted(groups.items()):
        pairs.sort()
        s95 = next((b for b, r2 in pairs if np.isfinite(r2) and r2 >= 0.95), None)
        s99 = next((b for b, r2 in pairs if np.isfinite(r2) and r2 >= 0.99), None)
        thresh_rows.append({
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

    by_param = defaultdict(dict)
    for r in thresh_rows:
        by_param[r["parameter"]][r["method"]] = r["shots_to_R2_095"]
    print(f"\n  Shots to R2>=0.95:")
    print(f"  {'Parameter':14s}  {'Canary':>10s}  {'QE':>10s}")
    print(f"  {'-'*40}")
    for pname, d in by_param.items():
        sc = d.get("Canary", "-")
        sq = d.get("QiskitExperiments", "-")
        print(f"  {pname:14s}  {str(sc):>10s}  {str(sq):>10s}")

    return path


def plot_results(rows):
    idx_data = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        idx_data[row["parameter"]][row["method"]][row["budget"]] = row

    budgets = sorted(SHOT_BUDGETS)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))

    for col, pname in enumerate(PORDER):
        ax  = axes[0, col]
        ax2 = axes[1, col]

        for method in METHODS:
            xs, ys, los, his, frs = [], [], [], [], []
            for b in budgets:
                r = idx_data[pname][method].get(b)
                if r is None:
                    continue
                xs.append(b)
                frs.append(1.0 - r["n_valid"] / r["n_total"])
                if np.isfinite(r["r2"]):
                    ys.append(r["r2"])
                    los.append(r["r2"] - r["r2_lo"])
                    his.append(r["r2_hi"] - r["r2"])
                else:
                    ys.append(float("nan"))
                    los.append(0.0)
                    his.append(0.0)

            xs  = np.array(xs)
            ys  = np.array(ys)
            los = np.array(los)
            his = np.array(his)
            finite = np.isfinite(ys)

            if finite.any():
                ax.fill_between(xs[finite],
                                (ys - los)[finite],
                                (ys + his)[finite],
                                color=COLOUR[method], alpha=0.12)
                ax.errorbar(xs[finite], ys[finite],
                            yerr=[los[finite], his[finite]],
                            color=COLOUR[method], marker=MARKER[method],
                            label=LABEL[method],
                            linewidth=1.8, markersize=5, capsize=3,
                            elinewidth=1.0, alpha=0.95)

            ax2.plot(xs, frs, color=COLOUR[method], marker=MARKER[method],
                     linewidth=1.6, markersize=4, alpha=0.9,
                     label=LABEL[method])

        ax.axhline(0.95, color="#555", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.axvline(CANARY_TOTAL, color=COLOUR["Canary"], linestyle=":",
                   linewidth=1.0, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlim(budgets[0] * 0.75, budgets[-1] * 1.35)
        ax.set_ylim(-0.15, 1.05)
        ax.set_title(PARAM_LATEX[pname], fontsize=12)
        ax.set_xlabel("Total shots", fontsize=9)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))
        ax.grid(True, which="major", alpha=0.25)
        ax.grid(True, which="minor", alpha=0.08)
        if col == 0:
            ax.set_ylabel(r"$R^2$ (bootstrap median, 95% CI)", fontsize=9)
            ax.legend(fontsize=6.5, loc="lower right", framealpha=0.85)

        ax2.axhline(0.05, color="#555", linestyle=":", linewidth=0.8, alpha=0.6)
        ax2.set_xscale("log")
        ax2.set_xlim(budgets[0] * 0.75, budgets[-1] * 1.35)
        ax2.set_ylim(-0.02, 1.05)
        ax2.set_xlabel("Total shots", fontsize=9)
        ax2.grid(True, which="major", alpha=0.25)
        if col == 0:
            ax2.set_ylabel("Failure rate (NaN / total)", fontsize=9)
            ax2.legend(fontsize=6.5, loc="upper right", framealpha=0.85)

    fig.suptitle(
        "Fig. 7 - Canary vs. Qiskit Experiments: "
        r"$R^2$ and failure rate vs. total shot budget"
        "\n(N=200 instances, 2000-resample bootstrap 95% CI; "
        "QE: osc_freq=arch-typical prior, arch-prior delays, equal budget; "
        "vertical dotted = Canary 9,900 shots; dashed = 0.95 target)",
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
    print(f"  Seed={SEED}  N={N_INSTANCES}  bootstrap={N_BOOTSTRAP}  workers={n_workers}", flush=True)

    rows = run_sweep(n_workers)
    save_benchmark_csv(rows)
    if _budget_env:
        print(f"\n[split-run mode: JOB_TAG={JOB_TAG}] "
              f"Skipping threshold CSV and plot — run 7_merge_results.py "
              f"after all split jobs complete to produce final outputs.",
              flush=True)
    else:
        save_threshold_csv(rows)
        plot_results(rows)