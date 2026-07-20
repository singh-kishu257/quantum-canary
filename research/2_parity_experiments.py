import importlib.util, pathlib, sys, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone

_spec = importlib.util.spec_from_file_location(
    "inversion", pathlib.Path(__file__).parent / "1_inversion.py")
inv = importlib.util.module_from_spec(_spec)
sys.modules["inversion"] = inv
_spec.loader.exec_module(inv)

SCRIPT_DIR  = pathlib.Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

N_INSTANCES   = 200
SHOTS_T1      = 1000
SHOTS_RAMSEY  = 3000
SHOTS_GATE    = 1000
SEED          = 42
ARCHITECTURES = ["superconducting", "trapped_ion", "neutral_atom"]
rng = np.random.default_rng(SEED)

COLORS  = {"superconducting": "#1f77b4", "trapped_ion": "#ff7f0e", "neutral_atom": "#2ca02c"}
MARKERS = {"superconducting": "o",       "trapped_ion": "s",       "neutral_atom": "^"}
LABELS  = {"superconducting": "Superconducting (IBM-like)",
           "trapped_ion":     "Trapped ion (IonQ-like)",
           "neutral_atom":    "Neutral atom (QuEra-like)"}


def shot_budget(arch_name: str) -> int:
    c = inv.ARCH_DEFAULTS[arch_name]
    n_t1 = 3 if c.get("t1_mode","3point") == "3point" else 2
    return n_t1*SHOTS_T1 + 2*SHOTS_RAMSEY + 3*SHOTS_GATE


def sample_true_params(arch_name: str, t_opt: float):
    c      = inv.ARCH_DEFAULTS[arch_name]
    T1     = c["T1_s"]   * rng.uniform(0.4, 1.6)
    T2     = min(c["T2_s"] * rng.uniform(0.3, 1.2), 1.95 * T1)
    dw_max = 0.90 * np.pi / t_opt
    dw     = rng.uniform(-dw_max, dw_max)
    eps    = c["eps_typical"] * rng.uniform(0.3, 5.0)
    return T1, T2, dw, eps


def simulate_inversion(arch_name, T1_true, T2_true, dw_true, eps_true):
    profile  = inv.BackendProfile.from_architecture(arch_name)
    arch     = inv.ARCH_DEFAULTS[arch_name]
    t_opt    = profile.ramsey_t_opt_s

    t1_delays = profile.t1_delays_s
    t1_counts = []
    for t in t1_delays:
        p1 = float(inv.forward_t1(t, T1_true)) if t > 0 else 1.0
        n1 = int(rng.binomial(SHOTS_T1, np.clip(p1, 0, 1)))
        t1_counts.append({"0": SHOTS_T1 - n1, "1": n1})

    p1_x, p1_y = inv.forward_ramsey_xy(t_opt, T2_true, dw_true)
    ramsey_counts = []
    for p in (p1_x, p1_y):
        n1 = int(rng.binomial(SHOTS_RAMSEY, float(np.clip(p, 0, 1))))
        ramsey_counts.append({"0": SHOTS_RAMSEY - n1, "1": n1})

    N_vals = np.array(inv.GATE_REP_N_DT, dtype=float)
    p0_vals = inv.forward_gate(N_vals, eps_true)
    gate_counts = []
    for p in p0_vals:
        n0 = int(rng.binomial(SHOTS_GATE, float(np.clip(p, 0, 1))))
        gate_counts.append({"0": n0, "1": SHOTS_GATE - n0})

    if arch["t1_mode"] == "2point":
        T1, sT1, r_t1 = inv._invert_t1_2point(
            t1_counts[0]["1"] / SHOTS_T1,
            t1_counts[1]["1"] / SHOTS_T1,
            t1_delays[1], profile.T1_prior_s, arch)
    else:
        p1_t1 = np.array([c["1"] / SHOTS_T1 for c in t1_counts])
        T1, sT1, r_t1 = inv._invert_t1_3point(p1_t1, np.array(t1_delays), profile.T1_prior_s, arch)

    p1_x_n = ramsey_counts[0]["1"] / SHOTS_RAMSEY
    p1_y_n = ramsey_counts[1]["1"] / SHOTS_RAMSEY
    T2, sT2, dw, sdw, r_ram = inv._invert_ramsey_xy(
        p1_x_n, p1_y_n, t_opt, profile.T2_prior_s, arch)
    T2 = min(T2, 2.0 * T1)

    p0_meas = np.array([c["0"] / SHOTS_GATE for c in gate_counts])
    eps, seps, r_gate = inv._invert_gate(p0_meas, N_vals, arch)

    return inv.InversionResult(
        backend_name="digital_twin", qubit_id=0,
        timestamp=datetime.now(timezone.utc).isoformat(),
        architecture=arch_name,
        T1_s=T1, T1_sigma_s=sT1,
        T2_s=T2, T2_sigma_s=sT2,
        delta_omega=dw, delta_omega_sigma=sdw,
        epsilon_sx=eps, epsilon_sx_sigma=seps,
        t1_residual=r_t1, ramsey_residual=r_ram, gate_residual=r_gate)


def compute_metrics(true_vals, rec_vals):
    t = np.array(true_vals); r = np.array(rec_vals)
    ss_res = np.sum((r - t) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    r2   = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((r - t) ** 2)))
    return r2, rmse


def run_experiment():
    rec = {k: [] for k in ["arch",
        "T1_true","T1_rec","T1_sig",
        "T2_true","T2_rec","T2_sig",
        "dw_true","dw_rec","dw_sig",
        "eps_true","eps_rec","eps_sig",
        "t1_resid","ram_resid","gate_resid"]}

    for arch in ARCHITECTURES:
        profile = inv.BackendProfile.from_architecture(arch)
        t_opt   = profile.ramsey_t_opt_s
        success = 0
        while success < N_INSTANCES:
            T1_t, T2_t, dw_t, eps_t = sample_true_params(arch, t_opt)
            try:
                r = simulate_inversion(arch, T1_t, T2_t, dw_t, eps_t)
            except Exception:
                continue
            if not (np.isfinite(r.T1_s) and np.isfinite(r.T2_s)):
                continue
            rec["arch"].append(arch)
            rec["T1_true"].append(T1_t);   rec["T1_rec"].append(r.T1_s);        rec["T1_sig"].append(r.T1_sigma_s)
            rec["T2_true"].append(T2_t);   rec["T2_rec"].append(r.T2_s);        rec["T2_sig"].append(r.T2_sigma_s)
            rec["dw_true"].append(dw_t);   rec["dw_rec"].append(r.delta_omega); rec["dw_sig"].append(r.delta_omega_sigma)
            rec["eps_true"].append(eps_t); rec["eps_rec"].append(r.epsilon_sx); rec["eps_sig"].append(r.epsilon_sx_sigma)
            rec["t1_resid"].append(r.t1_residual)
            rec["ram_resid"].append(r.ramsey_residual)
            rec["gate_resid"].append(r.gate_residual)
            success += 1
    return rec


def save_csv(rec):
    path = DATA_DIR / "parity_results.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rec.keys()))
        w.writeheader()
        for i in range(len(rec["arch"])):
            w.writerow({k: rec[k][i] for k in rec})
    return path


def make_figure(rec):
    arch_arr = np.array(rec["arch"])
    T1_t  = np.array(rec["T1_true"]);  T1_r  = np.array(rec["T1_rec"]);  T1_s  = np.array(rec["T1_sig"])
    T2_t  = np.array(rec["T2_true"]);  T2_r  = np.array(rec["T2_rec"]);  T2_s  = np.array(rec["T2_sig"])
    dw_t  = np.array(rec["dw_true"]);  dw_r  = np.array(rec["dw_rec"]);  dw_s  = np.array(rec["dw_sig"])
    eps_t = np.array(rec["eps_true"]); eps_r = np.array(rec["eps_rec"]); eps_s = np.array(rec["eps_sig"])

    fig = plt.figure(figsize=(7.16, 6.6))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.42,
                            left=0.11, right=0.97, top=0.93, bottom=0.10)

    panel_specs = [
        (gs[0,0], T1_t,  T1_r,  T1_s,  r"True $T_1$ (s)",              r"Recovered $T_1$ (s)",              "(a)", True,  None),
        (gs[0,1], T2_t,  T2_r,  T2_s,  r"True $T_2$ (s)",              r"Recovered $T_2$ (s)",              "(b)", True,  None),
        (gs[1,0], dw_t,  dw_r,  dw_s,  r"True $\Delta\omega$ (rad/s)", r"Recovered $\Delta\omega$ (rad/s)", "(c)", False,
         "Sign fully recovered\nvia $\\arctan_2(Y,X)$.\nRange: $|\\Delta\\omega|\\leq 0.9\\pi/T_2^{\\rm prior}$"),
        (gs[1,1], eps_t, eps_r, eps_s, r"True $\varepsilon_{sx}$",     r"Recovered $\varepsilon_{sx}$",     "(d)", True,  None),
    ]

    for (spec, tv, rv, sv, xl, yl, lab, loglog, note) in panel_specs:
        ax = fig.add_subplot(spec)
        r2_all, rmse_all = compute_metrics(tv, rv)

        for arch in ARCHITECTURES:
            mask = arch_arr == arch
            ax.errorbar(tv[mask], rv[mask], yerr=sv[mask],
                        fmt=MARKERS[arch], color=COLORS[arch],
                        markersize=3.2, linewidth=0, elinewidth=0.55,
                        capsize=1.4, alpha=0.65, label=LABELS[arch])

        lo = min(tv.min(), rv.min())
        hi = max(tv.max(), rv.max())
        if loglog and lo > 0:
            lo *= 0.82; hi *= 1.22
            ax.set_xscale("log"); ax.set_yscale("log")
        else:
            pad = (hi - lo) * 0.09
            lo -= pad; hi += pad
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, zorder=5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(xl, fontsize=8.5); ax.set_ylabel(yl, fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, which="both" if loglog else "major", linewidth=0.35, alpha=0.45)
        ax.text(0.04, 0.96,
                f"$R^2={r2_all:.4f}$\nRMSE$={rmse_all:.2e}$",
                transform=ax.transAxes, fontsize=7.5, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#cccccc", linewidth=0.6))
        ax.text(-0.16, 1.05, lab, transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", va="top")
        if note:
            ax.text(0.04, 0.60, note, transform=ax.transAxes, fontsize=6.5,
                    color="#555555", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#fffbe6",
                              edgecolor="#cccc88", linewidth=0.5))

    handles, leg_labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=3,
               fontsize=7.5, frameon=True, bbox_to_anchor=(0.5, 0.005),
               handletextpad=0.4, columnspacing=0.8, framealpha=0.9, edgecolor="#cccccc")

    budgets = ", ".join([f"{a.split('_')[0].title()}: {shot_budget(a):,}" for a in ARCHITECTURES])
    fig.suptitle(
        f"Fig. 2 — Parameter Recovery Accuracy (Digital Twin, $N={N_INSTANCES}$/arch, "
        f"seed={SEED}; shots/qubit — {budgets})",
        fontsize=8.0, y=0.98)

    out = FIGURES_DIR / "fig2_parity.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"[{ts}] Fig. 2 — Parity Experiment (Digital Twin)")
    print(f"  Protocol : Dual-quadrature Ramsey (Shnaiderov et al. 2025) + gate-rep depolarizing")
    print(f"  Note     : RPE (Kimmel et al.) used for real IonQ hardware in 5_ionq_hardware.py")
    print()
    for arch in ARCHITECTURES:
        c = inv.ARCH_DEFAULTS[arch]
        p = inv.BackendProfile.from_architecture(arch)
        t_opt  = p.ramsey_t_opt_s
        dw_max = 0.90 * np.pi / t_opt
        n_t1   = 3 if c.get("t1_mode","3point") == "3point" else 2
        budget = shot_budget(arch)
        print(f"  {arch:16s}: t_opt={t_opt:.3g}s  Δω_max={dw_max/(2*np.pi):.3g}Hz  "
              f"n_t1={n_t1}  shots={budget:,}")
    print()

    rec      = run_experiment()
    csv_path = save_csv(rec)
    fig_path = make_figure(rec)

    print(f"  Results (n={N_INSTANCES} per architecture):")
    arch_arr = np.array(rec["arch"])
    for arch in ARCHITECTURES:
        mask = arch_arr == arch
        c = inv.ARCH_DEFAULTS[arch]
        s = c["time_scale"]; u = c["display_unit"]
        T1_r2, T1_rmse = compute_metrics(np.array(rec["T1_true"])[mask]*s, np.array(rec["T1_rec"])[mask]*s)
        T2_r2, T2_rmse = compute_metrics(np.array(rec["T2_true"])[mask]*s, np.array(rec["T2_rec"])[mask]*s)
        dw_r2, dw_rmse = compute_metrics(np.array(rec["dw_true"])[mask], np.array(rec["dw_rec"])[mask])
        ep_r2, ep_rmse = compute_metrics(np.array(rec["eps_true"])[mask], np.array(rec["eps_rec"])[mask])
        print(f"\n    {arch} (shots={shot_budget(arch):,})")
        print(f"      T1 : R²={T1_r2:.4f}  RMSE={T1_rmse:.3e} {u}")
        print(f"      T2 : R²={T2_r2:.4f}  RMSE={T2_rmse:.3e} {u}")
        print(f"      Δω : R²={dw_r2:.4f}  RMSE={dw_rmse:.3e} rad/s")
        print(f"      ε  : R²={ep_r2:.4f}  RMSE={ep_rmse:.3e}")

    print(f"\n  Data  : {csv_path}")
    print(f"  Figure: {fig_path}")