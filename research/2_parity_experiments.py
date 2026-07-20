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

N_INSTANCES   = 300
SHOTS_T1_GATE = 1000
SHOTS_RAMSEY  = 3000
SEED          = 42
ARCHITECTURE  = "superconducting"
rng = np.random.default_rng(SEED)


def sample_true_params():
    c   = inv.ARCH_DEFAULTS[ARCHITECTURE]
    T1  = c["T1_s"] * rng.uniform(0.4, 1.6)
    T2  = min(c["T2_s"] * rng.uniform(0.3, 1.2), 1.95 * T1)
    dw  = 2 * np.pi * 1e3 * rng.uniform(-5.0, 5.0)
    eps = c["eps_typical"] * rng.uniform(0.3, 5.0)
    return T1, T2, dw, eps


def simulate_inversion(T1_true, T2_true, dw_true, eps_true):
    profile  = inv.BackendProfile.from_architecture(ARCHITECTURE)
    arch     = inv.ARCH_DEFAULTS[ARCHITECTURE]
    t1d      = np.array(profile.t1_delays_s)
    rd       = np.array(profile.ramsey_delays_s)
    Nv       = np.array(inv.GATE_REP_N, dtype=float)

    p1t1 = inv.forward_t1(t1d, T1_true)
    p1r  = inv.forward_ramsey(rd, T2_true, dw_true)
    p0g  = inv.forward_gate(Nv, eps_true)

    def c1(p, sh):
        n1 = int(rng.binomial(sh, float(np.clip(p, 0, 1))))
        return {"0": sh - n1, "1": n1}

    def c0(p, sh):
        n0 = int(rng.binomial(sh, float(np.clip(p, 0, 1))))
        return {"0": n0, "1": sh - n0}

    counts = ([c1(p, SHOTS_T1_GATE) for p in p1t1] +
              [c1(p, SHOTS_RAMSEY)  for p in p1r] +
              [c0(p, SHOTS_T1_GATE) for p in p0g])

    T1,  sT1,  r_t1          = inv._invert_t1(
        np.array([counts[i]["1"] / SHOTS_T1_GATE for i in range(3)]),
        t1d, profile.T1_prior_s, arch)
    T2, sT2, dw, sdw, r_ram  = inv._invert_ramsey(
        np.array([counts[i+3]["1"] / SHOTS_RAMSEY for i in range(3)]),
        rd, arch["T2_s"], arch)
    T2 = min(T2, 2.0 * T1)
    eps, seps, r_gate         = inv._invert_gate(
        np.array([counts[i+6]["0"] / SHOTS_T1_GATE for i in range(3)]),
        Nv, arch)

    from datetime import datetime, timezone as tz
    return inv.InversionResult(
        backend_name="digital_twin", qubit_id=0,
        timestamp=datetime.now(tz.utc).isoformat(),
        architecture=ARCHITECTURE,
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
    rec = {k: [] for k in [
        "T1_true","T1_rec","T1_sig",
        "T2_true","T2_rec","T2_sig",
        "dw_true","dw_rec","dw_sig",
        "eps_true","eps_rec","eps_sig",
        "t1_resid","ram_resid","gate_resid"]}
    successes = 0
    while successes < N_INSTANCES:
        T1_t, T2_t, dw_t, eps_t = sample_true_params()
        try:
            r = simulate_inversion(T1_t, T2_t, dw_t, eps_t)
        except Exception:
            continue
        if not (np.isfinite(r.T1_s) and np.isfinite(r.T2_s)):
            continue
        rec["T1_true"].append(T1_t);      rec["T1_rec"].append(r.T1_s);            rec["T1_sig"].append(r.T1_sigma_s)
        rec["T2_true"].append(T2_t);      rec["T2_rec"].append(r.T2_s);            rec["T2_sig"].append(r.T2_sigma_s)
        rec["dw_true"].append(abs(dw_t)); rec["dw_rec"].append(abs(r.delta_omega)); rec["dw_sig"].append(r.delta_omega_sigma)
        rec["eps_true"].append(eps_t);    rec["eps_rec"].append(r.epsilon_sx);     rec["eps_sig"].append(r.epsilon_sx_sigma)
        rec["t1_resid"].append(r.t1_residual)
        rec["ram_resid"].append(r.ramsey_residual)
        rec["gate_resid"].append(r.gate_residual)
        successes += 1
    return rec


def save_csv(rec):
    path = DATA_DIR / "parity_results.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rec.keys()))
        w.writeheader()
        n = len(rec["T1_true"])
        for i in range(n):
            w.writerow({k: rec[k][i] for k in rec})
    return path


def make_figure(rec):
    T1_t = np.array(rec["T1_true"]) * 1e6;  T1_r = np.array(rec["T1_rec"]) * 1e6;  T1_s = np.array(rec["T1_sig"]) * 1e6
    T2_t = np.array(rec["T2_true"]) * 1e6;  T2_r = np.array(rec["T2_rec"]) * 1e6;  T2_s = np.array(rec["T2_sig"]) * 1e6
    dw_t = np.array(rec["dw_true"]) / (2*np.pi*1e3); dw_r = np.array(rec["dw_rec"]) / (2*np.pi*1e3); dw_s = np.array(rec["dw_sig"]) / (2*np.pi*1e3)
    eps_t = np.array(rec["eps_true"]); eps_r = np.array(rec["eps_rec"]); eps_s = np.array(rec["eps_sig"])

    fig = plt.figure(figsize=(7.16, 6.5))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.40,
                            left=0.11, right=0.97, top=0.92, bottom=0.09)

    panels = [
        (gs[0,0], T1_t, T1_r, T1_s,
         r"True $T_1$ ($\mu$s)", r"Recovered $T_1$ ($\mu$s)", "(a)", None),
        (gs[0,1], T2_t, T2_r, T2_s,
         r"True $T_2$ ($\mu$s)", r"Recovered $T_2$ ($\mu$s)", "(b)", None),
        (gs[1,0], dw_t, dw_r, dw_s,
         r"True $|\Delta\omega|/2\pi$ (kHz)",
         r"Recovered $|\Delta\omega|/2\pi$ (kHz)", "(c)",
         "Sign ambiguity removed;\n$|\\Delta\\omega|$ plotted.\nOperating range: $|\\Delta\\omega|/2\\pi \\leq 5.6$ kHz."),
        (gs[1,1], eps_t, eps_r, eps_s,
         r"True $\varepsilon_{sx}$", r"Recovered $\varepsilon_{sx}$", "(d)", None),
    ]

    for (spec, tv, rv, sv, xl, yl, lab, note) in panels:
        ax = fig.add_subplot(spec)
        r2, rmse = compute_metrics(tv, rv)
        lo = min(tv.min(), rv.min()) * 0.90
        hi = max(tv.max(), rv.max()) * 1.10
        ax.errorbar(tv, rv, yerr=sv, fmt="o", color="#1f77b4",
                    markersize=3.2, linewidth=0, elinewidth=0.55, capsize=1.4, alpha=0.60)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, zorder=5)
        ax.set_xlabel(xl, fontsize=8.5); ax.set_ylabel(yl, fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.4, alpha=0.45)
        rmse_fmt = f"{rmse:.2e}" if "varepsilon" in xl else f"{rmse:.3f}"
        ax.text(0.04, 0.96, f"$R^2={r2:.4f}$\nRMSE$={rmse_fmt}$",
                transform=ax.transAxes, fontsize=7.5, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#cccccc", linewidth=0.6))
        ax.text(-0.16, 1.05, lab, transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", va="top")
        if note:
            ax.text(0.04, 0.55, note, transform=ax.transAxes, fontsize=6.5,
                    color="#555555", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#fffbe6",
                              edgecolor="#cccc88", linewidth=0.5))

    total = 3*SHOTS_T1_GATE + 3*SHOTS_RAMSEY + 3*SHOTS_T1_GATE
    fig.suptitle(
        f"Fig. 2 — Parameter Recovery Accuracy (Digital Twin, Superconducting, "
        f"$N={len(rec['T1_true'])}$ instances, {total:,} shots/qubit, seed={SEED})",
        fontsize=8.5, y=0.97)

    out = FIGURES_DIR / "fig2_parity.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    ts    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = 3*SHOTS_T1_GATE + 3*SHOTS_RAMSEY + 3*SHOTS_T1_GATE
    c     = inv.ARCH_DEFAULTS[ARCHITECTURE]
    print(f"[{ts}] Parity experiment — Digital Twin")
    print(f"  Architecture : {ARCHITECTURE}")
    print(f"  Instances    : {N_INSTANCES}  seed={SEED}")
    print(f"  Circuits     : 9 total  (3 T1 + 3 Ramsey + 3 gate rep)")
    print(f"  Shots        : T1/gate={SHOTS_T1_GATE}  Ramsey={SHOTS_RAMSEY}  "
          f"total={total:,}/qubit")
    ramsey_times = [f"{r*c['T2_s']*1e6:.1f}µs" for r in [0.15, 0.50, 1.00]]
    print(f"  Ramsey points: [0.15, 0.50, 1.00] x T2_prior = {ramsey_times}")
    print(f"  Δω range     : ±5.0 kHz  (anti-aliasing limit: 5.6 kHz)")

    rec      = run_experiment()
    csv_path = save_csv(rec)
    fig_path = make_figure(rec)

    print(f"\n  Results (n={len(rec['T1_true'])}):")
    for name, tv, rv, unit in [
        ("T1",   np.array(rec["T1_true"])*1e6,           np.array(rec["T1_rec"])*1e6,            "µs"),
        ("T2",   np.array(rec["T2_true"])*1e6,           np.array(rec["T2_rec"])*1e6,            "µs"),
        ("|Δω|", np.array(rec["dw_true"])/(2*np.pi*1e3), np.array(rec["dw_rec"])/(2*np.pi*1e3), "kHz"),
        ("ε",    np.array(rec["eps_true"]),               np.array(rec["eps_rec"]),               ""),
    ]:
        r2, rmse = compute_metrics(tv, rv)
        print(f"    {name:5s}  R²={r2:.4f}  RMSE={rmse:.3e} {unit}")

    print(f"\n  Data  : {csv_path}")
    print(f"  Figure: {fig_path}")