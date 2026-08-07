import importlib.util, pathlib, sys, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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

SEED          = 42
N_INSTANCES   = 100
ARCHITECTURES = ["superconducting", "trapped_ion", "neutral_atom"]

BASELINE_T1     = 300
BASELINE_RAMSEY = 1000
BASELINE_GATE   = 500
BASELINE_ECHO   = 500
_N_T1  = 3
_N_RAM = 6
_N_GAT = 3
_N_ECH = 3
BASELINE_BUDGET = (_N_T1*BASELINE_T1 + _N_RAM*BASELINE_RAMSEY
                   + _N_GAT*BASELINE_GATE + _N_ECH*BASELINE_ECHO)

_FRAC_T1  = (_N_T1  * BASELINE_T1)    / BASELINE_BUDGET
_FRAC_RAM = (_N_RAM * BASELINE_RAMSEY) / BASELINE_BUDGET
_FRAC_GAT = (_N_GAT * BASELINE_GATE)  / BASELINE_BUDGET
_FRAC_ECH = (_N_ECH * BASELINE_ECHO)  / BASELINE_BUDGET

BUDGET_MIN  = 1000
BUDGET_MAX  = 20000
BUDGET_STEP = 50
BUDGETS     = list(range(BUDGET_MIN, BUDGET_MAX + 1, BUDGET_STEP))

ARCH_COLORS = {
    "superconducting": "#1f77b4",
    "trapped_ion":     "#ff7f0e",
    "neutral_atom":    "#2ca02c",
}
ARCH_LABELS = {
    "superconducting": "Superconducting",
    "trapped_ion":     "Trapped ion",
    "neutral_atom":    "Neutral atom",
}
PARAM_STYLES = {"T1": "-", "T2": "--", "dw": "-.", "eps": ":"}
PARAM_LABELS = {
    "T1":  r"$T_1$",
    "T2":  r"$T_2$",
    "dw":  r"$|\Delta\omega|$",
    "eps": r"$\varepsilon_{sx}$",
}


def shots_from_budget(total):
    st1 = max(10, round(total * _FRAC_T1  / _N_T1))
    sr  = max(10, round(total * _FRAC_RAM / _N_RAM))
    sg  = max(10, round(total * _FRAC_GAT / _N_GAT))
    se  = max(10, round(total * _FRAC_ECH / _N_ECH))
    return st1, sr, sg, se


def actual_budget(st1, sr, sg, se):
    return _N_T1*st1 + _N_RAM*sr + _N_GAT*sg + _N_ECH*se


def _spawn_rngs(arch_idx, bud_idx, iid):
    ss = np.random.SeedSequence([SEED, arch_idx, bud_idx, iid])
    t1_ss, ram_ss, gat_ss, ech_ss = ss.spawn(4)
    return (np.random.default_rng(t1_ss),
            np.random.default_rng(ram_ss),
            np.random.default_rng(gat_ss),
            np.random.default_rng(ech_ss))


def sample_true_params(arch, dw_max, rng_obj):
    c   = inv.ARCH_DEFAULTS[arch]
    T1  = c["T1_s"]        * rng_obj.uniform(0.6, 1.4)
    T2  = min(c["T2_s"]    * rng_obj.uniform(0.6, 1.4), 1.95*T1)
    dw  = rng_obj.choice([-1,1]) * rng_obj.uniform(0.2*dw_max, dw_max)
    eps = c["eps_typical"] * rng_obj.uniform(0.3, 5.0)
    return T1, T2, dw, eps


def simulate_one(arch, T1_t, T2_t, dw_t, eps_t,
                 st1, sr, sg, se, arch_idx, bud_idx, iid):
    profile = inv.BackendProfile.from_architecture(arch)
    arch_c  = inv.ARCH_DEFAULTS[arch]
    _, meta = inv.build_probe_circuits(profile)

    rng_t1, rng_ram, rng_gat, rng_ech = _spawn_rngs(arch_idx, bud_idx, iid)

    p0 = arch_c["p0_given_1"]
    p1 = arch_c["p1_given_0"]
    def spam(p): return p*(1-p0) + (1-p)*p1

    def c1(rng_obj, p, sh):
        n1 = int(rng_obj.binomial(sh, float(np.clip(p, 0, 1))))
        return {"0": sh-n1, "1": n1}
    def c0(rng_obj, p, sh):
        n0 = int(rng_obj.binomial(sh, float(np.clip(p, 0, 1))))
        return {"0": n0, "1": sh-n0}

    t1_counts = [c1(rng_t1, spam(float(inv.forward_t1(d, T1_t))), st1)
                 for d in meta["t1_delays_s"]]

    ramsey_counts = []
    for t in meta["ramsey_delays_s"]:
        px, py = inv.forward_ramsey_xy(t, T2_t, dw_t)
        ramsey_counts.append(c1(rng_ram, px, sr))
        ramsey_counts.append(c1(rng_ram, py, sr))

    gate_counts = [c0(rng_gat, spam(p), sg)
                   for p in inv.forward_gate(
                       np.array(inv.GATE_REP_N_DT, float), eps_t)]

    T2_echo_t = min(T2_t, 2.0*T1_t)
    echo_counts = [c1(rng_ech, spam(float(inv.forward_echo(t, T2_echo_t))), se)
                   for t in meta["echo_delays_s"]]

    return inv.lindblad_inversion(
        t1_counts + ramsey_counts + gate_counts + echo_counts,
        meta, profile,
        shots_t1=st1, shots_ramsey=sr, shots_gate=sg, shots_echo=se,
        qubit_id=0, timestamp=datetime.now(timezone.utc).isoformat())


def r2(true_vals, rec_vals):
    t = np.asarray(true_vals, float)
    r = np.asarray(rec_vals,  float)
    ss_res = np.sum((r-t)**2)
    ss_tot = np.sum((t-np.mean(t))**2)
    return float(1.0 - ss_res/ss_tot) if ss_tot > 0 else 0.0


def run_ablation():
    results = {}
    for arch_idx, arch in enumerate(ARCHITECTURES):
        profile   = inv.BackendProfile.from_architecture(arch)
        dw_max    = profile.dw_max_rad_s
        param_rng = np.random.default_rng(
            np.random.SeedSequence([SEED, arch_idx, 9999]))
        true_params = [sample_true_params(arch, dw_max, param_rng)
                       for _ in range(N_INSTANCES)]

        for bud_idx, budget in enumerate(BUDGETS):
            st1, sr, sg, se = shots_from_budget(budget)
            ab = actual_budget(st1, sr, sg, se)

            T1_t=[]; T1_r=[]; T2_t=[]; T2_r=[]
            dw_t=[]; dw_r=[]; ep_t=[]; ep_r=[]
            fails = 0

            for iid, (T1_true, T2_true, dw_true, eps_true) in enumerate(true_params):
                try:
                    res = simulate_one(arch, T1_true, T2_true, dw_true, eps_true,
                                       st1, sr, sg, se, arch_idx, bud_idx, iid)
                except (RuntimeError, ValueError):
                    fails += 1
                    continue
                if not (np.isfinite(res.T1_s) and np.isfinite(res.T2_s)):
                    fails += 1
                    continue
                T1_t.append(T1_true);          T1_r.append(res.T1_s)
                T2_t.append(T2_true);          T2_r.append(res.T2_s)
                dw_t.append(abs(dw_true));     dw_r.append(abs(res.delta_omega))
                ep_t.append(eps_true);         ep_r.append(res.epsilon_sx)

            results[(arch, bud_idx)] = {
                "budget_actual": ab,
                "T1_r2":  r2(T1_t, T1_r) if T1_r else float("nan"),
                "T2_r2":  r2(T2_t, T2_r) if T2_r else float("nan"),
                "dw_r2":  r2(dw_t, dw_r) if dw_r else float("nan"),
                "eps_r2": r2(ep_t, ep_r) if ep_r else float("nan"),
                "n":      len(T1_r),
                "fails":  fails,
            }

            if bud_idx % 40 == 0 or budget == BASELINE_BUDGET:
                rec = results[(arch, bud_idx)]
                print(f"  {arch[:14]:14s}  budget={ab:6d}  "
                      f"T2_R²={rec['T2_r2']:.4f}  n={rec['n']}")

    return results


def save_csv(results):
    rows = []
    for (arch, bud_idx), rec in results.items():
        rows.append({
            "architecture":   arch,
            "budget_nominal": BUDGETS[bud_idx],
            "budget_actual":  rec["budget_actual"],
            "n_success":      rec["n"],
            "fit_failures":   rec["fails"],
            "T1_r2":          rec["T1_r2"],
            "T2_r2":          rec["T2_r2"],
            "dw_r2":          rec["dw_r2"],
            "eps_r2":         rec["eps_r2"],
        })
    path = DATA_DIR / "shot_ablation_results.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def make_figure(results):
    fig, ax = plt.subplots(figsize=(7.16, 4.8))

    for arch in ARCHITECTURES:
        budgets_plot = []
        curves = {"T1": [], "T2": [], "dw": [], "eps": []}
        for bud_idx in range(len(BUDGETS)):
            rec = results[(arch, bud_idx)]
            if rec["n"] < 5:
                continue
            budgets_plot.append(rec["budget_actual"])
            curves["T1"].append(rec["T1_r2"])
            curves["T2"].append(rec["T2_r2"])
            curves["dw"].append(rec["dw_r2"])
            curves["eps"].append(rec["eps_r2"])

        bp = np.array(budgets_plot)
        col = ARCH_COLORS[arch]
        for pkey in ["T1", "T2", "dw", "eps"]:
            ax.plot(bp, np.array(curves[pkey]),
                    color=col, linestyle=PARAM_STYLES[pkey],
                    linewidth=1.0, alpha=0.85)

    ax.axvline(x=BASELINE_BUDGET, color="#333333", linestyle="--",
               linewidth=1.0, zorder=5)
    ax.axhline(y=0.94, color="#888888", linestyle=":", linewidth=0.8, zorder=4)

    arch_handles = [
        mlines.Line2D([], [], color=ARCH_COLORS[a], linewidth=2.0,
                      label=ARCH_LABELS[a])
        for a in ARCHITECTURES
    ]
    param_handles = [
        mlines.Line2D([], [], color="#333333",
                      linestyle=PARAM_STYLES[p], linewidth=1.5,
                      label=PARAM_LABELS[p])
        for p in ["T1", "T2", "dw", "eps"]
    ]
    ref_handles = [
        mlines.Line2D([], [], color="#333333", linestyle="--",
                      linewidth=1.0, label=f"Design ({BASELINE_BUDGET:,} shots)"),
        mlines.Line2D([], [], color="#888888", linestyle=":",
                      linewidth=0.8, label=r"$R^2 = 0.94$"),
    ]

    leg1 = ax.legend(handles=arch_handles,
                     loc="lower left", bbox_to_anchor=(0.01, 0.01),
                     fontsize=7.5, framealpha=0.92, edgecolor="#cccccc",
                     title="Architecture", title_fontsize=7.5)
    ax.add_artist(leg1)
    ax.legend(handles=param_handles + ref_handles,
              loc="lower right", bbox_to_anchor=(0.99, 0.01),
              fontsize=7.5, framealpha=0.92, edgecolor="#cccccc",
              title="Parameter / Reference", title_fontsize=7.5)

    ax.set_xlim(BUDGET_MIN * 0.95, BUDGET_MAX * 1.02)
    ax.set_ylim(-0.08, 1.05)
    ax.set_xlabel("Total shot budget (shots/qubit)", fontsize=9.0)
    ax.set_ylabel(r"$R^2$", fontsize=9.0)
    ax.tick_params(labelsize=8.0)
    ax.grid(True, which="major", linewidth=0.3, alpha=0.45)
    ax.set_title(
        f"Fig. 4 — Shot-Budget Efficiency  "
        f"(3 architectures × 4 parameters,  $N={N_INSTANCES}$/budget,  seed={SEED})",
        fontsize=8.5, pad=7)

    fig.tight_layout()
    out_pdf = FIGURES_DIR / "fig4_shotbudget.pdf"
    out_png = FIGURES_DIR / "fig4_shotbudget.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


def print_results(results):
    baseline_bud_idx = BUDGETS.index(BASELINE_BUDGET)
    print()
    print("="*70)
    print("  SHOT ABLATION — RESULTS AT DESIGN BUDGET")
    print("="*70)
    for arch in ARCHITECTURES:
        rec = results[(arch, baseline_bud_idx)]
        print(f"\n  {arch}  (n={rec['n']}, budget={rec['budget_actual']})")
        print(f"    T1  R²={rec['T1_r2']:.4f}")
        print(f"    T2  R²={rec['T2_r2']:.4f}")
        print(f"    Δω  R²={rec['dw_r2']:.4f}")
        print(f"    ε   R²={rec['eps_r2']:.4f}")
        print(f"    Fail rate: {rec['fails']}/{N_INSTANCES}")

    print()
    print("  First budget where T2 R²≥0.94 per architecture:")
    for arch in ARCHITECTURES:
        for bud_idx in range(len(BUDGETS)):
            rec = results[(arch, bud_idx)]
            if rec["n"] > 0 and rec["T2_r2"] >= 0.94:
                print(f"    {arch}: {rec['budget_actual']:,} shots "
                      f"(nominal {BUDGETS[bud_idx]:,})")
                break

    print()
    print("  First budget where ALL 4 params R²≥0.94 per architecture:")
    for arch in ARCHITECTURES:
        for bud_idx in range(len(BUDGETS)):
            rec = results[(arch, bud_idx)]
            if (rec["n"] > 0
                    and all(rec[k] >= 0.94
                            for k in ["T1_r2","T2_r2","dw_r2","eps_r2"])):
                print(f"    {arch}: {rec['budget_actual']:,} shots "
                      f"(nominal {BUDGETS[bud_idx]:,})")
                break
    print()


if __name__ == "__main__":
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"[{ts}] Fig. 4 — Shot-Budget Ablation")
    print(f"  Architectures : {', '.join(ARCHITECTURES)}")
    print(f"  Budget range  : {BUDGET_MIN}–{BUDGET_MAX} step={BUDGET_STEP}")
    print(f"  Budget levels : {len(BUDGETS)}")
    print(f"  N per level   : {N_INSTANCES}/architecture")
    print(f"  Seed          : {SEED}  (distinct from Fig.2=43, Fig.3=44)")
    print(f"  Design budget : {BASELINE_BUDGET} shots")
    print(f"  Shot fractions: T1={_FRAC_T1:.2%}  Ramsey={_FRAC_RAM:.2%}  "
          f"Gate={_FRAC_GAT:.2%}  Echo={_FRAC_ECH:.2%}")
    print()
    print("  Selected per-circuit allocations:")
    print(f"  {'Nominal':>8}  {'Actual':>8}  {'T1':>5}  {'Ramsey':>7}  "
          f"{'Gate':>6}  {'Echo':>6}")
    for b in [1000, 2500, 5000, 9900, 15000, 20000]:
        if b not in BUDGETS:
            continue
        st1, sr, sg, se = shots_from_budget(b)
        ab = actual_budget(st1, sr, sg, se)
        mark = " ← design" if b == BASELINE_BUDGET else ""
        print(f"  {b:>8,}  {ab:>8,}  {st1:>5}  {sr:>7}  {sg:>6}  {se:>6}{mark}")
    print()
    print("  Running ablation (this will take a while)...")
    print()

    results  = run_ablation()
    print_results(results)
    csv_path = save_csv(results)
    fig_path = make_figure(results)
    print(f"  Data  : {csv_path}")
    print(f"  Figure: {fig_path}")