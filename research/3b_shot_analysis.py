import pathlib, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d

SCRIPT_DIR  = pathlib.Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

CSV_PATH = DATA_DIR / "shot_ablation_results.csv"

ARCHITECTURES = ["superconducting", "trapped_ion", "neutral_atom"]
PARAMS        = ["T1", "T2", "dw", "eps"]
PARAM_KEYS    = {"T1": "T1_r2", "T2": "T2_r2", "dw": "dw_r2", "eps": "eps_r2"}
PARAM_LABELS  = {
    "T1":  r"$T_1$",
    "T2":  r"$T_2$",
    "dw":  r"$|\Delta\omega|$",
    "eps": r"$\varepsilon_{sx}$",
}
ARCH_LABELS = {
    "superconducting": "Superconducting",
    "trapped_ion":     "Trapped ion",
    "neutral_atom":    "Neutral atom",
}

R2_THRESHOLD  = 0.94
SMOOTH_WINDOW = 21
DESIGN_BUDGET = 9900


def load_csv():
    data = {arch: {"budget": [], "T1_r2": [], "T2_r2": [], "dw_r2": [], "eps_r2": []}
            for arch in ARCHITECTURES}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            arch = row["architecture"]
            if arch not in data:
                continue
            data[arch]["budget"].append(int(row["budget_actual"]))
            for p in PARAMS:
                val = row[PARAM_KEYS[p]]
                data[arch][PARAM_KEYS[p]].append(
                    float(val) if val not in ("", "nan") else float("nan"))
    for arch in ARCHITECTURES:
        idx = np.argsort(data[arch]["budget"])
        data[arch]["budget"] = np.array(data[arch]["budget"])[idx]
        for p in PARAMS:
            data[arch][PARAM_KEYS[p]] = np.array(data[arch][PARAM_KEYS[p]])[idx]
    return data


def smooth(arr, window=SMOOTH_WINDOW):
    valid = np.isfinite(arr)
    tmp   = arr.copy()
    tmp[~valid] = np.nanmean(arr)
    out = uniform_filter1d(tmp, size=window, mode="nearest")
    out[~valid] = float("nan")
    return out


def find_n_star_last_crossing(budgets, r2_smooth, threshold=R2_THRESHOLD):
    above      = r2_smooth >= threshold
    last_below = np.where(~above)[0]
    if len(last_below) == 0:
        return int(budgets[0])
    last_below_idx = int(last_below[-1])
    if last_below_idx + 1 >= len(budgets):
        return None
    return int(budgets[last_below_idx + 1])


def build_worst_case(data):
    ref_budgets = data[ARCHITECTURES[0]]["budget"]
    worst       = np.full(len(ref_budgets), np.inf)
    binding     = [("", "") for _ in range(len(ref_budgets))]

    for arch in ARCHITECTURES:
        for p in PARAMS:
            r2_raw    = data[arch][PARAM_KEYS[p]]
            r2_smooth = smooth(r2_raw)
            for i, val in enumerate(r2_smooth):
                if np.isfinite(val) and val < worst[i]:
                    worst[i]   = val
                    binding[i] = (arch, p)

    worst[~np.isfinite(worst)] = float("nan")
    return ref_budgets, worst, binding


def find_per_arch_binding(data):
    results = {}
    for arch in ARCHITECTURES:
        budgets  = data[arch]["budget"]
        worst_r2 = np.full(len(budgets), np.inf)
        for p in PARAMS:
            r2_smooth = smooth(data[arch][PARAM_KEYS[p]])
            for i, v in enumerate(r2_smooth):
                if np.isfinite(v) and v < worst_r2[i]:
                    worst_r2[i] = v
        worst_r2[~np.isfinite(worst_r2)] = float("nan")
        worst_smooth = smooth(worst_r2)
        ns = find_n_star_last_crossing(budgets, worst_smooth)

        binding_param = None
        if ns is not None:
            idx = np.searchsorted(budgets, ns)
            min_r2 = np.inf
            for p in PARAMS:
                r2_smooth = smooth(data[arch][PARAM_KEYS[p]])
                if idx < len(r2_smooth) and r2_smooth[idx] < min_r2:
                    min_r2 = r2_smooth[idx]
                    binding_param = p

        results[arch] = {
            "budgets":       budgets,
            "worst_smooth":  worst_smooth,
            "n_star":        ns,
            "binding_param": binding_param,
        }
    return results


def print_results(data, arch_results, global_ns):
    print()
    print("="*72)
    print("  SHOT EFFICIENCY — RIGOROUS RESULTS")
    print(f"  Threshold : R² ≥ {R2_THRESHOLD}  |  Design budget: {DESIGN_BUDGET:,} shots")
    print(f"  Smoothing : {SMOOTH_WINDOW}-point uniform filter (last-crossing N* definition)")
    print("="*72)
    print()
    print(f"  Per-architecture worst-case N*")
    print(f"  (N* = last shot count where any parameter falls below R²={R2_THRESHOLD})")
    print()
    print(f"  {'Architecture':18s}  {'N* (shots)':>12}  "
          f"{'Binding param':>16}  {'Safety vs design':>18}")
    print("  " + "-"*68)
    for arch in ARCHITECTURES:
        r      = arch_results[arch]
        ns     = r["n_star"]
        bp     = PARAM_LABELS.get(r["binding_param"], "—") if r["binding_param"] else "—"
        safety = f"{DESIGN_BUDGET/ns:.2f}×" if ns else "N/A"
        print(f"  {ARCH_LABELS[arch]:18s}  {ns if ns else 'N/A':>12,}  "
              f"{bp:>16}  {safety:>18}")

    print()
    print(f"  Global N* (worst case across all archs + params): {global_ns:,} shots")
    print(f"  Design budget:                                    {DESIGN_BUDGET:,} shots")
    print(f"  Safety margin:                                    "
          f"{DESIGN_BUDGET/global_ns:.2f}× N*")
    print()
    print("  Paper statement:")
    print(f"  N* = {global_ns:,} shots is the minimum budget at which all four")
    print(f"  Lindblad parameters simultaneously achieve R² ≥ {R2_THRESHOLD} across")
    print(f"  all three architectures. The design budget of {DESIGN_BUDGET:,} shots")
    print(f"  operates at {DESIGN_BUDGET/global_ns:.2f}× N*, providing a robustness margin")
    print(f"  above the simulation-derived minimum.")
    print()


def make_figure(data, arch_results, global_ns):
    budgets_global, worst_global, _ = build_worst_case(data)
    worst_global_smooth = smooth(worst_global, window=SMOOTH_WINDOW)

    fig, ax = plt.subplots(figsize=(7.16, 4.2))

    ax.plot(budgets_global, worst_global_smooth,
            color="#1a1a2e", linewidth=2.2, zorder=6,
            label=r"Worst-case $R^2$ (min over all arch. $\times$ params)")

    ax.axhline(y=R2_THRESHOLD, color="#c0392b", linewidth=1.2,
               linestyle="--", zorder=4,
               label=f"Threshold $R^2 = {R2_THRESHOLD}$")

    ax.axvline(x=global_ns, color="#e67e22", linewidth=1.4,
               linestyle="-.", zorder=5,
               label=f"Global $N^* = {global_ns:,}$ shots")

    ax.axvline(x=DESIGN_BUDGET, color="#27ae60", linewidth=1.4,
               linestyle="--", zorder=5,
               label=f"Design budget $= {DESIGN_BUDGET:,}$ shots")

    ax.axvspan(global_ns, DESIGN_BUDGET,
               alpha=0.10, color="#27ae60", zorder=3)

    mid    = (global_ns + DESIGN_BUDGET) / 2
    y_ann  = R2_THRESHOLD - 0.03
    ax.annotate(
        f"Safety margin\n{DESIGN_BUDGET/global_ns:.2f}× $N^*$",
        xy=(mid, y_ann),
        xytext=(mid, y_ann - 0.07),
        ha="center", va="top", fontsize=7.5, color="#1a6e38",
        arrowprops=dict(arrowstyle="-", color="#1a6e38",
                        lw=0.8, linestyle="dashed"),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#1a6e38", linewidth=0.6, alpha=0.9))

    arch_colors = {"superconducting": "#1f77b4",
                   "trapped_ion":     "#ff7f0e",
                   "neutral_atom":    "#2ca02c"}
    for arch in ARCHITECTURES:
        r  = arch_results[arch]
        ns = r["n_star"]
        if ns is None:
            continue
        ax.axvline(x=ns, color=arch_colors[arch], linewidth=0.9,
                   linestyle=":", alpha=0.7, zorder=4)
        ax.text(ns, R2_THRESHOLD + 0.005, ARCH_LABELS[arch][:4],
                ha="center", va="bottom", fontsize=6.5,
                color=arch_colors[arch], rotation=90,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                          edgecolor="none", alpha=0.7))

    ax.fill_between(budgets_global, worst_global_smooth,
                    alpha=0.06, color="#1a1a2e", zorder=2)

    ax.set_xlim(1000, 20000)
    ax.set_ylim(0.55, 1.03)
    ax.set_xlabel("Total shot budget (shots/qubit)", fontsize=9.0)
    ax.set_ylabel(r"Worst-case $R^2$  (min over all arch. $\times$ params)", fontsize=9.0)
    ax.tick_params(labelsize=8.0)
    ax.grid(True, which="major", linewidth=0.3, alpha=0.4)

    ax.legend(loc="lower right", fontsize=7.8, framealpha=0.95,
              edgecolor="#cccccc", handlelength=2.0)

    ax.set_title(
        f"Fig. 4 — Shot-Budget Efficiency  "
        f"($N^* = {global_ns:,}$ shots,  design = {DESIGN_BUDGET:,} shots,  "
        f"safety = {DESIGN_BUDGET/global_ns:.2f}× $N^*$)",
        fontsize=8.5, pad=7)

    fig.tight_layout()
    out_pdf = FIGURES_DIR / "fig4_efficiency.pdf"
    out_png = FIGURES_DIR / "fig4_efficiency.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


if __name__ == "__main__":
    print(f"Loading {CSV_PATH} ...")
    data         = load_csv()
    arch_results = find_per_arch_binding(data)

    budgets_g, worst_g, _ = build_worst_case(data)
    worst_g_smooth         = smooth(worst_g)
    global_ns              = find_n_star_last_crossing(budgets_g, worst_g_smooth)

    print_results(data, arch_results, global_ns)
    fig_path = make_figure(data, arch_results, global_ns)
    print(f"  Figure: {fig_path}")