"""
3b_shot_analysis.py

Location : research/3b_shot_analysis.py
Reads    : research/data/fig4_shot_ablation.csv
           Columns: shots, architecture, parameter, r2

Produces :
  research/figures/fig3b_shot_ablation.pdf  —  2×2 panel figure,
  one subplot per parameter, R² vs total shot budget under realistic
  hardware noise (SC/TI live-cal ±15%, NA arch-default), three curves
  per subplot (one per architecture). Horizontal dashed line at R²=0.95.
  Red dotted vertical at the minimum budget where all three architectures
  simultaneously reach R²≥0.95.

Prints to terminal:
  For each parameter, the minimum total shot budget where R²≥0.95 holds
  for ALL three architectures simultaneously, plus the implied per-probe
  shot counts at that budget. Ends with the binding budget — the single
  total that satisfies every parameter at once.
"""

import pathlib, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── paths (script lives in research/, data in research/data/) ────────────────
SCRIPT_DIR  = pathlib.Path(__file__).resolve().parent
CSV_PATH    = SCRIPT_DIR / "data" / "fig4_shot_ablation.csv"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ── protocol constants — must match 2_parity_experiments.py ──────────────────
DEFAULT_TOTAL        = 9_900
DEFAULT_SHOTS_T1     = 300
DEFAULT_SHOTS_RAMSEY = 1_000
DEFAULT_SHOTS_GATE   = 500
DEFAULT_SHOTS_ECHO   = 500
N_T1 = 3; N_RAMSEY = 6; N_GATE = 3; N_ECHO = 3

ARCHITECTURES = ["superconducting", "trapped_ion", "neutral_atom"]
PARAMETERS    = ["T1", "T2", "delta_omega", "epsilon_sx"]

ARCH_LABELS = {"superconducting": "Superconducting",
               "trapped_ion":     "Trapped ion",
               "neutral_atom":    "Neutral atom"}
PARAM_LABELS = {"T1":          r"$T_1$",
                "T2":          r"$T_2$",
                "delta_omega": r"$|\Delta\omega|$",
                "epsilon_sx":  r"$\varepsilon_{sx}$"}
PARAM_TITLES = {"T1":          r"$T_1$ Recovery",
                "T2":          r"$T_2$ Recovery",
                "delta_omega": r"$|\Delta\omega|$ Recovery",
                "epsilon_sx":  r"$\varepsilon_{sx}$ Recovery"}

ARCH_COLOR  = {"superconducting": "#1f77b4",
               "trapped_ion":     "#ff7f0e",
               "neutral_atom":    "#2ca02c"}
ARCH_LS     = {"superconducting": "-",
               "trapped_ion":     "--",
               "neutral_atom":    ":"}
ARCH_MARKER = {"superconducting": "o",
               "trapped_ion":     "s",
               "neutral_atom":    "^"}

R2_THRESHOLD = 0.95


def shots_from_budget(budget):
    """Proportional per-probe shot counts at a given total budget."""
    scale = budget / DEFAULT_TOTAL
    st1  = max(1, round(DEFAULT_SHOTS_T1     * scale))
    sram = max(1, round(DEFAULT_SHOTS_RAMSEY  * scale))
    sg   = max(1, round(DEFAULT_SHOTS_GATE    * scale))
    se   = max(1, round(DEFAULT_SHOTS_ECHO    * scale))
    return st1, sram, sg, se


def find_min_budget_all_archs(df, param, threshold=R2_THRESHOLD):
    """
    Minimum total shot budget where R² >= threshold simultaneously
    for all three architectures. Returns None if never achieved.
    """
    sub     = df[df["parameter"] == param]
    budgets = sorted(sub["shots"].unique())
    for b in budgets:
        at_b = sub[sub["shots"] == b]
        if all(
            len(at_b[at_b["architecture"] == arch]["r2"].values) > 0
            and at_b[at_b["architecture"] == arch]["r2"].values[0] >= threshold
            for arch in ARCHITECTURES
        ):
            return b
    return None


def load_data():
    if not CSV_PATH.exists():
        sys.exit(f"ERROR: {CSV_PATH} not found.")
    df = pd.read_csv(CSV_PATH)
    missing = {"shots","architecture","parameter","r2"} - set(df.columns)
    if missing:
        sys.exit(f"ERROR: missing columns: {missing}")
    df["shots"] = df["shots"].astype(int)
    df["r2"]    = df["r2"].astype(float)
    return df


def print_report(df):
    print()
    print("=" * 72)
    print(f"  R² ≥ {R2_THRESHOLD} FOR ALL ARCHITECTURES SIMULTANEOUSLY — PER PARAMETER")
    print("=" * 72)

    binding = 0
    for param in PARAMETERS:
        b_star = find_min_budget_all_archs(df, param)
        print(f"\n  {PARAM_LABELS[param]}  ({param})")
        if b_star is None:
            print(f"    R²≥{R2_THRESHOLD} not reached within sweep range for all architectures.")
            continue

        st1, sram, sg, se = shots_from_budget(b_star)
        actual = N_T1*st1 + N_RAMSEY*sram + N_GATE*sg + N_ECHO*se
        print(f"    Min budget : {b_star:,} shots  (actual {actual:,})")
        print(f"    Per probe  :")
        print(f"      T1     : {N_T1} × {st1:4d} shots = {N_T1*st1:5d}")
        print(f"      Ramsey : {N_RAMSEY} × {sram:4d} shots = {N_RAMSEY*sram:5d}")
        print(f"      Gate   : {N_GATE} × {sg:4d} shots = {N_GATE*sg:5d}")
        print(f"      Echo   : {N_ECHO} × {se:4d} shots = {N_ECHO*se:5d}")
        print(f"    R² at {b_star:,} shots:")
        sub = df[(df["parameter"] == param) & (df["shots"] == b_star)]
        for arch in ARCHITECTURES:
            vals = sub[sub["architecture"] == arch]["r2"].values
            r2s  = f"{vals[0]:.4f}" if len(vals) else "n/a"
            ok   = " ✓" if len(vals) and vals[0] >= R2_THRESHOLD else " ✗"
            print(f"      {ARCH_LABELS[arch]:20s}: R²={r2s}{ok}")

        binding_arch = max(
            ARCHITECTURES,
            key=lambda a: (sub[sub["architecture"]==a]["r2"].values[0]
                           if len(sub[sub["architecture"]==a]["r2"]) else 0),
            # max of the budget required — find which arch crosses last
        )
        # correct logic: which arch has lowest r2 at b_star
        binding_arch = min(
            ARCHITECTURES,
            key=lambda a: (sub[sub["architecture"]==a]["r2"].values[0]
                           if len(sub[sub["architecture"]==a]["r2"]) else 1.0)
        )
        print(f"    Binding architecture: {ARCH_LABELS[binding_arch]}"
              f"  (lowest R² at threshold crossing)")
        binding = max(binding, b_star)

    print()
    print("-" * 72)
    if binding > 0:
        st1, sram, sg, se = shots_from_budget(binding)
        actual = N_T1*st1 + N_RAMSEY*sram + N_GATE*sg + N_ECHO*se
        print(f"  BINDING BUDGET  (satisfies ALL parameters simultaneously)")
        print(f"    Total  : {binding:,} shots  (actual {actual:,})")
        print(f"    T1     : {N_T1} × {st1:4d} shots = {N_T1*st1:5d}")
        print(f"    Ramsey : {N_RAMSEY} × {sram:4d} shots = {N_RAMSEY*sram:5d}")
        print(f"    Gate   : {N_GATE} × {sg:4d} shots = {N_GATE*sg:5d}")
        print(f"    Echo   : {N_ECHO} × {se:4d} shots = {N_ECHO*se:5d}")
        suffix = "sufficient ✓" if 9_900 >= binding else "BELOW binding budget ✗"
        print(f"    Deployed 9,900-shot budget is {suffix}")
    print("=" * 72)
    print()
    return binding


PARAM_COLOR = {"T1":          "#1f77b4",
               "T2":          "#d62728",
               "delta_omega": "#2ca02c",
               "epsilon_sx":  "#ff7f0e"}
PARAM_MARKER= {"T1":          "o",
               "T2":          "s",
               "delta_omega": "^",
               "epsilon_sx":  "D"}


def make_figure(df, binding):
    fig, ax = plt.subplots(figsize=(7.16, 4.20))

    budgets = sorted(df["shots"].unique())

    for param in PARAMETERS:
        sub  = df[df["parameter"] == param]
        stats = sub.groupby("shots")["r2"].agg(["mean","std"]).reindex(budgets)
        mean  = stats["mean"]
        std   = stats["std"]

        ax.fill_between(budgets,
                        mean.values - std.values,
                        mean.values + std.values,
                        color=PARAM_COLOR[param],
                        alpha=0.15,
                        linewidth=0)
        ax.plot(budgets, mean.values,
                color=PARAM_COLOR[param],
                linestyle="-",
                linewidth=1.4,
                marker=PARAM_MARKER[param],
                markersize=3.5,
                markevery=5,
                alpha=0.90,
                label=PARAM_LABELS[param])

    # R²=0.95 reference line
    ax.axhline(R2_THRESHOLD, color="#555555", linestyle="--",
               linewidth=0.9, zorder=1)
    ax.text(1050, R2_THRESHOLD + 0.003,
            r"$R^2 = 0.95$", fontsize=7.0, color="#555555", va="bottom")

    # deployed budget
    ax.axvline(9_900, color="#333333", linestyle="--",
               linewidth=1.0, zorder=2, alpha=0.6)
    ax.text(9_900 + 180, 0.695,
            "9,900", fontsize=6.5,
            color="#333333", va="bottom", rotation=90)

    ax.set_xlim(800, 15_500)
    ax.set_ylim(0.68, 1.005)
    ax.set_xlabel("Total shot budget", fontsize=8.5)
    ax.set_ylabel(r"$R^2$ (mean across architectures, realistic noise)",
                  fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, which="major", linewidth=0.35, alpha=0.45)

    ax.legend(fontsize=8.0, frameon=True, loc="lower right",
              framealpha=0.9, edgecolor="#cccccc",
              handlelength=2.2, handletextpad=0.5)

    fig.suptitle(
        r"Fig. 3b — Shot-Budget Efficiency Under Realistic Hardware Noise"
        "\n"
        r"(mean $R^2$ ± 1$\sigma$ across SC / TI / NA, $N=300$/arch, seed=45)",
        fontsize=8.5, y=1.01)

    fig.tight_layout()
    out = FIGURES_DIR / "fig3b_shot_ablation.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out}")
    return out


if __name__ == "__main__":
    df      = load_data()
    print(f"  Loaded {len(df):,} rows | "
          f"shots {df['shots'].min():,}–{df['shots'].max():,} | "
          f"{df['architecture'].nunique()} archs | "
          f"{df['parameter'].nunique()} params")
    binding = print_report(df)
    make_figure(df, binding)