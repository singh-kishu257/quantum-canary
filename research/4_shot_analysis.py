"""
Reads : research/data/fig4_shot_ablation.csv
Output: research/figures/fig4_efficiency.pdf

Single panel. Mean R² across all three architectures vs total shot budget,
for all four parameters. Rolling average applied to suppress finite-N
sampling noise. Deployed 9,900-shot budget marked.
"""

import pathlib, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR  = pathlib.Path(__file__).resolve().parent
CSV_PATH    = SCRIPT_DIR / "data" / "fig4_shot_ablation.csv"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

PARAMETERS = ["T1", "T2", "delta_omega", "epsilon_sx"]

PARAM_LABEL  = {"T1":          r"$T_1$",
                "T2":          r"$T_2$",
                "delta_omega": r"$|\Delta\omega|$",
                "epsilon_sx":  r"$\varepsilon_{sx}$"}
PARAM_COLOR  = {"T1":          "#1f77b4",
                "T2":          "#d62728",
                "delta_omega": "#2ca02c",
                "epsilon_sx":  "#ff7f0e"}
PARAM_MARKER = {"T1":          "o",
                "T2":          "s",
                "delta_omega": "^",
                "epsilon_sx":  "D"}

SMOOTH_WINDOW = 7   # rolling-average window (budget steps = 700 shots)
DEPLOYED      = 9_900


def load():
    if not CSV_PATH.exists():
        sys.exit(f"ERROR: {CSV_PATH} not found.")
    df = pd.read_csv(CSV_PATH)
    df["shots"] = df["shots"].astype(int)
    df["r2"]    = df["r2"].astype(float)
    return df


def make_figure(df):
    fig, ax = plt.subplots(figsize=(7.16, 3.80))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.14)

    budgets = sorted(df["shots"].unique())

    for param in PARAMETERS:
        sub  = df[df["parameter"] == param]
        mean = sub.groupby("shots")["r2"].mean().reindex(budgets).values

        # smoothed line
        smoothed = pd.Series(mean).rolling(
            SMOOTH_WINDOW, center=True, min_periods=1).mean().values

        # faint raw line
        ax.plot(budgets, mean,
                color=PARAM_COLOR[param],
                linestyle="-", linewidth=0.5,
                alpha=0.20, zorder=2)

        # smooth line with markers
        ax.plot(budgets, smoothed,
                color=PARAM_COLOR[param],
                linestyle="-", linewidth=1.6,
                marker=PARAM_MARKER[param],
                markersize=3.5, markevery=10,
                alpha=0.95, zorder=3,
                label=PARAM_LABEL[param])

    # R²=0.95 reference
    ax.axhline(0.95, color="#666666", linestyle="--", linewidth=0.9, zorder=1)
    ax.text(1000, 0.952, r"$R^2=0.95$",
            fontsize=7.0, color="#666666", va="bottom")

    # deployed budget
    ax.axvline(DEPLOYED, color="#333333", linestyle="--",
               linewidth=1.1, zorder=4, alpha=0.7)
    ax.text(DEPLOYED + 200, 0.705,
            f"{DEPLOYED:,} shots\n(deployed)",
            fontsize=6.5, color="#333333", va="bottom", rotation=90)

    ax.set_xlim(800, 15_500)
    ax.set_ylim(0.68, 1.005)
    ax.set_xlabel("Total shot budget", fontsize=8.5)
    ax.set_ylabel(r"$R^2$ (mean across SC / TI / NA, realistic noise)",
                  fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, which="major", linewidth=0.35, alpha=0.45)

    ax.legend(fontsize=8.5, frameon=True, loc="lower right",
              framealpha=0.9, edgecolor="#cccccc",
              handlelength=2.0, handletextpad=0.5,
              ncol=2, columnspacing=1.0)

    ax.set_title(
        r"Fig. 4 — Shot-Budget Efficiency  "
        r"($N=300$/arch, realistic noise, seed=45)  "
        r"[smoothed: 7-point rolling mean]",
        fontsize=7.5, pad=5)

    out = FIGURES_DIR / "fig4_efficiency.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    make_figure(load())