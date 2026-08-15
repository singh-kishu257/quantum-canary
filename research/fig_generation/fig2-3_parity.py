from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT     = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FW, FH = 3.6, 3.5

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8.0,
    "axes.labelsize": 9.0,
    "axes.linewidth": 0.7,
    "axes.axisbelow": True,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 8.0,
    "legend.frameon": False,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ARCHS       = ["superconducting", "trapped_ion", "neutral_atom"]
ARCH_LABELS = {"superconducting": "Superconducting",
               "trapped_ion": "Trapped ion",
               "neutral_atom": "Neutral atom"}
COLORS      = {"superconducting": "#1f77b4",
               "trapped_ion": "#ff7f0e",
               "neutral_atom": "#2ca02c"}
MARKERS     = {"superconducting": "o", "trapped_ion": "s", "neutral_atom": "^"}

PANELS = [
    ("T1",  r"$T_1$ (s)"),
    ("T2",  r"$T_2$ (s)"),
    ("dw",  r"$|\Delta\omega|$ (rad/s)"),
    ("eps", r"$\varepsilon_{sx}$"),
]

FIG2_TITLE = ("Parameter-Recovery Accuracy Under\nIdeal Markovian Simulation")
FIG3_TITLE = ("Parameter-Recovery Accuracy Under\nRealistic Hardware Mismatch")


def load(path):
    if not path.exists():
        sys.exit(f"ERROR: missing input file: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"ERROR: no data rows in {path}")
    rec = {"arch": np.array([r["arch"] for r in rows])}
    for key, _ in PANELS:
        for suf in ("true", "rec", "sig"):
            rec[f"{key}_{suf}"] = np.array(
                [float(r[f"{key}_{suf}"]) for r in rows], dtype=float)
    return rec


def stats(t, r):
    ss_res = np.sum((r - t) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    r2   = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((r - t) ** 2)))
    mean_t = np.mean(t)
    rel_pct = (rmse / mean_t * 100.0) if mean_t > 0 else float('nan')
    return r2, rel_pct


def make_parity_figure(rec, title, out_stem):
    fig, axes = plt.subplots(2, 2, figsize=(FW, FH))
    fig.patch.set_facecolor("white")

    for ax, lab, (pkey, psym) in zip(
            axes.flat, ("(a)", "(b)", "(c)", "(d)"), PANELS):
        t_all = rec[f"{pkey}_true"]
        r_all = rec[f"{pkey}_rec"]
        m = (np.isfinite(t_all) & np.isfinite(r_all)
             & (t_all > 0) & (r_all > 0))
        r2, rel_pct = stats(t_all[m], r_all[m])

        for arch in ARCHS:
            ma = m & (rec["arch"] == arch)
            if not ma.any():
                continue
            t, r = t_all[ma], r_all[ma]
            s = np.nan_to_num(rec[f"{pkey}_sig"][ma], nan=0.0)
            ax.errorbar(t, r, yerr=s,
                        fmt=MARKERS[arch], color=COLORS[arch],
                        ms=2.0, lw=0.0, elinewidth=0.5,
                        capsize=1.0, capthick=0.5, alpha=0.80,
                        zorder=3, label=ARCH_LABELS[arch])

        vals = np.concatenate([t_all[m], r_all[m]])
        lo, hi = vals.min() * 0.75, vals.max() * 1.30
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.85, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.minorticks_off()
        ax.set_xlabel(f"True {psym}", labelpad=2)
        ax.set_ylabel(f"Recovered {psym}", labelpad=2)
        ax.grid(True, which="major", lw=0.3, color="#cccccc", alpha=0.6)

        ax.text(0.02, 0.98,
                f"$R^2={r2:.4f}$\nRMSE={rel_pct:.1f}%",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8.0, zorder=1,
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec="#cccccc", lw=0.5, alpha=0.95))

        ax.text(0.97, 0.03, lab, transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=9.0, fontweight="bold", zorder=4)

    handles = [Line2D([0], [0], marker=MARKERS[a], color=COLORS[a], lw=0,
                      ms=3.0, markeredgecolor="white", markeredgewidth=0.3,
                      label=ARCH_LABELS[a]) for a in ARCHS]
    fig.legend(handles, [ARCH_LABELS[a] for a in ARCHS],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.3, columnspacing=1.1, handletextpad=0.4)

    fig.text(0.5, 0.995, title, ha="center", va="top", fontsize=10.0)

    fig.subplots_adjust(left=0.125, right=0.99, top=0.885, bottom=0.185,
                        wspace=0.40, hspace=0.42)

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"{out_stem}.{ext}"
        fig.savefig(out, dpi=600, facecolor="white")
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    ideal    = load(DATA_DIR / "parity_results_combined.csv")
    mismatch = load(DATA_DIR / "mismatch_results_combined.csv")
    make_parity_figure(ideal,    FIG2_TITLE, "fig2_parity_ideal")
    make_parity_figure(mismatch, FIG3_TITLE, "fig3_parity_mismatch")
    print("Done.")


if __name__ == "__main__":
    main()