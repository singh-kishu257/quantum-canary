import csv
import pathlib

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

ROOT     = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "qe_benchmark.csv"
if not DATA_PATH.exists():
    DATA_PATH = DATA_DIR / "qe-benchmark.csv"
if not DATA_PATH.exists():
    raise SystemExit(f"ERROR: missing input file: {DATA_DIR / 'qe_benchmark.csv'}")

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
PARAMS       = ["T1", "T2", "delta_omega", "epsilon_sx"]
PARAM_LABELS = {"T1": r"$T_1$", "T2": r"$T_2$",
                "delta_omega": r"$|\Delta\omega|$",
                "epsilon_sx": r"$\varepsilon_{sx}$"}
METHODS = ["Canary", "QiskitExperiments"]
COLOUR  = {"Canary": "#1F3864", "QiskitExperiments": "#C00000"}
MARKER  = {"Canary": "o", "QiskitExperiments": "s"}
LINESTY = {"Canary": "-", "QiskitExperiments": "--"}
LABEL   = {"Canary": "Quantum Canary (this work)",
           "QiskitExperiments": r"Qiskit Experiments ($T_1$+$T_2$R+$T_2$H+RB)"}
NATIVE  = 9900
YMIN, YMAX = -0.15, 1.05

data = {a: {p: {m: {} for m in METHODS} for p in PARAMS} for a in ARCHS}
with open(DATA_PATH, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        a, p, m = row["architecture"], row["parameter"], row["method"]
        if a in data and p in data[a] and m in data[a][p]:
            data[a][p][m][int(row["budget"])] = (
                float(row["r2"]), float(row["r2_lo"]), float(row["r2_hi"]))

FW, FH = 7.16, 5.9
fig, axes = plt.subplots(3, 4, figsize=(FW, FH), sharex=True, sharey=True)
fig.patch.set_facecolor("white")

for ri, arch in enumerate(ARCHS):
    for ci, p in enumerate(PARAMS):
        ax = axes[ri, ci]
        for m in METHODS:
            d = data[arch][p][m]
            budgets = sorted(d)
            ys  = [np.clip(d[b][0], YMIN, YMAX) if np.isfinite(d[b][0]) else np.nan
                   for b in budgets]
            los = [np.clip(d[b][1], YMIN, YMAX) if np.isfinite(d[b][1]) else np.nan
                   for b in budgets]
            his = [np.clip(d[b][2], YMIN, YMAX) if np.isfinite(d[b][2]) else np.nan
                   for b in budgets]
            ax.fill_between(budgets, los, his, color=COLOUR[m], alpha=0.12, zorder=1)
            ax.plot(budgets, ys, color=COLOUR[m], lw=1.1, ls=LINESTY[m],
                    marker=MARKER[m], ms=2.6, mfc=COLOUR[m], mec=COLOUR[m],
                    zorder=3, clip_on=True)
        ax.axhline(0.95, color="#888888", lw=0.6, ls="--", zorder=2)
        ax.axvline(NATIVE, color="#3A4A5A", lw=0.7, ls=":", zorder=2, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlim(750, 67500)
        ax.set_ylim(YMIN, YMAX)
        ax.set_xticks([1000, 10000, 50000])
        ax.set_xticklabels(["1k", "10k", "50k"])
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
        ax.grid(True, which="major", lw=0.3, color="#cccccc", alpha=0.6)
        ax.text(0.97, 0.03, f"({chr(ord('a') + ri * 4 + ci)})",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9.0, fontweight="bold")
        if ri == 0:
            ax.set_title(PARAM_LABELS[p], fontsize=10.0, pad=4)
        if ci == 0:
            ax.set_ylabel(f"{ARCH_LABELS[arch]}\n$R^2$", fontsize=9.0)
        if ri == 2:
            ax.set_xlabel("Total shot budget", fontsize=9.0, labelpad=2)

handles = [Line2D([0], [0], color=COLOUR[m], lw=1.1, ls=LINESTY[m],
                  marker=MARKER[m], ms=3.0, mfc=COLOUR[m], mec=COLOUR[m],
                  label=LABEL[m]) for m in METHODS]
fig.legend(handles, [LABEL[m] for m in METHODS], loc="lower center",
           ncol=2, bbox_to_anchor=(0.5, 0.005), handlelength=2.2,
           columnspacing=1.5, handletextpad=0.6)

fig.text(0.5, 0.995,
         r"Quantum Canary vs. Qiskit Experiments — Recovery $R^2$ vs. Total Shot Budget",
         ha="center", va="top", fontsize=11.0, fontweight="semibold")

fig.subplots_adjust(left=0.075, right=0.99, top=0.94, bottom=0.10,
                    wspace=0.12, hspace=0.28)

for ext in ("pdf", "png"):
    out = OUT_DIR / f"fig5_qe_benchmark.{ext}"
    fig.savefig(out, dpi=600, facecolor="white")
    print(f"Saved: {out}")
plt.close(fig)