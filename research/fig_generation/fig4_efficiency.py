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
COLOUR = {"T1": "#3A6B9E", "T2": "#4A8A60",
          "delta_omega": "#7050A0", "epsilon_sx": "#A05030"}
MARKER = {"T1": "o", "T2": "s", "delta_omega": "^", "epsilon_sx": "D"}
NATIVE  = 9900
YMIN, YMAX = -0.15, 1.05

data = {a: {p: {} for p in PARAMS} for a in ARCHS}
with open(DATA_PATH, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["method"] != "Canary":
            continue
        a, p = row["architecture"], row["parameter"]
        if a in data and p in data[a]:
            data[a][p][int(row["budget"])] = (
                float(row["r2"]), float(row["r2_lo"]), float(row["r2_hi"]))

FW, FH = 3.6, 3.5
fig, axes = plt.subplots(3, 1, figsize=(FW, FH), sharex=True, sharey=True)
fig.patch.set_facecolor("white")

for ri, arch in enumerate(ARCHS):
    ax = axes[ri]
    for p in PARAMS:
        d = data[arch][p]
        budgets = sorted(d)
        ys  = [np.clip(d[b][0], YMIN, YMAX) if np.isfinite(d[b][0]) else np.nan
               for b in budgets]
        los = [np.clip(d[b][1], YMIN, YMAX) if np.isfinite(d[b][1]) else np.nan
               for b in budgets]
        his = [np.clip(d[b][2], YMIN, YMAX) if np.isfinite(d[b][2]) else np.nan
               for b in budgets]
        ax.fill_between(budgets, los, his, color=COLOUR[p], alpha=0.13, zorder=1)
        ax.plot(budgets, ys, color=COLOUR[p], lw=1.3, ls="-",
                marker=MARKER[p], ms=5.0, mfc=COLOUR[p], mec=COLOUR[p],
                zorder=3, clip_on=True)
    ax.axhline(0.95, color="#888888", lw=0.6, ls="--", zorder=2)
    ax.axvline(NATIVE, color="#3A4A5A", lw=0.7, ls=":", zorder=2, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlim(750, 130000)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xticks([1000, 10000, 100000])
    ax.set_xticklabels(["1k", "10k", "100k"])
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.grid(True, which="major", lw=0.3, color="#cccccc", alpha=0.6)
    ax.text(0.97, 0.05, ARCH_LABELS[arch],
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, fontweight="bold")
    if ri == 1:
        ax.set_ylabel(r"$R^2$", fontsize=9.5)
    if ri == 2:
        ax.set_xlabel("Total shot budget", fontsize=9.0, labelpad=1)

handles = [Line2D([0], [0], color=COLOUR[p], lw=1.3, ls="-",
                  marker=MARKER[p], ms=5.5, mfc=COLOUR[p], mec=COLOUR[p],
                  label=PARAM_LABELS[p]) for p in PARAMS]
fig.legend(handles, [PARAM_LABELS[p] for p in PARAMS], loc="lower center",
           ncol=4, bbox_to_anchor=(0.5, 0.005), handlelength=1.8,
           columnspacing=1.2, handletextpad=0.5)

fig.subplots_adjust(left=0.115, right=0.97, top=0.926, bottom=0.155,
                    hspace=0.07)

for ext in ("pdf", "png"):
    out = OUT_DIR / f"fig4_efficiency.{ext}"
    fig.savefig(out, dpi=600, facecolor="white")
    print(f"Saved: {out}")
plt.close(fig)
