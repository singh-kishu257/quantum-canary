import json
import pathlib

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "hardware" / "ionq_canary_allq_0-19_20260810T213159.json"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FW, FH = 7.16, 3.47

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

C_NORMAL = "#1F3864"
C_SPEC = "#3A4A5A"

with open(DATA_PATH) as f:
    d = json.load(f)

N_Q = len(d["qubits"])
published = d["published_device_eps_1q"]
qubits = np.arange(N_Q)
eps = np.array([d["canary"][str(q)]["eps_sx"] for q in qubits])
sigma = np.array([d["canary"][str(q)]["eps_sx_sigma"] for q in qubits])
ratio = eps / published

fig, axes = plt.subplots(1, 2, figsize=(FW, FH),
                         gridspec_kw={"width_ratios": [3, 1], "wspace": 0.08})
fig.patch.set_facecolor("white")

ax = axes[0]
ax.axhspan(published / 2, published * 2, color=C_NORMAL, alpha=0.10, zorder=1)
ax.axhline(published, color=C_SPEC, lw=1.0, ls="--", zorder=3)
ax.errorbar(qubits, eps, yerr=sigma, fmt="o", color=C_NORMAL, ms=4.0,
            lw=0.0, elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=4)
ax.set_yscale("log")
ax.set_xlim(-0.8, N_Q - 0.2)
ax.set_ylim(3e-5, 8e-4)
ax.set_xticks(qubits)
ax.set_xticklabels([str(q) for q in qubits])
ax.set_yticks([1e-4, 2e-4, 4e-4])
ax.set_yticklabels([r"$1\times10^{-4}$", r"$2\times10^{-4}$",
                    r"$4\times10^{-4}$"])
ax.minorticks_off()
ax.set_xlabel("Qubit index", labelpad=2)
ax.set_title(r"Quantum Canary $\varepsilon_{sx}$, all 20 qubits",
             fontsize=10.0, pad=4)
ax.grid(True, which="major", lw=0.3, color="#cccccc", alpha=0.6)

mean_v = float(np.mean(eps))
median_v = float(np.median(eps))
ax.text(0.02, 0.03, f"mean $\\varepsilon_{{sx}}={mean_v:.2e}$\n"
                    f"median $\\varepsilon_{{sx}}={median_v:.2e}$\n"
                    f"$N={N_Q}$ qubits",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0,
        zorder=5, bbox=dict(boxstyle="round,pad=0.18", fc="white",
                            ec="#cccccc", lw=0.5, alpha=0.95))

handles = [
    Line2D([0], [0], color=C_SPEC, lw=1.0, ls="--",
           label=r"IonQ published $\varepsilon_{1q}=2.0\times10^{-4}$"),
    Patch(facecolor=C_NORMAL, alpha=0.10, edgecolor="none",
          label=r"$\times2$ band around spec"),
    Line2D([0], [0], marker="o", color=C_NORMAL, lw=0, ms=4.0,
           label=r"Quantum Canary $\varepsilon_{sx}$ (20 qubits)"),
]
ax.legend(handles, [h.get_label() for h in handles], loc="upper right",
          bbox_to_anchor=(1.015, 1.015), handlelength=1.4,
          handletextpad=0.5)

ax.text(0.97, 0.03, "(a)", transform=ax.transAxes, ha="right",
        va="bottom", fontsize=9.0, fontweight="bold", zorder=6)

ax2 = axes[1]
bins = np.logspace(np.log10(0.3), np.log10(3.0), 12)
counts, edges = np.histogram(ratio, bins=bins)
for lo, hi, c in zip(edges[:-1], edges[1:], counts):
    ax2.barh(0.5 * (lo + hi), c, height=(hi - lo) * 0.85, color=C_NORMAL,
             edgecolor="white", lw=0.4, zorder=3)
ax2.axhline(1.0, color=C_SPEC, lw=1.0, ls="--", zorder=4)
ax2.axhspan(0.5, 2.0, color=C_NORMAL, alpha=0.10, zorder=1)
ax2.set_yscale("log")
ax2.set_ylim(3e-5 / published, 8e-4 / published)
ax2.set_yticks([0.25, 0.5, 1, 2, 4])
ax2.set_yticklabels(["0.25×", "0.5×", "1×", "2×", "4×"])
ax2.minorticks_off()
ax2.set_xticks([0, 5, 10])
ax2.set_xlabel("Count", labelpad=2)
ax2.set_ylabel(r"$\varepsilon_{sx}/\varepsilon_{\rm spec}$", labelpad=0)
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.set_title("Ratio distribution", fontsize=10.0, pad=4)
ax2.grid(True, which="major", axis="x", lw=0.3, color="#cccccc", alpha=0.6)
ax2.text(0.97, 0.03, "(b)", transform=ax2.transAxes, ha="right",
         va="bottom", fontsize=9.0, fontweight="bold", zorder=6)

fig.subplots_adjust(left=0.07, right=0.93, top=0.914, bottom=0.145)

for ext in ("pdf", "png"):
    out = OUT_DIR / f"fig8_hardware_ionq.{ext}"
    fig.savefig(out, dpi=600, facecolor="white")
    print(f"Saved: {out}")
plt.close(fig)