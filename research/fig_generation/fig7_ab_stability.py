from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "hardware"
OUT_DIR = ROOT / "figures"

QUBIT_ORDER: list = []

FW, FH = 3.5, 5.4

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

PANELS = [
    ("T1",  r"$T_1$ (µs)"),
    ("T2",  r"$T_2$ (µs)"),
    ("dw",  r"$|\Delta\omega|$ (rad/s)"),
    ("eps", r"$\varepsilon_{sx}$"),
]

def find_run(label):
    fs = sorted(DATA_DIR.glob(f"run_{label}_*.json"),
                key=lambda p: p.stat().st_mtime)
    if not fs:
        raise FileNotFoundError(f"No run_{label}_*.json in {DATA_DIR}")
    return fs[-1]

def load(p):
    with p.open() as f:
        return json.load(f)

def qcolor(qid):
    return plt.cm.tab10.colors[QUBIT_ORDER.index(qid) % 10]

def letter(ax, lab):
    ax.text(0.97, 0.03, lab, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9.0, fontweight="bold", zorder=4)

def statbox(ax, txt):
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.0, zorder=5,
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec="#cccccc", lw=0.5, alpha=0.95))

def val_sig(q, key):
    r = q["result_full"]
    if key == "T1":
        return r["T1_s"] * 1e6, r["T1_sigma_s"] * 1e6
    if key == "T2":
        return r["T2_s"] * 1e6, r["T2_sigma_s"] * 1e6
    if key == "dw":
        return abs(r["delta_omega"]), r["delta_omega_sigma"]
    return r["epsilon_sx"], r["epsilon_sx_sigma"]

def panel_parity(ax, A, B, key, psym, lab):
    a_by = {q["qubit_id"]: q for q in A["qubits"]}
    b_by = {q["qubit_id"]: q for q in B["qubits"]}

    vals = []
    zvals = []
    for qid in QUBIT_ORDER:
        xa, sa = val_sig(a_by[qid], key)
        xb, sb = val_sig(b_by[qid], key)

        vals += [xa, xb]
        denom = np.sqrt(sa**2 + sb**2)
        zvals.append(abs((xa - xb) / denom) if denom > 0 else np.nan)

        xlo = min(sa, 0.9 * xa) if xa > 0 else sa
        ylo = min(sb, 0.9 * xb) if xb > 0 else sb

        ax.errorbar(xa, xb, xerr=[[xlo], [sa]], yerr=[[ylo], [sb]],
                    fmt="o", color=qcolor(qid), ms=4.0, lw=0.0,
                    elinewidth=0.6, capsize=1.2, capthick=0.6,
                    alpha=0.9, zorder=3)

    lo, hi = min(vals) * 0.75, max(vals) * 1.30
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.85, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.minorticks_off()
    ax.set_xlabel(f"Run A {psym}", labelpad=2)
    ax.set_ylabel(f"Run B {psym}", labelpad=2)
    ax.grid(True, which="major", lw=0.3, color="#cccccc", alpha=0.6)

    n_ok = int(np.nansum(np.asarray(zvals) <= 2.0))
    statbox(ax, f"{n_ok}/8 within $2\sigma$")

    if key == "eps" and 15 in a_by and 15 in b_by:
        xa, _ = val_sig(a_by[15], key)
        xb, _ = val_sig(b_by[15], key)
        ax.annotate("Q15", (xa, xb), textcoords="offset points",
                    xytext=(5, -4), fontsize=7.5, fontweight="bold",
                    color="#333333")

    letter(ax, lab)

def panel_z_heatmap(ax, A, B):
    qids = sorted([q["qubit_id"] for q in A["qubits"]])
    a_by = {q["qubit_id"]: q for q in A["qubits"]}
    b_by = {q["qubit_id"]: q for q in B["qubits"]}

    params = [
        ("T1_s", "T1_sigma_s", r"$T_1$"),
        ("T2_s", "T2_sigma_s", r"$T_2$"),
        ("delta_omega", "delta_omega_sigma", r"$|\Delta\omega|$"),
        ("epsilon_sx", "epsilon_sx_sigma", r"$\varepsilon_{sx}$")
    ]

    z_matrix = np.zeros((len(qids), 4))
    for i, qid in enumerate(qids):
        for j, (vk, sk, _) in enumerate(params):
            va = a_by[qid]["result_full"][vk]
            sa = a_by[qid]["result_full"][sk]
            vb = b_by[qid]["result_full"][vk]
            sb = b_by[qid]["result_full"][sk]
            if vk == "delta_omega":
                va, vb = abs(va), abs(vb)
            denom = np.sqrt(sa**2 + sb**2)
            z_matrix[i, j] = (va - vb) / denom if denom > 0 else 0.0

    im = ax.imshow(z_matrix, aspect="auto", cmap=plt.cm.RdBu_r,
                   vmin=-10, vmax=10, origin="upper")

    for i in range(len(qids)):
        for j in range(4):
            v = z_matrix[i, j]
            txt = f"{v:.1f}" if abs(v) < 10 else f"{v:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.0,
                    color="white" if abs(v) > 6 else "black",
                    fontweight="bold" if abs(v) > 2.0 else "normal")

    ax.set_xticks(range(4))
    ax.set_xticklabels([p[2] for p in params])
    ax.set_yticks(range(len(qids)))
    ax.set_yticklabels([f"Q{q}" for q in qids])
    ax.set_title("Standardized between-session differences",
                 fontsize=9.0, pad=4)

    cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.015)
    cb.ax.set_title(r"$z$", fontsize=8.0)
    cb.ax.tick_params(labelsize=7.5)
    cb.set_ticks([-10, -5, 0, 5, 10])

    ax.text(0.995, 0.02, "(e)", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9.0, fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                      alpha=0.9))

def make_figure(A, B, out_stem):
    global QUBIT_ORDER
    QUBIT_ORDER = sorted([q["qubit_id"] for q in A["qubits"]])

    fig = plt.figure(figsize=(FW, FH))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.0, 1.15],
                           hspace=0.60, wspace=0.38)

    panel_parity(fig.add_subplot(gs[0, 0]), A, B, *PANELS[0], "(a)")
    panel_parity(fig.add_subplot(gs[0, 1]), A, B, *PANELS[1], "(b)")
    panel_parity(fig.add_subplot(gs[1, 0]), A, B, *PANELS[2], "(c)")
    panel_parity(fig.add_subplot(gs[1, 1]), A, B, *PANELS[3], "(d)")
    panel_z_heatmap(fig.add_subplot(gs[2, :]), A, B)

    handles = [Line2D([0], [0], marker="o", color=qcolor(q), lw=0, ms=3.5,
                      markeredgecolor="white", markeredgewidth=0.3,
                      label=f"Q{q}") for q in QUBIT_ORDER]
    fig.legend(handles, [f"Q{q}" for q in QUBIT_ORDER], loc="lower center",
               ncol=8, bbox_to_anchor=(0.5, 0.006), handlelength=1.0,
               columnspacing=0.8, handletextpad=0.3)

    fig.text(0.5, 0.995, "Temporal Stability of Repeated\n"
                         "Quantum Canary Characterization on ibm_fez",
             ha="center", va="top", fontsize=10.0, fontweight="semibold")

    fig.subplots_adjust(left=0.12, right=0.90, top=0.93, bottom=0.085)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{out_stem}.{ext}"
        fig.savefig(path, dpi=600, facecolor="white")
        print(f"Saved: {path}")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", default=None)
    ap.add_argument("--run-b", default=None)
    ap.add_argument("--output", default="fig7_ab_stability")
    args = ap.parse_args()

    pa, pb = find_run("A"), find_run("B")
    if args.run_a:
        pa = Path(args.run_a)
    if args.run_b:
        pb = Path(args.run_b)

    A, B = load(pa), load(pb)
    print(f"Run A : {pa.name}")
    print(f"Run B : {pb.name}")
    make_figure(A, B, args.output)

if __name__ == "__main__":
    main()