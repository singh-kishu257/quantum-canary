from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "hardware"
OUT_DIR = ROOT / "figures"

QUBIT_ORDER: list = []

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

def fwd_t1(tau, T1, p0g1=0.0, p1g0=0.0):
    p = np.exp(-np.asarray(tau, float) / T1)
    return p * (1 - p0g1) + (1 - p) * p1g0

def fwd_echo(tau, T2e, p0g1=0.0, p1g0=0.0):
    p = 0.5 * (1 - np.exp(-np.asarray(tau, float) / T2e))
    return p * (1 - p0g1) + (1 - p) * p1g0

def fwd_gate_dose(x, p0g1=0.0, p1g0=0.0):
    p = 0.5 * (1 + np.exp(-4.0 * np.asarray(x, float)))
    return p * (1 - p0g1) + (1 - p) * p1g0

def find_run(label):
    fs = sorted(DATA_DIR.glob(f"run_{label}_*.json"), key=lambda p: p.stat().st_mtime)
    if not fs:
        raise FileNotFoundError(f"No run_{label}_*.json in {DATA_DIR}")
    return fs[-1]

def load(p):
    with p.open() as f:
        return json.load(f)

def recs(q, probe):
    return [r for r in q["model_fit"]["records"] if r["probe"] == probe]

def ra(rs, key):
    return np.asarray([r[key] for r in rs], float)

def qcolor(qid):
    return plt.cm.tab10.colors[QUBIT_ORDER.index(qid) % 10]

def letter(ax, lab):
    ax.text(0.97, 0.03, lab, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9.0, fontweight="bold", zorder=4)

def panel_t1_all(ax, run):
    x_ref = np.linspace(0, 3.8, 400)
    ax.plot(x_ref, np.exp(-x_ref), color="black", lw=1.0, ls="--", alpha=0.4, zorder=1)
    for q in run["qubits"]:
        r = q["result_full"]
        cal = q["calibration"]
        rs = recs(q, "T1")
        tau = ra(rs, "condition")
        xn = tau / r["T1_s"]
        o = np.argsort(xn)
        xq = np.linspace(0, max(xn) * 1.15, 200)
        ax.plot(xq, fwd_t1(xq * r["T1_s"], r["T1_s"], cal["p0_given_1"], cal["p1_given_0"]),
                color=qcolor(q["qubit_id"]), lw=0.9, alpha=0.5, zorder=2)
        ax.errorbar(xn[o], ra(rs, "p_meas")[o], yerr=ra(rs, "sigma")[o],
                    fmt="o", ms=2.2, color=qcolor(q["qubit_id"]),
                    capsize=1.0, lw=0.6, elinewidth=0.6, zorder=3)
    ax.set_xlim(0, 3.8)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel(r"Normalized delay $\tau/\hat{T}_1$")
    ax.set_ylabel(r"$P(|1\rangle)$")
    ax.set_title(r"$T_1$ decay", fontsize=10.0, pad=4)
    ax.grid(True, which="major", lw=0.3, color="#cccccc", alpha=0.6)
    letter(ax, "(a)")

def panel_ramsey_all(ax, run):
    x_ref = np.linspace(0, 3.8, 400)
    ax.plot(x_ref, np.exp(-x_ref), color="black", lw=1.0, ls="--", alpha=0.4, zorder=1)
    for q in run["qubits"]:
        r = q["result_full"]
        rx = recs(q, "Ramsey-X")
        ry = recs(q, "Ramsey-Y")
        X = 1 - 2 * ra(rx, "p_meas")
        Y = 1 - 2 * ra(ry, "p_meas")
        R = np.sqrt(X**2 + Y**2)
        sR = np.sqrt(ra(rx, "sigma")**2 + ra(ry, "sigma")**2)
        xn = ra(rx, "condition") / r["T2_ramsey_s"]
        o = np.argsort(xn)
        ax.errorbar(xn[o], R[o], yerr=sR[o], fmt="s", ms=2.2,
                    color=qcolor(q["qubit_id"]), capsize=1.0, lw=0.6,
                    elinewidth=0.6, zorder=3)
    ax.set_xlim(0, 3.8)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel(r"Normalized delay $\tau/\hat{T}_2^{\rm Ram}$")
    ax.set_ylabel(r"Bloch amplitude $|R|$")
    ax.set_title(r"Ramsey $|R|$", fontsize=10.0, pad=4)
    ax.grid(True, which="major", lw=0.3, color="#cccccc", alpha=0.6)
    letter(ax, "(b)")

def panel_echo_all(ax, run):
    x_ref = np.linspace(0, 3.8, 400)
    ax.plot(x_ref, 0.5 * (1 - np.exp(-x_ref)), color="black", lw=1.0, ls="--", alpha=0.4, zorder=1)
    for q in run["qubits"]:
        r = q["result_full"]
        cal = q["calibration"]
        rs = recs(q, "Echo")
        tau = ra(rs, "condition")
        xn = tau / r["T2_echo_s"]
        o = np.argsort(xn)
        xq = np.linspace(0, max(xn) * 1.15, 200)
        ax.plot(xq, fwd_echo(xq * r["T2_echo_s"], r["T2_echo_s"], cal["p0_given_1"], cal["p1_given_0"]),
                color=qcolor(q["qubit_id"]), lw=0.9, alpha=0.5, zorder=2)
        ax.errorbar(xn[o], ra(rs, "p_meas")[o], yerr=ra(rs, "sigma")[o],
                    fmt="v", ms=2.2, color=qcolor(q["qubit_id"]),
                    capsize=1.0, lw=0.6, elinewidth=0.6, zorder=3)
    ax.set_xlim(0, 3.8)
    ax.set_ylim(-0.03, 0.62)
    ax.set_xlabel(r"Normalized delay $\tau/\hat{T}_2^{\rm echo}$")
    ax.set_ylabel(r"$P(|1\rangle)$")
    ax.set_title(r"Hahn echo", fontsize=10.0, pad=4)
    ax.grid(True, which="major", lw=0.3, color="#cccccc", alpha=0.6)
    letter(ax, "(c)")

def panel_gate_all(ax, run):
    xmax = 0.0
    for q in run["qubits"]:
        rs = recs(q, "Gate")
        xmax = max(xmax, np.max(ra(rs, "condition")) * q["result_full"]["epsilon_sx"])
    x_ref = np.linspace(0, xmax * 1.15, 400)
    ax.plot(x_ref, 0.5 * (1 + np.exp(-4.0 * x_ref)), color="black", lw=1.0,
            ls="--", alpha=0.4, zorder=1)
    for q in run["qubits"]:
        r = q["result_full"]
        cal = q["calibration"]
        rs = recs(q, "Gate")
        N = ra(rs, "condition")
        x = N * r["epsilon_sx"]
        o = np.argsort(x)
        xd = np.linspace(0, max(x) * 1.15, 200)
        ax.plot(xd, fwd_gate_dose(xd, cal["p0_given_1"], cal["p1_given_0"]),
                color=qcolor(q["qubit_id"]), lw=0.9, alpha=0.5, zorder=2)
        ax.errorbar(x[o], ra(rs, "p_meas")[o], yerr=ra(rs, "sigma")[o],
                    fmt="D", ms=2.2, color=qcolor(q["qubit_id"]),
                    capsize=1.0, lw=0.6, elinewidth=0.6, zorder=3)
    ax.axhline(0.5, color="#aaaaaa", lw=0.6, ls="--")
    ax.set_xlim(0, xmax * 1.15)
    ax.set_ylim(0.45, 1.03)
    ax.set_xlabel(r"Normalized dose $N\cdot\hat{\varepsilon}_{sx}$")
    ax.set_ylabel(r"$P(|0\rangle)$")
    ax.set_title(r"Gate repetition", fontsize=10.0, pad=4)
    ax.grid(True, which="major", lw=0.3, color="#cccccc", alpha=0.6)
    letter(ax, "(d)")

def panel_residuals(ax, run):
    probes = ["T1", "Ramsey-X", "Ramsey-Y", "Echo", "Gate"]
    probe_labels = ["$T_1$", "Ramsey", "Echo", "Gate"]
    probe_map = {"T1": 0, "Ramsey-X": 1, "Ramsey-Y": 1, "Echo": 2, "Gate": 3}

    zs = {0: [], 1: [], 2: [], 3: []}
    for q in run["qubits"]:
        for p in probes:
            rs = recs(q, p)
            zs[probe_map[p]].extend(ra(rs, "z"))

    rng = np.random.default_rng(42)
    for i in range(4):
        y = zs[i]
        x = rng.uniform(-0.2, 0.2, size=len(y)) + i
        ax.scatter(x, y, s=8, alpha=0.6, color="#1f77b4", zorder=3)

    ax.axhline(2, color="#888888", lw=0.6, ls="--", zorder=2)
    ax.axhline(-2, color="#888888", lw=0.6, ls="--", zorder=2)
    ax.axhline(0, color="#cccccc", lw=0.6, ls="-", zorder=1)

    ax.set_xticks(range(4))
    ax.set_xticklabels(probe_labels, rotation=30, ha="right")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-8, 8)
    ax.set_ylabel(r"Residual $z$-score")
    ax.set_title(r"Residual $z$-scores", fontsize=10.0, pad=4)
    ax.grid(True, which="major", lw=0.3, color="#cccccc", alpha=0.6)
    letter(ax, "(e)")

def panel_t2_consistency(ax, run):
    t2r = np.array([q["result_full"]["T2_ramsey_s"] * 1e6 for q in run["qubits"]])
    t2e = np.array([q["result_full"]["T2_echo_s"] * 1e6 for q in run["qubits"]])
    t2r_e = np.array([q["result_full"]["T2_ramsey_sigma_s"] * 1e6 for q in run["qubits"]])
    t2e_e = np.array([q["result_full"]["T2_echo_sigma_s"] * 1e6 for q in run["qubits"]])

    qids = [q["qubit_id"] for q in run["qubits"]]

    lim_min = 0
    lim_max = max(np.max(t2r), np.max(t2e)) * 1.1

    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=0.8, alpha=0.85, zorder=2)

    for i, qid in enumerate(qids):
        ax.errorbar(t2r[i], t2e[i], xerr=t2r_e[i], yerr=t2e_e[i],
                    fmt="o", ms=3.5, color=qcolor(qid),
                    capsize=1.5, lw=0.6, elinewidth=0.6, zorder=3)

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel(r"$T_2^{\rm Ram}$ (µs)")
    ax.set_ylabel(r"$T_2^{\rm echo}$ (µs)")
    ax.set_title(r"$T_2$ self-consistency", fontsize=10.0, pad=4)
    ax.grid(True, which="major", lw=0.3, color="#cccccc", alpha=0.6)
    letter(ax, "(f)")

def panel_admissibility(ax, run):
    qids = [q["qubit_id"] for q in run["qubits"]]
    ratios = [q["admissibility"]["T2_over_2T1_ratio"] for q in run["qubits"]]

    x = np.arange(len(qids))
    ax.scatter(x, ratios, s=25, color="#1f77b4", zorder=3)
    ax.axhline(1.0, color="#C00000", lw=0.8, ls="--", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{q}" for q in qids], rotation=45, ha="right")
    ax.set_xlim(-0.5, len(qids) - 0.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(r"$T_2 / (2T_1)$")
    ax.set_title(r"Admissibility $T_2 \leq 2T_1$", fontsize=10.0, pad=4)
    ax.grid(True, which="major", lw=0.3, color="#cccccc", alpha=0.6, axis="y")
    letter(ax, "(g)")

def panel_chi2_heatmap(ax, run):
    qids = [q["qubit_id"] for q in run["qubits"]]
    probes = [r"$T_1$", "Ramsey", "Echo", "Gate"]
    keys = ["t1_chi2_dof", "ramsey_chi2_dof", "echo_chi2_dof", "gate_chi2_dof"]
    raw = np.array([[q["result_full"][k] for k in keys] for q in run["qubits"]])

    order = np.argsort(np.mean(raw[:, :3], axis=1))
    raw = raw[order]
    labs = [f"Q{qids[i]}" for i in order]

    cmap = LinearSegmentedColormap.from_list(
        "chi2", ["#f7fbff", "#c6dbef", "#fdae6b", "#d62728"], N=256)
    disp = np.clip(raw, 0, 15)
    im = ax.imshow(disp, aspect="auto", cmap=cmap, vmin=0, vmax=15, origin="upper")

    for i in range(len(qids)):
        for j in range(4):
            v = raw[i, j]
            txt = f"{v:.1f}" if v < 10 else f"{v:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.0,
                    color="white" if disp[i, j] > 10 else "black",
                    fontweight="bold" if v > 7.0 else "normal")

    ax.set_xticks(range(4))
    ax.set_xticklabels(probes, rotation=30, ha="right")
    ax.set_yticks(range(len(qids)))
    ax.set_yticklabels(labs)
    ax.set_title(r"Run A $\chi^2/\nu$", fontsize=10.0, pad=4)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\chi^2/\nu$", fontsize=8.0)
    cb.ax.tick_params(labelsize=7.5)
    cb.set_ticks([0, 5, 10, 15])

    ax.text(1.02, -0.02, "(h)", transform=ax.transAxes, ha="left",
            va="top", fontsize=9.0, fontweight="bold", zorder=4)

def make_figure(A, B, out_stem):
    global QUBIT_ORDER
    QUBIT_ORDER = sorted([q["qubit_id"] for q in A["qubits"]])

    fig = plt.figure(figsize=(7.16, 4.2))
    gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1.1, 1.0], hspace=0.55, wspace=0.45)
    fig.subplots_adjust(left=0.06, right=0.95, top=0.90, bottom=0.16)

    panel_t1_all(fig.add_subplot(gs[0, 0]), A)
    panel_ramsey_all(fig.add_subplot(gs[0, 1]), A)
    panel_echo_all(fig.add_subplot(gs[0, 2]), A)
    panel_gate_all(fig.add_subplot(gs[0, 3]), A)

    panel_residuals(fig.add_subplot(gs[1, 0]), A)
    panel_t2_consistency(fig.add_subplot(gs[1, 1]), A)
    panel_admissibility(fig.add_subplot(gs[1, 2]), A)
    panel_chi2_heatmap(fig.add_subplot(gs[1, 3]), A)

    handles = [Line2D([0], [0], marker="o", color=qcolor(q), lw=0, ms=3.0,
                      markeredgecolor="white", markeredgewidth=0.3,
                      label=f"Q{q}") for q in QUBIT_ORDER]
    fig.legend(handles, [f"Q{q}" for q in QUBIT_ORDER], loc="lower center",
               ncol=8, bbox_to_anchor=(0.5, 0.0), handlelength=1.3,
               columnspacing=1.0, handletextpad=0.4, fontsize=8.0)

    fig.text(0.5, 0.995, "Quantum Canary Hardware Validation on IBM Heron r2 Processor",
             ha="center", va="top", fontsize=11.0, fontweight="semibold")

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
    ap.add_argument("--output", default="fig6_hardware_ibm")
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