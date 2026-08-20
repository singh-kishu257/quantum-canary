from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
from scipy.stats import chi2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PROBES = ["t1", "ramsey", "echo", "gate"]
COLORS = {"t1": "#1f77b4", "ramsey": "#ff7f0e", "echo": "#2ca02c", "gate": "#d62728"}
LABELS = {"t1": r"$T_1$", "ramsey": "Ramsey", "echo": "Echo", "gate": "Gate"}
THRESHOLD = 3.0

def load_null(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def load_grad():
    path = DATA_DIR / "gradsummary_ALL.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "probe" not in df.columns:
        df = pd.read_csv(path, header=None,
                         names=["arch", "noise", "amp", "probe", "mean", "lo", "hi", "n"])
    return df

def panel_null(ax, null_arch, null_live):
    x = np.linspace(0, 6, 500)
    
    for d in [2, 5]:
        cdf_vals = chi2.cdf(x * d, df=d)
        ls = "--" if d == 2 else ":"
        lbl = f"Theoretical (dof={d})"
        ax.plot(x, cdf_vals, 'k', ls=ls, lw=1.0, alpha=0.6, label=lbl)

    for p in PROBES:
        if p in null_arch.columns:
            vals_arch = null_arch[p].dropna().values
            if len(vals_arch) > 0:
                xs = np.sort(vals_arch)
                ys = np.arange(1, len(xs) + 1) / len(xs)
                ax.plot(xs, ys, color=COLORS[p], lw=1.2, label=f"{LABELS[p]} (arch)")
            
        if not null_live.empty and p in null_live.columns:
            vals_live = null_live[p].dropna().values
            if len(vals_live) > 0:
                xs_l = np.sort(vals_live)
                ys_l = np.arange(1, len(xs_l) + 1) / len(xs_l)
                ax.plot(xs_l, ys_l, color=COLORS[p], lw=0.8, alpha=0.5, ls="-")

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$\chi^2/\nu$")
    ax.set_ylabel("Cumulative Probability")
    ax.axvline(THRESHOLD, color="#888888", lw=0.8, ls="--", alpha=0.8, zorder=2)
    ax.text(THRESHOLD + 0.1, 0.95, f"Threshold\n({THRESHOLD})", fontsize=7.0, color="#555555")
    ax.grid(True, which="major", lw=0.3, color="#cccccc", alpha=0.6)
    ax.legend(loc="lower right", fontsize=6.5)
    ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, fontsize=9.0, fontweight="bold", va="top")

def panel_grad(ax, grad_df, noise_class, panel_label):
    if grad_df.empty:
        ax.text(0.5, 0.5, "No Gradient Data", transform=ax.transAxes, ha="center", va="center")
        return

    arch_col = "arch" if "arch" in grad_df.columns else "architecture"
    if arch_col in grad_df.columns:
        sc_grad = grad_df[grad_df[arch_col] == "superconducting"]
    else:
        sc_grad = grad_df
        
    noise_col = "noise" if "noise" in sc_grad.columns else "noise_class"
    sub = sc_grad[sc_grad[noise_col] == noise_class]
    
    amp_col = "amp" if "amp" in sub.columns else "amplitude"
    sub = sub[sub[amp_col] > 0]

    mean_col = "mean" if "mean" in sub.columns else "mean_chi2_dof"
    lo_col = "lo" if "lo" in sub.columns else "ci_lo"
    hi_col = "hi" if "hi" in sub.columns else "ci_hi"

    for p in PROBES:
        p_sub = sub[sub["probe"] == p].sort_values(amp_col)
        if p_sub.empty: 
            continue
        
        amps = p_sub[amp_col].values
        means = p_sub[mean_col].values
        los = p_sub[lo_col].values
        his = p_sub[hi_col].values
        
        ax.plot(amps, means, color=COLORS[p], marker='o', ms=3, lw=1.2, label=LABELS[p], zorder=3)
        ax.fill_between(amps, los, his, color=COLORS[p], alpha=0.15, zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("Injection Amplitude")
    ax.set_ylabel(r"Mean $\chi^2/\nu$")
    ax.axhline(1.0, color="#888888", lw=0.8, ls="--", alpha=0.8, zorder=1)
    ax.axhline(THRESHOLD, color="#888888", lw=0.8, ls="--", alpha=0.8, zorder=1)
    ax.grid(True, which="major", lw=0.3, color="#cccccc", alpha=0.6, axis="y")
    ax.text(0.97, 0.97, f"({panel_label})", transform=ax.transAxes, fontsize=9.0, fontweight="bold", va="top", ha="right")

def make_figure():
    null_arch = load_null(DATA_DIR / "nullraw_ALL.csv")
    null_live = load_null(DATA_DIR / "nullraw_live_ALL.csv")
    grad_df = load_grad()

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.2))
    fig.patch.set_facecolor("white")
    
    panel_null(axes[0, 0], null_arch, null_live)
    panel_grad(axes[0, 1], grad_df, "t1_telegraph", "b")
    panel_grad(axes[1, 0], grad_df, "quasistatic_dephasing", "c")
    panel_grad(axes[1, 1], grad_df, "coherent_gate", "d")
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    probe_handles = [h for h, l in zip(handles, labels) if l in [f"{LABELS[p]} (arch)" for p in PROBES]]
    probe_labels = [LABELS[p] for p in PROBES]
    
    if probe_handles:
        fig.legend(probe_handles, probe_labels, loc='lower center', ncol=4, 
                   bbox_to_anchor=(0.5, -0.04), frameon=False)
    
    fig.subplots_adjust(left=0.08, right=0.96, top=0.96, bottom=0.15, wspace=0.35, hspace=0.45)
    
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig9_chi2_calibration.{ext}"
        fig.savefig(out, dpi=600, facecolor="white", bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)

if __name__ == "__main__":
    make_figure()