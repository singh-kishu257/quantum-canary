"""Fig. 5 -- robustness in the NISQ simulation (research/data/nisq_benchmark.csv): Canary (8,640 shots) and
qiskit-experiments in IBM's standard configuration (92,000 shots) across every condition that differs between the
simulated qubits (nisq_common.conditions), each against the parameter whose probe model it enters most directly.
Qubits in fifths of the condition, worse to the right; top row median |error| with its 95% CI, bottom row 2-sigma
coverage with its 95% CI. Each top panel lists its trends that are significant (Spearman rank correlation of the
condition with the per-qubit error, Holm over all 80 condition x parameter x tool x platform tests of Table S6).
Mechanisms of fixed strength (T1 drift, 1/f and OU frequency drift, motional heating, detuning during gates, drift
during the measurement) act on every qubit with its own realization and are not varied here.
QE failed fits are excluded (conservative for Canary). RevTeX page width (7.06 in).
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import nisq_common as c

c.style()
d = c.load()
k = c.conditions(d)
ARMS = (("canary_x1", c.TOOL["canary"], "Canary"), ("qe_ibm", c.TOOL["qe"], "QE standard"))
PANELS = (("coherent", "epsilon_sx", "Coherent share of\ngate error", r"$\varepsilon$ (%)"),
          ("readout", "T1_s", "Readout error, mean of\n$P(0|1)$ and $P(1|0)$", r"$T_1$ (%)"),
          ("switching", "T1_s", "$T_1$ switching rate\n(1/s)", r"$T_1$ (%)"),
          ("ratio", "delta_omega", "$T_2^*/T_2$\n(lower: more slow dephasing)", r"$\Delta\omega/2\pi$ (Hz)"),
          ("detuning", "delta_omega", "Detuning $|\\Delta\\omega|/2\\pi$\n(kHz, SC)", r"$\Delta\omega/2\pi$ (Hz)"),
          ("prior", "T1_s", "Calibration prior's\nrelative error", r"$T_1$ (%)"))
grid = c.robustness_grid(d, k)
SIGNIFICANT = {(r.platform, r.condition, r.parameter, r.tool) for r in grid[(grid.p_holm < 0.05) & (grid.rho > 0)].itertuples()}


def binned(arch, arm, param, cond):
    fifth, x = c.fifths(k, arch, cond)
    g = c.sel(d, arch, arm, param).set_index("instance_id").join(fifth.rename("fifth")).join(x.rename("x"))
    out = []
    for _, b in g.groupby("fifth"):
        m, lo, hi = c.median_ci(b.err)
        n_in, n = int(np.sum(np.abs(b.z_score) <= 2)), len(b)
        clo, chi = c.clopper(n_in, n)
        out.append((b.x.median(), m, lo, hi, 100 * n_in / n, 100 * clo, 100 * chi))
    return np.array(out)


W, H = c.FULL_WIDTH, 3.75
fig, axes = plt.subplots(2, len(PANELS), figsize=(W, H), sharex="col")
fig.subplots_adjust(left=0.06, right=0.995, top=0.865, bottom=0.17, wspace=0.52, hspace=0.12)
scale = {"detuning": 1e-3}
for col, (cond, param, xlabel, ylabel) in enumerate(PANELS):
    ax_e, ax_c = axes[0, col], axes[1, col]
    archs = [(a, t) for a, t in c.ARCH if k.loc[a][cond].notna().any() and not (cond == "detuning" and t == "TI")]
    trends = []
    for arch, tag in archs:
        for arm, colour, label in ARMS:
            v = binned(arch, arm, param, cond)
            xs = v[:, 0] * scale.get(cond, 1)
            face = colour if tag == "SC" else "white"
            ls = "-" if tag == "SC" else (0, (3, 1.5))
            ax_e.fill_between(xs, v[:, 2], v[:, 3], color=colour, alpha=0.13, lw=0)
            ax_e.plot(xs, v[:, 1], ls=ls, color=colour, lw=0.9, marker="o", ms=2.6, mfc=face, mec=colour, mew=0.6)
            ax_c.fill_between(xs, v[:, 5], v[:, 6], color=colour, alpha=0.13, lw=0)
            ax_c.plot(xs, v[:, 4], ls=ls, color=colour, lw=0.9, marker="o", ms=2.6, mfc=face, mec=colour, mew=0.6)
            if (tag, cond, param, label) in SIGNIFICANT:
                trends.append(("Canary" if label == "Canary" else "QE") + ("" if len(archs) == 1 else f" {tag}"))
    ax_e.set_ylim(0, None)
    ax_e.set_ylabel(ylabel, labelpad=1)
    ax_e.text(0.04, 0.97, "trend: " + (", ".join(trends) if trends else "none"), transform=ax_e.transAxes,
              ha="left", va="top", fontsize=c.SIZE - 1, color=c.INK if trends else c.GREY)
    ax_c.axhline(95.45, color=c.GREY, lw=0.6, ls=(0, (3, 2)))
    ax_c.set_ylim(48, 101)
    ax_c.set_yticks([50, 70, 95.45])
    ax_c.set_yticklabels(["50", "70", "95.4"])
    ax_c.set_xlabel(xlabel, fontsize=c.SIZE - 1, labelpad=2)
    ax_c.tick_params(axis="x", labelsize=c.SIZE - 1)
    if c.WORSE[cond] < 0:
        ax_c.invert_xaxis()
    if cond == "readout":
        ax_c.set_xscale("log")
        ax_c.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{100 * x:g}%"))
        ax_c.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax_c.set_xticks([0.003, 0.01, 0.03])
    ax_e.text(0.5, 1.02, f"({'abcdef'[col]})", transform=ax_e.transAxes, ha="center", va="bottom")
axes[1, 0].set_ylabel("2$\\sigma$ coverage (%)", labelpad=1)
axes[1, 2].xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x * 1e4:g}"))
axes[1, 2].set_xlabel("$T_1$ switching rate\n($10^{-4}$/s)", fontsize=c.SIZE - 1, labelpad=2)
handles = [mpl.lines.Line2D([], [], color=colour, lw=0.9, label=label) for _, colour, label in ARMS]
handles += [mpl.lines.Line2D([], [], color=c.GREY, lw=0.9, marker="o", ms=2.6, label="superconducting"),
            mpl.lines.Line2D([], [], color=c.GREY, lw=0.9, ls=(0, (3, 1.5)), marker="o", ms=2.6, mfc="white",
                             label="trapped ion"),
            mpl.patches.Patch(color=c.GREY, alpha=0.3, lw=0, label="95% CI"),
            mpl.lines.Line2D([], [], color=c.GREY, lw=0.6, ls=(0, (3, 2)), label="nominal 95.4%")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=6, frameon=False,
           handletextpad=0.4, columnspacing=1.2, handlelength=1.8)
c.save(fig, "fig5")
