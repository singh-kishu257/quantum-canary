"""Fig. 2 -- ideal regime: Canary's estimates against the simulator's truth (research/data/ideal_parity.csv).
(a) signed relative error of T1, T2, eps; (b) signed delta-omega error; (c) 68.3% and 95.4% error-bar coverage;
(d) fraction of qubits over each probe's chi2/dof limit against the nominal 1.25%. RevTeX single column (3.40 in).
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import beta, binom, chi2
import nisq_common as nc

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "ideal_parity.csv"

nc.style()
SIZE = nc.SIZE
INK, GREY = "#1A1A1A", "#6E6E6E"


def soften(c, s=0.78):
    """Fig. 1's colour treatment: saturation x s."""
    import colorsys
    h, l, sat = colorsys.rgb_to_hls(*mpl.colors.to_rgb(c))
    return mpl.colors.to_hex(colorsys.hls_to_rgb(h, l, sat * s))


COL = {"T1_s": soften("#0072B2"), "T2_s": soften("#009E73"), "delta_omega": soften("#E69F00"),
       "epsilon_sx": soften("#CC79A7")}
ARCH = (("superconducting", "SC"), ("trapped_ion", "TI"))
LABEL = {"T1_s": "$T_1$", "T2_s": "$T_2$", "delta_omega": r"$\Delta\omega$", "epsilon_sx": r"$\varepsilon$"}
PROBE = {"T1_s": ("T1", 2), "T2_s": ("echo", 2), "delta_omega": ("Ramsey", 6), "epsilon_sx": ("RB", 1)}
NOMINAL = 0.0125


def median_ci(x):
    """Median and its exact 95% confidence interval from order statistics."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    lo, hi = int(binom.ppf(0.025, n, 0.5)), int(binom.isf(0.025, n, 0.5))
    return float(np.median(x)), float(x[max(lo - 1, 0)]), float(x[min(hi, n - 1)])


def clopper(k, n):
    """Exact (Clopper-Pearson) 95% interval of a proportion k/n."""
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def strip(ax, x, y, col, rng):
    ax.scatter(x + rng.uniform(-0.17, 0.17, len(y)), y, s=1.6, color=col, alpha=0.55, lw=0, zorder=2,
               rasterized=True)
    m, lo, hi = median_ci(y)
    ax.plot([x - 0.21, x + 0.21], [m, m], color=INK, lw=1.3, zorder=4, solid_capstyle="butt")
    ax.plot([x, x], [lo, hi], color=INK, lw=0.9, zorder=4)


def panel_label(ax, s, y=-0.30):
    ax.text(0.5, y, s, transform=ax.transAxes, ha="center", va="top")


d = pd.read_csv(DATA)
assert set(d.fit_status) == {"ok"}, "a failed fit would be silently dropped"
n_inst = d.groupby(["architecture", "parameter"]).size()
assert n_inst.nunique() == 1, n_inst
N = int(n_inst.iloc[0])
rng = np.random.default_rng(2)

W, H = nc.COL_WIDTH, 4.75
fig = plt.figure(figsize=(W, H))
ax_a = fig.add_axes([0.155, 0.705, 0.825, 0.265])
ax_b1 = fig.add_axes([0.155, 0.370, 0.115, 0.215])
ax_b2 = fig.add_axes([0.375, 0.370, 0.115, 0.215])
ax_c = fig.add_axes([0.640, 0.370, 0.340, 0.215])
ax_d = fig.add_axes([0.155, 0.120, 0.825, 0.152])

# (a) signed relative error, T1 / T2 / eps
params_a = ("T1_s", "T2_s", "epsilon_sx")
lim = 0.0
for i, p in enumerate(params_a):
    for j, (arch, _) in enumerate(ARCH):
        y = 100 * d[(d.parameter == p) & (d.architecture == arch)].eval("canary_value / true_value - 1").values
        strip(ax_a, 3 * i + j, y, COL[p], rng)
        lim = max(lim, np.abs(y).max())
lim = 5 * np.ceil(lim / 5)
ax_a.axhline(0, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
ax_a.set_xlim(-0.6, 3 * len(params_a) - 1.4)
ax_a.set_ylim(-lim, lim)
ax_a.set_xticks([3 * i + j for i in range(len(params_a)) for j in range(2)])
ax_a.set_xticklabels([a for _ in params_a for _, a in ARCH])
for i, p in enumerate(params_a):
    ax_a.text(3 * i + 0.5, -0.14, LABEL[p], transform=ax_a.get_xaxis_transform(), ha="center", va="top")
ax_a.set_ylabel("Signed relative error (%)")
ax_a.text(0.02, 0.97, f"$n$ = {N} per strip; bar: median, 95% CI", transform=ax_a.transAxes,
          ha="left", va="top")
panel_label(ax_a, "(a)", y=-0.27)

# (b) signed delta-omega error: SC in Hz, TI in mHz
for ax, (arch, tag), scale, unit in ((ax_b1, ARCH[0], 1.0, "Hz"), (ax_b2, ARCH[1], 1e3, "mHz")):
    g = d[(d.parameter == "delta_omega") & (d.architecture == arch)]
    y = scale * (g.canary_value - g.true_value).values / (2 * np.pi)
    strip(ax, 0, y, COL["delta_omega"], rng)
    ax.axhline(0, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    m = np.abs(y).max() * 1.08
    ax.set_ylim(-m, m)
    ax.set_xlim(-0.45, 0.45)
    ax.set_xticks([0])
    ax.set_xticklabels([tag])
    ax.set_title(f"({unit})", fontsize=SIZE, pad=3)
ax_b1.set_ylabel(r"$\Delta\omega/2\pi$ error")
fig.text((0.155 + 0.490) / 2, 0.370 - 0.30 * 0.215, "(b)", ha="center", va="top")

# (c) coverage at 1 and 2 sigma
params_c = ("T1_s", "T2_s", "delta_omega", "epsilon_sx")
for i, p in enumerate(params_c):
    for j, (arch, _) in enumerate(ARCH):
        z = np.abs(d[(d.parameter == p) & (d.architecture == arch)].z_score.values)
        for k_sig, marker in ((2, "o"), (1, "s")):
            k = int(np.sum(z <= k_sig))
            lo, hi = clopper(k, len(z))
            x = i + (j - 0.5) * 0.36
            ax_c.plot([x, x], [100 * lo, 100 * hi], color=COL[p], lw=0.8, zorder=2)
            ax_c.plot(x, 100 * k / len(z), marker=marker, ms=3.4, mew=0.8, color=COL[p],
                      mfc=COL[p] if j == 0 else "white", zorder=3, ls="none")
for level in (68.27, 95.45):
    ax_c.axhline(level, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
ax_c.set_xticks(range(len(params_c)))
ax_c.set_xticklabels([LABEL[p] for p in params_c])
ax_c.set_xlim(-0.55, len(params_c) - 0.45)
ax_c.set_ylim(55, 100)
ax_c.set_yticks([60, 68.27, 80, 95.45])
ax_c.set_yticklabels(["60", "68.3", "80", "95.4"])
ax_c.set_ylabel("Coverage (%)")
handles = [mpl.lines.Line2D([], [], marker="o", ls="none", ms=3.4, color=INK, label="SC"),
           mpl.lines.Line2D([], [], marker="o", ls="none", ms=3.4, color=INK, mfc="white", label="TI")]
ax_c.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.63), ncol=2, frameon=False, handletextpad=0.1,
            columnspacing=1.2, borderaxespad=0.0)
panel_label(ax_c, "(c)", y=-0.30)

# (d) chi2/dof over its limit
for i, p in enumerate(params_c):
    probe, dof = PROBE[p]
    limit = chi2.ppf(1 - NOMINAL, dof) / dof
    for j, (arch, _) in enumerate(ARCH):
        c = d[(d.parameter == p) & (d.architecture == arch)].chi2_dof.values
        k = int(np.sum(c > limit))
        lo, hi = clopper(k, len(c))
        x = i + (j - 0.5) * 0.30
        ax_d.plot([x, x], [100 * lo, 100 * hi], color=COL[p], lw=0.8, zorder=2)
        ax_d.plot(x, 100 * k / len(c), marker="o", ms=3.4, mew=0.8, color=COL[p],
                  mfc=COL[p] if j == 0 else "white", zorder=3, ls="none")
ax_d.axhline(100 * NOMINAL, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
ax_d.text(-0.45, 4.3, "dashed: nominal 1.25%", ha="left", va="center", color=GREY)
ax_d.set_xticks(range(len(params_c)))
ax_d.set_xticklabels([PROBE[p][0] + "\n" + f"{PROBE[p][1]} dof" for p in params_c])
ax_d.set_xlim(-0.5, len(params_c) - 0.5)
ax_d.set_ylim(0, 5)
ax_d.set_ylabel(r"Over $\chi^2$/dof limit (%)")
panel_label(ax_d, "(d)", y=-0.46)

nc.save(fig, "fig2")
