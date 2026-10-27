"""Fig. 7 -- drift on IBM Kingston over the full-chip sandwich C1 QE1 C2 QE2 C3 (about one hour;
research/data/hardware/ibm_kingston_20261026T014902Z), every measurement used (hw_common.drift).
(a) Per parameter, each qubit on the chip's layout (from IBM's coupling map). Coloured: Canary's value changed
across its three checks (Cochran's Q, 2 dof, Holm over every qubit and parameter), by how much beyond its error
bars (between-check SD tau, DerSimonian-Laird; % of the qubit's mean, delta-omega in Hz; square-root scale capped
at the 95th percentile of the changed qubits). Filled grey: no change detected (tau is not shown there: with three
checks it exceeds zero by chance for 37% of unchanged qubits). Open grey: not analysed. (b) Do the two tools see the same change? Per qubit, QE's change QE2 - QE1 against
Canary's change over the same interval (its trajectory interpolated linearly to the times of QE's two jobs);
Spearman rho with its 95% CI (Fieller), Holm over the four parameters; points beyond the 98th percentile drawn
at the edge. A change after QE2 (seen only by C3) enters Canary's interpolated change in part. QE: floor-fixed refits for T1 and epsilon.
Delta-omega only where both QE values lie inside Canary's unambiguous range. RevTeX page width (7.06 in).
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import hw_common as h
import nisq_common as c

c.style()
pos, pairs = h.chip_layout()
dr = h.drift()
PARAMS = (("T1_s", "$T_1$", "%"), ("T2_s", "$T_2$", "%"), ("delta_omega", r"$\Delta\omega/2\pi$", "Hz"),
          ("epsilon_sx", r"$\varepsilon$", "%"))
rho = {p: spearmanr(dr[dr.parameter == p].canary_change, dr[dr.parameter == p].qe_change) for p, _, _ in PARAMS}
p_holm = dict(zip(rho, c.holm([rho[p].pvalue for p in rho])))
cmap = mpl.colormaps["YlGnBu"]

W, H = c.FULL_WIDTH, 4.1
fig = plt.figure(figsize=(W, H))
for i, (p, label, unit) in enumerate(PARAMS):
    g = dr[dr.parameter == p].set_index("qubit")
    x0 = 0.005 + 0.25 * i
    ax = fig.add_axes([x0, 0.545, 0.195, 0.345])
    cax = fig.add_axes([x0 + 0.198, 0.585, 0.008, 0.265])
    sig = g.canary_p_holm < 0.05
    norm = mpl.colors.PowerNorm(0.5, 0, float(np.ceil(np.percentile(g.canary_tau[sig], 95))))
    for a, b in pairs:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color="#D2D2D2", lw=0.5, zorder=1)
    for q, (x, y) in pos.items():
        if q in g.index and sig[q]:
            ax.scatter(x, y, s=16, color=cmap(norm(g.canary_tau[q])), edgecolor=c.INK, lw=0.5, zorder=3)
        elif q in g.index:
            ax.scatter(x, y, s=9, color="#DCDCDC", edgecolor="none", zorder=2)
        else:
            ax.scatter(x, y, s=9, facecolor="white", edgecolor="#B4B4B4", lw=0.4, zorder=2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.8, 15.8)
    ax.set_ylim(-14.8, 0.8)
    ax.axis("off")
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="max")
    cb.ax.tick_params(labelsize=c.SIZE - 2, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    cb.set_label(f"$\\tau$ ({unit})", fontsize=c.SIZE - 1, labelpad=1)
    ax.set_title(f"{label}: {int((g.canary_p_holm < 0.05).sum())} of {len(g)} changed", fontsize=c.SIZE, pad=2)

    ax = fig.add_axes([0.055 + 0.25 * i, 0.09, 0.17, 0.30])
    lim = float(np.percentile(np.abs(np.r_[g.canary_change, g.qe_change]), 98))
    xs, ys = np.clip(g.qe_change, -lim, lim), np.clip(g.canary_change, -lim, lim)
    ax.axhline(0, color="#D2D2D2", lw=0.5, zorder=0)
    ax.axvline(0, color="#D2D2D2", lw=0.5, zorder=0)
    ax.plot([-lim, lim], [-lim, lim], color=c.GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.scatter(xs, ys, s=4, facecolor=c.INK, edgecolor="none", alpha=0.6, zorder=2, rasterized=True)
    ax.set_xlim(-1.05 * lim, 1.05 * lim)
    ax.set_ylim(-1.05 * lim, 1.05 * lim)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=c.SIZE - 1)
    ax.set_xlabel(f"QE change ({unit})", fontsize=c.SIZE - 1, labelpad=1)
    ax.set_ylabel(f"Canary change ({unit})", fontsize=c.SIZE - 1, labelpad=1)
    r, n = rho[p].statistic, len(g)
    lo, hi = np.tanh(np.arctanh(r) + np.array([-1.96, 1.96]) * np.sqrt(1.06 / (n - 3)))
    ph = p_holm[p]
    mant, expo = f"{ph:.1e}".split("e")
    p_txt = f"$p = {mant}\\times10^{{{int(expo)}}}$" if ph < 0.01 else f"$p$ = {ph:.2f}"
    ax.text(0.03, 0.98, f"$\\rho$ = {r:+.2f}\n[{lo:+.2f}, {hi:+.2f}]\n" + p_txt,
            transform=ax.transAxes, ha="left", va="top", fontsize=c.SIZE - 1, zorder=5,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))
    ax.set_title(label, fontsize=c.SIZE, pad=2)
fig.text(0.5, 0.97, "(a) Canary: qubits that changed across C1, C2, C3 (Holm), coloured by the size of the change; grey: no change detected",
         ha="center", va="center")
fig.text(0.5, 0.475, "(b) Same qubits, same interval: QE2 $-$ QE1 against Canary interpolated to QE's job times",
         ha="center", va="center")
c.save(fig, "fig7")

for p, label, unit in PARAMS:
    g = dr[dr.parameter == p]
    print(f"{p:12s} n {len(g):3d} | Canary changed {int((g.canary_p_holm < 0.05).sum()):3d}, median tau {g.canary_tau.median():.2f} {unit}"
          f" | QE changed {int((g.qe_p_holm < 0.05).sum()):3d}, median tau {g.qe_tau.median():.2f} {unit}"
          f" | rho {rho[p].statistic:+.3f}, p {rho[p].pvalue:.1e}, Holm {p_holm[p]:.1e}")
