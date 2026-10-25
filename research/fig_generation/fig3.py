"""Fig. 3 -- NISQ regime: Canary (8,640 shots) against qiskit-experiments (research/data/nisq_benchmark.csv).
(a) per-qubit error ratio Canary / QE at equal shots (8,640); (b) the same against IBM's standard QE
configuration (92,000 shots); (c) 68.3% and 95.4% error-bar coverage of the three arms.
QE failed fits are excluded (conservative for Canary; counts in Table I). IEEE double column (7.16 in).
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import nisq_common as c

c.style()
d = c.load()
ARMS = (("canary_x1", "Canary, 8,640"), ("qe_equal", "QE, 8,640"), ("qe_ibm", "QE standard, 92,000"))
COMPARE = (("qe_equal", "(a) Canary / QE at equal shots (8,640 each)"),
           ("qe_ibm", "(b) Canary (8,640) / QE standard configuration (92,000)"))

stats = {}
for arm, _ in COMPARE:
    for p in c.PARAMS:
        for arch, _ in c.ARCH:
            ea, eb = c.paired(d, arch, p, "canary_x1", arm)
            keep = ea != eb
            r = np.log10(ea[keep] / eb[keep])
            k = int(np.sum(r < 0))
            stats[arm, p, arch] = (r, k, len(r), c.sign_test_p(k, len(r)))
adj = dict(zip(stats, c.holm([v[3] for v in stats.values()])))

W, H = c.FULL_WIDTH, 4.1
fig = plt.figure(figsize=(W, H))
axes_ratio = [fig.add_axes([0.068, 0.555, 0.43, 0.385]), fig.add_axes([0.562, 0.555, 0.43, 0.385])]
ax_c = fig.add_axes([0.068, 0.115, 0.924, 0.265])

for ax, (arm, title) in zip(axes_ratio, COMPARE):
    for i, p in enumerate(c.PARAMS):
        for j, (arch, tag) in enumerate(c.ARCH):
            r, k, n, _ = stats[arm, p, arch]
            x = 2.6 * i + j
            q05, q25, q50, q75, q95 = np.percentile(r, [5, 25, 50, 75, 95])
            face = c.COL[p] if j == 0 else "white"
            ax.add_patch(mpl.patches.Rectangle((x - 0.32, q25), 0.64, q75 - q25, facecolor=face,
                                               edgecolor=c.COL[p], lw=0.8, zorder=3))
            ax.plot([x, x], [q05, q25], color=c.COL[p], lw=0.8, zorder=2)
            ax.plot([x, x], [q75, q95], color=c.COL[p], lw=0.8, zorder=2)
            ax.plot([x - 0.32, x + 0.32], [q50, q50], color=c.INK if j == 0 else c.COL[p], lw=1.3, zorder=4,
                    solid_capstyle="butt")
            mark = "" if adj[arm, p, arch] < 0.05 else "$^{\\mathrm{ns}}$"
            ax.text(x, 1.93, f"{100 * k / n:.0f}{mark}", ha="center", va="top")
    ax.axhline(0, color=c.GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_ylim(-2.45, 2)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
    ax.set_xlim(-0.7, 2.6 * 3 + 1.7)
    ax.set_xticks([2.6 * i + j for i in range(4) for j in range(2)])
    ax.set_xticklabels([t for _ in c.PARAMS for _, t in c.ARCH])
    for i, p in enumerate(c.PARAMS):
        ax.text(2.6 * i + 0.5, -0.13, c.LABEL[p], transform=ax.get_xaxis_transform(), ha="center", va="top")
    ax.text(0.5, -0.27, title, transform=ax.transAxes, ha="center", va="top")
axes_ratio[0].set_ylabel("Per-qubit error ratio")
axes_ratio[0].text(-0.62, -2.38, "below 1: Canary closer to truth", ha="left", va="bottom", color=c.GREY)
fig.text(0.068, 0.985, "Numbers: % of qubits on which Canary is closer to the truth "
         "(ns: not significant, two-sided sign test, Holm-corrected over all 16)", ha="left", va="top")

markers = ("o", "s", "D")
for i, p in enumerate(c.PARAMS):
    for j, (arch, tag) in enumerate(c.ARCH):
        for m, (arm, _) in enumerate(ARMS):
            z = np.abs(c.sel(d, arch, arm, p).z_score.values)
            x = 4.4 * i + 1.6 * j + (m - 1) * 0.42
            col = c.TOOL["canary"] if m == 0 else c.TOOL["qe"]
            for k_sig, filled in ((2, True), (1, False)):
                k = int(np.sum(z <= k_sig))
                lo, hi = c.clopper(k, len(z))
                ax_c.plot([x, x], [100 * lo, 100 * hi], color=col, lw=0.8, zorder=2)
                ax_c.plot(x, 100 * k / len(z), marker=markers[m], ms=3.2, mew=0.8, color=col,
                          mfc=col if filled else "white", ls="none", zorder=3)
for level in (68.27, 95.45):
    ax_c.axhline(level, color=c.GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
ax_c.set_ylim(0, 100)
ax_c.set_yticks([0, 20, 40, 68.27, 95.45])
ax_c.set_yticklabels(["0", "20", "40", "68.3", "95.4"])
ax_c.set_ylabel("Coverage (%)")
ax_c.set_xlim(-0.8, 4.4 * 3 + 2.4)
ax_c.set_xticks([4.4 * i + 1.6 * j for i in range(4) for j in range(2)])
ax_c.set_xticklabels([t for _ in c.PARAMS for _, t in c.ARCH])
for i, p in enumerate(c.PARAMS):
    ax_c.text(4.4 * i + 0.8, -0.15, c.LABEL[p], transform=ax_c.get_xaxis_transform(), ha="center", va="top")
handles = [mpl.lines.Line2D([], [], marker=mk, ls="none", ms=3.2, color=c.TOOL["canary"] if m == 0 else c.TOOL["qe"],
                            label=lab) for m, ((_, lab), mk) in enumerate(zip(ARMS, markers))]
handles += [mpl.lines.Line2D([], [], ls="none", label=" "),
            mpl.lines.Line2D([], [], marker="o", ls="none", ms=3.2, color=c.GREY, label="$|z| \\leq 2$ (filled)"),
            mpl.lines.Line2D([], [], marker="o", ls="none", ms=3.2, color=c.GREY, mfc="white",
                             label="$|z| \\leq 1$ (open)")]
ax_c.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=3, frameon=False,
            handletextpad=0.2, columnspacing=1.4, borderaxespad=0.2)
ax_c.text(0.5, -0.31, "(c) Error-bar coverage", transform=ax_c.transAxes, ha="center", va="top")

c.save(fig, "fig3")
for key, (r, k, n, p) in stats.items():
    print(key, f"n={n} better={100 * k / n:.1f}% median ratio={10 ** np.median(r):.3f} p_holm={adj[key]:.2g}")
