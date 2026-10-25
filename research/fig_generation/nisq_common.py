"""Shared data loading, statistics and IEEE style for the NISQ-benchmark figures and table (research/data/nisq_benchmark.csv)."""
import pathlib
import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.stats import beta, binom

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "nisq_benchmark.csv"
OUT_DIR = HERE.parent / "figures"
FULL_WIDTH = 7.16
FONT, SIZE = "Cambria", 8
INK, GREY = "#1A1A1A", "#6E6E6E"
PARAMS = ("T1_s", "T2_s", "delta_omega", "epsilon_sx")
ARCH = (("superconducting", "SC"), ("trapped_ion", "TI"))
LABEL = {"T1_s": "$T_1$", "T2_s": "$T_2$", "delta_omega": r"$\Delta\omega$", "epsilon_sx": r"$\varepsilon$"}
BUDGETS = (1080, 2160, 4320, 8640, 17280, 34560, 69120, 138240)


def style():
    mpl.rcParams.update({
        "font.family": FONT, "font.size": SIZE, "axes.labelsize": SIZE, "xtick.labelsize": SIZE,
        "ytick.labelsize": SIZE, "legend.fontsize": SIZE,
        "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold", "mathtext.sf": FONT, "mathtext.fallback": None,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def soften(c, s=0.78):
    """Fig. 1's colour treatment: saturation x s."""
    import colorsys
    h, l, sat = colorsys.rgb_to_hls(*mpl.colors.to_rgb(c))
    return mpl.colors.to_hex(colorsys.hls_to_rgb(h, l, sat * s))


COL = {"T1_s": soften("#0072B2"), "T2_s": soften("#009E73"), "delta_omega": soften("#E69F00"),
       "epsilon_sx": soften("#CC79A7")}
TOOL = {"canary": INK, "qe": soften("#D55E00")}


def load():
    """The CSV with each row's error: relative error in % (Delta-omega: absolute error in Hz)."""
    d = pd.read_csv(DATA)
    dw = d.parameter == "delta_omega"
    d["err"] = np.where(dw, (d.estimate - d.true_value).abs() / (2 * np.pi),
                        100 * (d.estimate / d.true_value - 1).abs())
    d["signed"] = np.where(dw, (d.estimate - d.true_value) / (2 * np.pi), 100 * (d.estimate / d.true_value - 1))
    d["ok"] = d.fit_status == "ok"
    return d


def sel(d, arch, arm, param, ok=True):
    g = d[(d.architecture == arch) & (d.arm == arm) & (d.parameter == param)]
    return g[g.ok] if ok else g


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


def sign_test_p(k, n):
    """Two-sided exact sign test of k successes in n against p = 1/2."""
    return float(min(1.0, 2 * min(binom.cdf(k, n, 0.5), binom.sf(k - 1, n, 0.5))))


def holm(pvals):
    """Holm-adjusted p-values, same order as given."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired(d, arch, param, arm_a, arm_b):
    """Per-qubit errors of two arms where both fitted: (err_a, err_b)."""
    a = sel(d, arch, arm_a, param).set_index("instance_id").err
    b = sel(d, arch, arm_b, param).set_index("instance_id").err
    j = a.index.intersection(b.index)
    return a.loc[j].values, b.loc[j].values


def save(fig, name):
    OUT_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=600)
    print("saved:", OUT_DIR / f"{name}.pdf", OUT_DIR / f"{name}.png")
