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


def conditions(d):
    """Per simulated qubit, every condition that differs between qubits: the two drawn on purpose (coherent share
    of the gate error, prior error envelope), T2*/T2 and the true detuning (stored in the CSV), and the readout
    error and T1 switching rate, rebuilt from each qubit's seed with research/nisq_benchmark.make_qubit (the
    qubit's description only, nothing is re-simulated). The rebuilt qubits are checked against the CSV's stored
    values. Cached in research/figures/nisq_conditions.csv."""
    cache = OUT_DIR / "nisq_conditions.csv"
    if cache.exists():
        return pd.read_csv(cache).set_index(["architecture", "instance_id"])
    import sys
    import warnings
    sys.path.insert(0, str(HERE.parent))
    import nisq_benchmark as nb
    warnings.filterwarnings("ignore")
    base = d[d.arm == "canary_x1"]
    rows = []
    for (arch, inst), g in base.groupby(["architecture", "instance_id"]):
        r0 = g.iloc[0]
        truth, prior, cf, pf = nb.make_qubit(arch, int(r0.seed))
        ratio = float(np.ravel(truth.architecture_specific.get("t2_ramsey_over_echo", [np.nan]))[0])
        assert np.isclose(cf, r0.coherent_fraction, rtol=1e-12) and np.isclose(pf, r0.perturbation_frac, rtol=1e-12)
        assert np.isnan(ratio) == np.isnan(r0.t2_ramsey_over_echo) and (np.isnan(ratio) or np.isclose(ratio, r0.t2_ramsey_over_echo, rtol=1e-12))
        tele = truth.drift_processes.get("T1_telegraph")
        rows.append(dict(architecture=arch, instance_id=int(inst), coherent=cf, prior=pf, ratio=ratio,
                         readout=0.5 * (truth.spam_p0_given_1[0] + truth.spam_p1_given_0[0]),
                         switching=tele.params["rate"] if tele is not None else np.nan,
                         detuning=abs(g[g.parameter == "delta_omega"].true_value.iloc[0]) / (2 * np.pi)))
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    return out.set_index(["architecture", "instance_id"])


WORSE = {"coherent": 1, "readout": 1, "switching": 1, "ratio": -1, "detuning": 1, "prior": 1}


def fifths(k, arch, cond):
    """Qubits of one platform in fifths of a condition, 0 = best, 4 = worst (T2*/T2 is worse when lower)."""
    x = k.loc[arch][cond].dropna()
    return pd.qcut(WORSE[cond] * x, 5, labels=False), x


def robustness_grid(d, k):
    """Every condition x parameter x tool (Canary, QE standard) x platform: Spearman rank correlation of the
    condition (signed so that positive = error grows as the condition worsens) with the per-qubit |error|, its
    Holm-corrected p over the whole grid, and median |error| and 2-sigma coverage in the best and worst fifths."""
    from scipy.stats import spearmanr
    rows = []
    for arch, tag in ARCH:
        for cond in WORSE:
            if k.loc[arch][cond].isna().all():
                continue
            fifth, x = fifths(k, arch, cond)
            for p in PARAMS:
                for arm, tool in (("canary_x1", "Canary"), ("qe_ibm", "QE standard")):
                    g = sel(d, arch, arm, p).set_index("instance_id").join(fifth.rename("f")).join(x.rename("x"))
                    rho, pv = spearmanr(WORSE[cond] * g.x, g.err)
                    med = g.groupby("f").err.median()
                    cov = g.groupby("f").z_score.apply(lambda z: 100 * np.mean(np.abs(z) <= 2))
                    rows.append(dict(platform=tag, condition=cond, parameter=p, tool=tool, n=len(g), rho=rho, p=pv,
                                     error_best=med.iloc[0], error_worst=med.iloc[-1], coverage_best=cov.iloc[0],
                                     coverage_worst=cov.iloc[-1]))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.values)
    return out


def save(fig, name):
    OUT_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=600)
    print("saved:", OUT_DIR / f"{name}.pdf", OUT_DIR / f"{name}.png")
