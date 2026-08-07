import importlib.util, pathlib, sys, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone

_spec = importlib.util.spec_from_file_location(
    "inversion", pathlib.Path(__file__).parent / "1_inversion.py")
inv = importlib.util.module_from_spec(_spec)
sys.modules["inversion"] = inv
_spec.loader.exec_module(inv)

SCRIPT_DIR  = pathlib.Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

N_INSTANCES   = 300
SHOTS_T1      = 300
SHOTS_RAMSEY  = 1000 
SHOTS_GATE    = 500
SHOTS_ECHO    = 500
SEED          = 43
ARCHITECTURES = ["superconducting", "trapped_ion", "neutral_atom"]
rng = np.random.default_rng(SEED)

COLORS  = {"superconducting": "#1f77b4", "trapped_ion": "#ff7f0e", "neutral_atom": "#2ca02c"}
MARKERS = {"superconducting": "o",       "trapped_ion": "s",       "neutral_atom": "^"}
LABELS  = {"superconducting": "Superconducting", "trapped_ion": "Trapped ion", "neutral_atom": "Neutral atom"}


def shot_budget(arch_name: str) -> int:
    return 3*SHOTS_T1 + 6*SHOTS_RAMSEY + 3*SHOTS_GATE + 3*SHOTS_ECHO


# Realistic current-hardware sampling ranges (not just ±40% of the arch
# default prior) — used for both Fig. 2 and Fig. 3 true-parameter draws.
TRUE_PARAM_RANGES = {
    "superconducting": {
        "T1_s": (80e-6, 400e-6),      # IBM Eagle/Heron, Jurcevic et al. (2021)
        "T2_s": (40e-6, 200e-6),
        "eps":  (1e-4, 2e-3),         # IBM published gate errors
    },
    "trapped_ion": {
        "T1_s": (100.0, 10000.0),     # IonQ Forte / Quantinuum H2
        "T2_s": (0.1, 3.0),           # T2* Ramsey, B-field limited
        "eps":  (1e-4, 2e-3),         # IonQ published gate errors
    },
    "neutral_atom": {
        "T1_s": (1.0, 100.0),         # QuEra/Rydberg cloud systems
        "T2_s": (0.1, 3.0),           # current cloud-accessible T2*
        "eps":  (1e-3, 1e-2),         # neutral atom gate errors
    },
}


def _log_uniform(r, lo: float, hi: float) -> float:
    return float(10 ** r.uniform(np.log10(lo), np.log10(hi)))


def sample_true_t1_t2_eps(arch_name: str, rng_obj=None):
    r = rng_obj if rng_obj is not None else rng
    ranges = TRUE_PARAM_RANGES[arch_name]
    T1_lo, T1_hi = ranges["T1_s"]
    T2_lo, T2_hi = ranges["T2_s"]
    eps_lo, eps_hi = ranges["eps"]

    # T1 spans >1 decade for trapped_ion/neutral_atom -> log-uniform; for
    # superconducting it spans <1 decade -> linear uniform is fine either way,
    # but log-uniform is used whenever the ratio exceeds 10x.
    if T1_hi / T1_lo > 10.0:
        T1 = _log_uniform(r, T1_lo, T1_hi)
    else:
        T1 = r.uniform(T1_lo, T1_hi)

    T2 = r.uniform(T2_lo, T2_hi)
    T2 = min(T2, 2.0*T1)

    eps = _log_uniform(r, eps_lo, eps_hi)
    return T1, T2, eps


def sample_dw(dw_max: float, rng_obj=None):
    r = rng_obj if rng_obj is not None else rng
    return r.choice([-1,1]) * r.uniform(0.2*dw_max, dw_max)


def sample_true_params(arch_name: str, dw_max: float, rng_obj=None):
    r = rng_obj if rng_obj is not None else rng
    T1, T2, eps = sample_true_t1_t2_eps(arch_name, r)
    dw = sample_dw(dw_max, r)
    return T1, T2, dw, eps


def _spawn_probe_rngs(base_seed: int, *keys, n_streams: int = 4):
    """Independent, reproducible child RNGs for the t1/ramsey/gate/echo probe
    families (n_streams=4), plus an optional 5th stream (n_streams=5) for
    the ±15% live-calibration prior jitter used in Fig. 3's live-cal arms.
    The extra stream is appended so the first 4 streams are unaffected."""
    ss = np.random.SeedSequence([base_seed, *keys])
    streams = ss.spawn(n_streams)
    rngs = [np.random.default_rng(s) for s in streams]
    return tuple(rngs)


def simulate_inversion(arch_name, T1_true, T2_true, dw_true, eps_true, instance_id,
                       use_true_prior=True):
    if use_true_prior:
        # Fig. 2 — ideal hardware: priors = truth, prior_confidence="live",
        # so delays are linearly centred on the true decay (Strategy A).
        profile = inv.BackendProfile.from_true_params(arch_name, T1_true, T2_true)
        # gate_rep_n is driven by constants["eps_typical"], which
        # from_true_params does not touch. For the "ideal" scenario the
        # gate-rep probe should likewise be centred on the true epsilon
        # (same rationale as the T1/T2 delay-centring above) — otherwise
        # a true epsilon far from the arch-typical value saturates or
        # under-drives the fixed [n_low, n_mid, n_max] repetition counts.
        custom_arch = dict(inv.ARCH_DEFAULTS[arch_name])
        custom_arch["eps_typical"] = eps_true
        profile.custom_arch = custom_arch
    else:
        # Fig. 3 — real-world: arch-default priors, prior_confidence=
        # "arch_default", so delays are log-spaced across the full plausible
        # range (Strategy B), independent of the (unknown) true values.
        profile = inv.BackendProfile.from_architecture(arch_name)
    arch     = inv.ARCH_DEFAULTS[arch_name]
    _, meta  = inv.build_probe_circuits(profile)

    t1_delays     = meta["t1_delays_s"]
    ramsey_delays = meta["ramsey_delays_s"]
    echo_delays   = meta["echo_delays_s"]
    Nv            = np.array(meta["gate_rep_N"], dtype=float)

    arch_idx = ARCHITECTURES.index(arch_name)
    rng_t1, rng_ramsey, rng_gate, rng_echo = _spawn_probe_rngs(SEED, arch_idx, instance_id)

    p0_given_1 = arch["p0_given_1"]
    p1_given_0 = arch["p1_given_0"]
    def spam(p):
        return p*(1.0 - p0_given_1) + (1.0 - p)*p1_given_0

    def c1(rng_obj, p, sh):
        n1 = int(rng_obj.binomial(sh, float(np.clip(p,0,1))))
        return {"0": sh-n1, "1": n1}
    def c0(rng_obj, p, sh):
        n0 = int(rng_obj.binomial(sh, float(np.clip(p,0,1))))
        return {"0": n0, "1": sh-n0}

    t1_counts = [c1(rng_t1, spam(float(inv.forward_t1(d, T1_true))), SHOTS_T1) for d in t1_delays]

    ramsey_counts = []
    for t in ramsey_delays:
        px, py = inv.forward_ramsey_xy(t, T2_true, dw_true)
        ramsey_counts.append(c1(rng_ramsey, px, SHOTS_RAMSEY))
        ramsey_counts.append(c1(rng_ramsey, py, SHOTS_RAMSEY))

    p0_gate = inv.forward_gate(Nv, eps_true)
    gate_counts = [c0(rng_gate, spam(p), SHOTS_GATE) for p in p0_gate]

    T2_echo_true = min(T2_true, 2.0*T1_true)
    echo_counts = [c1(rng_echo, spam(float(inv.forward_echo(t, T2_echo_true))), SHOTS_ECHO) for t in echo_delays]

    counts_list = t1_counts + ramsey_counts + gate_counts + echo_counts

    return inv.lindblad_inversion(
        counts_list, meta, profile,
        shots_t1=SHOTS_T1, shots_ramsey=SHOTS_RAMSEY,
        shots_gate=SHOTS_GATE, shots_echo=SHOTS_ECHO,
        qubit_id=0, timestamp=datetime.now(timezone.utc).isoformat())


def compute_metrics(true_vals, rec_vals):
    t = np.array(true_vals); r = np.array(rec_vals)
    ss_res = np.sum((r-t)**2); ss_tot = np.sum((t-np.mean(t))**2)
    r2   = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((r-t)**2)))
    return r2, rmse


def run_experiment(use_true_prior: bool = True):
    rec = {k: [] for k in ["arch",
        "T1_true","T1_rec","T1_sig",
        "T2_true","T2_rec","T2_sig",
        "T2_ramsey_rec","T2_ramsey_sig",
        "T2_echo_rec","T2_echo_sig",
        "dw_true","dw_rec","dw_sig",
        "eps_true","eps_rec","eps_sig",
        "t1_resid","ram_resid","gate_resid","echo_resid",
        "t1_chi2","ramsey_chi2","gate_chi2","echo_chi2"]}

    fit_failures  = {arch: 0 for arch in ARCHITECTURES}
    fit_attempts  = {arch: 0 for arch in ARCHITECTURES}

    for arch in ARCHITECTURES:
        arch_default_dw_max = inv.BackendProfile.from_architecture(arch).dw_max_rad_s
        success = 0
        while success < N_INSTANCES:
            T1_t, T2_t, eps_t = sample_true_t1_t2_eps(arch)
            if use_true_prior:
                # dw_max must reflect the delay grid actually used for this
                # instance (Strategy A, centred on T1_t/T2_t), not a fixed
                # arch-default t1 — otherwise the Ramsey phase can alias
                # (|dw*t1| > pi) and corrupt the arctan2 sign/angle recovery.
                dw_max = inv.BackendProfile.from_true_params(arch, T1_t, T2_t).dw_max_rad_s
            else:
                dw_max = arch_default_dw_max
            dw_t = sample_dw(dw_max)
            fit_attempts[arch] += 1
            try:
                r = simulate_inversion(arch, T1_t, T2_t, dw_t, eps_t, fit_attempts[arch],
                                       use_true_prior=use_true_prior)
            except (RuntimeError, ValueError):
                fit_failures[arch] += 1
                continue
            if not (np.isfinite(r.T1_s) and np.isfinite(r.T2_s)):
                fit_failures[arch] += 1
                continue
            rec["arch"].append(arch)
            rec["T1_true"].append(T1_t);   rec["T1_rec"].append(r.T1_s);        rec["T1_sig"].append(r.T1_sigma_s)
            rec["T2_true"].append(T2_t);   rec["T2_rec"].append(r.T2_s);        rec["T2_sig"].append(r.T2_sigma_s)
            rec["T2_ramsey_rec"].append(r.T2_ramsey_s); rec["T2_ramsey_sig"].append(r.T2_ramsey_sigma_s)
            rec["T2_echo_rec"].append(r.T2_echo_s);     rec["T2_echo_sig"].append(r.T2_echo_sigma_s)
            rec["dw_true"].append(abs(dw_t)); rec["dw_rec"].append(abs(r.delta_omega)); rec["dw_sig"].append(r.delta_omega_sigma)
            rec["eps_true"].append(eps_t); rec["eps_rec"].append(r.epsilon_sx); rec["eps_sig"].append(r.epsilon_sx_sigma)
            rec["t1_resid"].append(r.t1_residual)
            rec["ram_resid"].append(r.ramsey_residual)
            rec["gate_resid"].append(r.gate_residual)
            rec["echo_resid"].append(r.echo_residual)
            rec["t1_chi2"].append(r.t1_chi2_dof)
            rec["ramsey_chi2"].append(r.ramsey_chi2_dof)
            rec["gate_chi2"].append(r.gate_chi2_dof)
            rec["echo_chi2"].append(r.echo_chi2_dof)
            success += 1

    fit_failure_rate = {
        arch: fit_failures[arch] / fit_attempts[arch] if fit_attempts[arch] else 0.0
        for arch in ARCHITECTURES
    }
    return rec, fit_failure_rate


def save_csv(rec):
    path = DATA_DIR / "parity_results.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rec.keys()))
        w.writeheader()
        for i in range(len(rec["arch"])):
            w.writerow({k: rec[k][i] for k in rec})
    return path


def make_figure(rec):
    arch_arr = np.array(rec["arch"])
    T1_t = np.array(rec["T1_true"]); T1_r = np.array(rec["T1_rec"]); T1_s = np.array(rec["T1_sig"])
    T2_t = np.array(rec["T2_true"]); T2_r = np.array(rec["T2_rec"]); T2_s = np.array(rec["T2_sig"])
    dw_t = np.array(rec["dw_true"]); dw_r = np.array(rec["dw_rec"]); dw_s = np.array(rec["dw_sig"])
    eps_t = np.array(rec["eps_true"]); eps_r = np.array(rec["eps_rec"]); eps_s = np.array(rec["eps_sig"])

    fig = plt.figure(figsize=(7.16, 6.6))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.42,
                            left=0.11, right=0.97, top=0.93, bottom=0.10)

    panels = [
        (gs[0,0], T1_t,  T1_r,  T1_s,  r"True $T_1$ (s)",              r"Recovered $T_1$ (s)",              "(a)", True, None),
        (gs[0,1], T2_t,  T2_r,  T2_s,  r"True $T_2$ (s)",              r"Recovered $T_2$ (s)",              "(b)", True, None),
        (gs[1,0], dw_t,  dw_r,  dw_s,  r"True $|\Delta\omega|$ (rad/s)", r"Recovered $|\Delta\omega|$ (rad/s)", "(c)", True,
         "Sign resolved via $\\arctan_2(Y,X)$\nat $t_1=0.5\\,T_2^{\\rm prior}$.\nRange: $|\\Delta\\omega|\\leq 0.9\\pi/t_1$."),
        (gs[1,1], eps_t, eps_r, eps_s, r"True $\varepsilon_{sx}$",      r"Recovered $\varepsilon_{sx}$",     "(d)", True, None),
    ]

    for (spec, tv, rv, sv, xl, yl, lab, loglog, note) in panels:
        ax  = fig.add_subplot(spec)
        r2_all, rmse_all = compute_metrics(tv, rv)

        for arch in ARCHITECTURES:
            mask = arch_arr == arch
            ax.errorbar(tv[mask], rv[mask], yerr=sv[mask],
                        fmt=MARKERS[arch], color=COLORS[arch],
                        markersize=3.2, linewidth=0, elinewidth=0.55,
                        capsize=1.4, alpha=0.65, label=LABELS[arch])

        lo = min(tv.min(), rv.min()); hi = max(tv.max(), rv.max())
        if loglog and lo > 0:
            lo *= 0.82; hi *= 1.22
            ax.set_xscale("log"); ax.set_yscale("log")
        else:
            pad = (hi-lo)*0.09; lo -= pad; hi += pad

        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, zorder=5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(xl, fontsize=8.5); ax.set_ylabel(yl, fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, which="both" if loglog else "major", linewidth=0.35, alpha=0.45)
        ax.text(0.04, 0.96,
                f"$R^2={r2_all:.4f}$\nRMSE$={rmse_all:.2e}$",
                transform=ax.transAxes, fontsize=7.5, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#cccccc", linewidth=0.6))
        ax.text(-0.16, 1.05, lab, transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", va="top")
        if note:
            ax.text(0.04, 0.58, note, transform=ax.transAxes, fontsize=6.5,
                    color="#555555", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#fffbe6",
                              edgecolor="#cccc88", linewidth=0.5))

    handles, leg_labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=3,
               fontsize=7.5, frameon=True, bbox_to_anchor=(0.5, 0.005),
               handletextpad=0.4, columnspacing=0.8, framealpha=0.9, edgecolor="#cccccc")

    budget = shot_budget(ARCHITECTURES[0])
    fig.suptitle(
        f"Fig. 2 — Parameter Recovery Accuracy\n"
        f"(Ideal hardware, true-parameter priors, $N={N_INSTANCES}$/arch, "
        f"seed={SEED}, {budget:,} shots/qubit)",
        fontsize=8.5, y=0.98)

    out = FIGURES_DIR / "fig2_parity.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def sample_true_t1_t2_eps_realistic(arch_name: str, rng_obj):
    """Physically self-consistent (T1, T_phi -> T2) sampling: T2 = 1/(1/(2*T1) + 1/T_phi).
    Draws from the same realistic current-hardware ranges (TRUE_PARAM_RANGES)
    as Fig. 2, so Fig. 2 and Fig. 3 are directly comparable — only the delay
    grid (true-centred vs arch-default log-spaced) differs between them."""
    ranges = TRUE_PARAM_RANGES[arch_name]
    T1_lo, T1_hi = ranges["T1_s"]
    T2_lo, T2_hi = ranges["T2_s"]
    eps_lo, eps_hi = ranges["eps"]

    if T1_hi / T1_lo > 10.0:
        T1 = _log_uniform(rng_obj, T1_lo, T1_hi)
    else:
        T1 = rng_obj.uniform(T1_lo, T1_hi)

    T_phi = rng_obj.uniform(T2_lo, T2_hi)
    T2    = 1.0 / (1.0/(2.0*T1) + 1.0/T_phi)
    T2    = min(T2, 2.0*T1)

    eps = _log_uniform(rng_obj, eps_lo, eps_hi)
    return T1, T2, eps


def sample_true_params_realistic(arch_name: str, dw_max: float, rng_obj):
    T1, T2, eps = sample_true_t1_t2_eps_realistic(arch_name, rng_obj)
    dw = sample_dw(dw_max, rng_obj)
    return T1, T2, dw, eps


def simulate_realistic_inversion(arch_name, T1_true, T2_true, dw_true, eps_true,
                                 instance_id, rng_mismatch):
    arch     = inv.ARCH_DEFAULTS[arch_name]

    arch_idx   = ARCHITECTURES.index(arch_name)
    extra_seed = int(rng_mismatch.integers(1, 2**31 - 1))
    rng_t1, rng_ramsey, rng_gate, rng_echo, rng_param = _spawn_probe_rngs(
        SEED + 1, arch_idx, instance_id, extra_seed, n_streams=5)

    # Fig. 3 per-architecture real operating mode:
    #   superconducting / trapped_ion: live calibration is always available
    #     (IBM backend.properties() / IonQ characterization API) -> prior =
    #     truth +/- calibration uncertainty, Strategy A linear delays.
    #   neutral_atom: QuEra exposes no live calibration -> arch-default
    #     prior, Strategy B log-spaced delays. T2 may degrade honestly.
    if arch_name in ("superconducting", "trapped_ion"):
        T1_prior = float(np.clip(
            T1_true * (1.0 + rng_param.uniform(-0.15, 0.15)),
            arch["T1_min_s"], arch["T1_max_s"]))
        T2_prior = float(np.clip(
            T2_true * (1.0 + rng_param.uniform(-0.15, 0.15)),
            arch["T2_min_s"], min(2.0*T1_prior, arch["T1_max_s"])))
        profile = inv.BackendProfile.from_true_params(arch_name, T1_prior, T2_prior)
        # Same gate-rep centring rationale as Fig. 2 (simulate_inversion):
        # gate_rep_n is driven by constants["eps_typical"], untouched by
        # from_true_params. A live-calibration backend would also report a
        # live gate-error estimate, so centre it on eps_true here too.
        custom_arch = dict(arch)
        custom_arch["eps_typical"] = eps_true
        profile.custom_arch = custom_arch
    else:  # neutral_atom
        profile = inv.BackendProfile.from_architecture(arch_name)

    _, meta  = inv.build_probe_circuits(profile)

    t1_delays     = meta["t1_delays_s"]
    ramsey_delays = meta["ramsey_delays_s"]
    echo_delays   = meta["echo_delays_s"]
    Nv            = np.array(meta["gate_rep_N"], dtype=float)

    def c1(rng_obj, p, sh):
        n1 = int(rng_obj.binomial(sh, float(np.clip(p,0,1))))
        return {"0": sh-n1, "1": n1}
    def c0(rng_obj, p, sh):
        n0 = int(rng_obj.binomial(sh, float(np.clip(p,0,1))))
        return {"0": n0, "1": sh-n0}

    T2_echo_true = min(T2_true, 2.0*T1_true)

    if arch_name == "superconducting":
        p0_base = arch["p0_given_1"]
        p1_base = arch["p1_given_0"]
        spam_drift = float(np.clip(rng_t1.normal(0, 0.005), 0.0, 0.05))
        p0_eff = float(np.clip(p0_base + spam_drift, 0.0, 0.05))
        p1_eff = float(np.clip(p1_base + spam_drift, 0.0, 0.05))
        def spam(p):
            return p*(1.0 - p0_eff) + (1.0 - p)*p1_eff

        T1_local = float(np.clip(T1_true * (1.0 + rng_t1.normal(0, 0.08)),
                                 arch["T1_min_s"], arch["T1_max_s"]))
        T2_local = float(np.clip(T2_true * (1.0 + rng_ramsey.normal(0, 0.08)),
                                 arch["T2_min_s"], min(2.0*T1_local, arch["T1_max_s"])))

        t1_counts = []
        for d in t1_delays:
            p_ideal = float(inv.forward_t1(d, T1_local))
            t1_counts.append(c1(rng_t1, spam(p_ideal), SHOTS_T1))

        dephasing_rate = 1.0/T2_local - 1.0/(2.0*T1_local)
        if dephasing_rate > 0:
            T_phi_local = 1.0/dephasing_rate
            sigma_phi = 1.0/T_phi_local
        else:
            sigma_phi = 0.0

        ramsey_counts = []
        for t in ramsey_delays:
            decay = np.exp(-t/T2_local) * np.exp(-0.5*(t*sigma_phi)**2)
            px = 0.5*(1.0 - decay*np.cos(dw_true*t))
            py = 0.5*(1.0 - decay*np.sin(dw_true*t))
            ramsey_counts.append(c1(rng_ramsey, spam(px), SHOTS_RAMSEY))
            ramsey_counts.append(c1(rng_ramsey, spam(py), SHOTS_RAMSEY))

        p0_gate = inv.forward_gate(Nv, eps_true)
        gate_counts = [c0(rng_gate, spam(p), SHOTS_GATE) for p in p0_gate]

        T2_echo_local = min(T2_local, 2.0*T1_local)
        echo_counts = [c1(rng_echo, spam(float(inv.forward_echo(t, T2_echo_local))), SHOTS_ECHO)
                       for t in echo_delays]

    elif arch_name == "trapped_ion":
        t1_counts = []
        for d in t1_delays:
            T1_local = T1_true * np.exp(-d/(50.0*T1_true))
            p_ideal  = float(inv.forward_t1(d, T1_local))
            t1_counts.append(c1(rng_t1, p_ideal, SHOTS_T1))

        ramsey_counts = []
        for t in ramsey_delays:
            dw_local = dw_true + rng_ramsey.normal(0, 0.05*abs(dw_true))
            px, py   = inv.forward_ramsey_xy(t, T2_true, dw_local)
            px_eff   = px * (1.0 + 0.03*np.cos(2*np.pi*t/T2_true))
            py_eff   = py * (1.0 + 0.03*np.sin(2*np.pi*t/T2_true))
            ramsey_counts.append(c1(rng_ramsey, px_eff, SHOTS_RAMSEY))
            ramsey_counts.append(c1(rng_ramsey, py_eff, SHOTS_RAMSEY))

        p0_gate = inv.forward_gate(Nv, eps_true)
        gate_counts = [c0(rng_gate, p, SHOTS_GATE) for p in p0_gate]

        echo_counts = [c1(rng_echo, float(inv.forward_echo(t, T2_echo_true)), SHOTS_ECHO)
                       for t in echo_delays]

    else:  # neutral_atom
        t1_counts = [c1(rng_t1, float(inv.forward_t1(d, T1_true)), SHOTS_T1) for d in t1_delays]

        ramsey_counts = []
        for t in ramsey_delays:
            dw_local = dw_true + rng_ramsey.normal(0, 0.08*abs(dw_true))
            px, py   = inv.forward_ramsey_xy(t, T2_true, dw_local)
            ramsey_counts.append(c1(rng_ramsey, px, SHOTS_RAMSEY))
            ramsey_counts.append(c1(rng_ramsey, py, SHOTS_RAMSEY))

        eps_max = arch["eps_max"]
        gate_counts = []
        for N in Nv:
            eps_local = eps_true * (1.0 + rng_gate.normal(0, 0.10))
            eps_local = float(np.clip(eps_local, 0.0, eps_max))
            p0 = float(inv.forward_gate(np.array([N]), eps_local)[0])
            gate_counts.append(c0(rng_gate, p0, SHOTS_GATE))

        echo_counts = [c1(rng_echo, float(inv.forward_echo(t, T2_echo_true)), SHOTS_ECHO)
                       for t in echo_delays]

    counts_list = t1_counts + ramsey_counts + gate_counts + echo_counts

    return inv.lindblad_inversion(
        counts_list, meta, profile,
        shots_t1=SHOTS_T1, shots_ramsey=SHOTS_RAMSEY,
        shots_gate=SHOTS_GATE, shots_echo=SHOTS_ECHO,
        qubit_id=0, timestamp=datetime.now(timezone.utc).isoformat())


def run_realistic_mismatch_experiment():
    rng_mismatch = np.random.default_rng(SEED + 1)

    rec = {k: [] for k in ["arch",
        "T1_true","T1_rec","T1_sig",
        "T2_true","T2_rec","T2_sig",
        "T2_ramsey_rec","T2_ramsey_sig",
        "T2_echo_rec","T2_echo_sig",
        "dw_true","dw_rec","dw_sig",
        "eps_true","eps_rec","eps_sig",
        "t1_chi2","ramsey_chi2","gate_chi2","echo_chi2"]}

    fit_failures  = {arch: 0 for arch in ARCHITECTURES}
    fit_attempts  = {arch: 0 for arch in ARCHITECTURES}

    for arch in ARCHITECTURES:
        arch_default_dw_max = inv.BackendProfile.from_architecture(arch).dw_max_rad_s
        success = 0
        while success < N_INSTANCES:
            T1_t, T2_t, eps_t = sample_true_t1_t2_eps_realistic(arch, rng_mismatch)
            # dw_max must reflect the delay grid actually used for this
            # instance's architecture — same aliasing hazard as Fig. 2:
            # superconducting/trapped_ion now use Strategy A (live-cal
            # prior, close to truth) so dw_max should track T2_t, not a
            # fixed arch-default T2 that may differ by orders of
            # magnitude. neutral_atom stays on the fixed arch-default
            # Strategy B grid, so its dw_max stays fixed too.
            if arch in ("superconducting", "trapped_ion"):
                dw_max = inv.BackendProfile.from_true_params(
                    arch, T1_t, T2_t).dw_max_rad_s
            else:
                dw_max = arch_default_dw_max
            dw_t = sample_dw(dw_max, rng_mismatch)
            fit_attempts[arch] += 1
            try:
                r = simulate_realistic_inversion(arch, T1_t, T2_t, dw_t, eps_t,
                                                 fit_attempts[arch], rng_mismatch)
            except (RuntimeError, ValueError):
                fit_failures[arch] += 1
                continue
            if not (np.isfinite(r.T1_s) and np.isfinite(r.T2_s)):
                fit_failures[arch] += 1
                continue
            rec["arch"].append(arch)
            rec["T1_true"].append(T1_t);   rec["T1_rec"].append(r.T1_s);        rec["T1_sig"].append(r.T1_sigma_s)
            rec["T2_true"].append(T2_t);   rec["T2_rec"].append(r.T2_s);        rec["T2_sig"].append(r.T2_sigma_s)
            rec["T2_ramsey_rec"].append(r.T2_ramsey_s); rec["T2_ramsey_sig"].append(r.T2_ramsey_sigma_s)
            rec["T2_echo_rec"].append(r.T2_echo_s);     rec["T2_echo_sig"].append(r.T2_echo_sigma_s)
            rec["dw_true"].append(abs(dw_t)); rec["dw_rec"].append(abs(r.delta_omega)); rec["dw_sig"].append(r.delta_omega_sigma)
            rec["eps_true"].append(eps_t); rec["eps_rec"].append(r.epsilon_sx); rec["eps_sig"].append(r.epsilon_sx_sigma)
            rec["t1_chi2"].append(r.t1_chi2_dof)
            rec["ramsey_chi2"].append(r.ramsey_chi2_dof)
            rec["gate_chi2"].append(r.gate_chi2_dof)
            rec["echo_chi2"].append(r.echo_chi2_dof)
            success += 1

    fit_failure_rate = {
        arch: fit_failures[arch] / fit_attempts[arch] if fit_attempts[arch] else 0.0
        for arch in ARCHITECTURES
    }
    return rec, fit_failure_rate


def save_mismatch_csv(rec):
    path = DATA_DIR / "mismatch_results.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rec.keys()))
        w.writeheader()
        for i in range(len(rec["T1_true"])):
            w.writerow({k: rec[k][i] for k in rec})
    return path


def make_mismatch_figure(rec):
    arch_arr = np.array(rec["arch"])
    T1_t = np.array(rec["T1_true"]); T1_r = np.array(rec["T1_rec"]); T1_s = np.array(rec["T1_sig"])
    T2_t = np.array(rec["T2_true"]); T2_r = np.array(rec["T2_rec"]); T2_s = np.array(rec["T2_sig"])
    dw_t = np.array(rec["dw_true"]); dw_r = np.array(rec["dw_rec"]); dw_s = np.array(rec["dw_sig"])
    eps_t = np.array(rec["eps_true"]); eps_r = np.array(rec["eps_rec"]); eps_s = np.array(rec["eps_sig"])

    fig = plt.figure(figsize=(7.16, 6.6))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.42,
                            left=0.11, right=0.97, top=0.93, bottom=0.10)

    panels = [
        (gs[0,0], T1_t,  T1_r,  T1_s,  r"True $T_1$ (s)",              r"Recovered $T_1$ (s)",              "(a)", None),
        (gs[0,1], T2_t,  T2_r,  T2_s,  r"True $T_2$ (s)",              r"Recovered $T_2$ (s)",              "(b)", None),
        (gs[1,0], dw_t,  dw_r,  dw_s,  r"True $|\Delta\omega|$ (rad/s)", r"Recovered $|\Delta\omega|$ (rad/s)", "(c)",
         "Sign resolved via $\\arctan_2(Y,X)$\nat $t_1=0.5\\,T_2^{\\rm prior}$.\nRange: $|\\Delta\\omega|\\leq 0.9\\pi/t_1$."),
        (gs[1,1], eps_t, eps_r, eps_s, r"True $\varepsilon_{sx}$",      r"Recovered $\varepsilon_{sx}$",     "(d)", None),
    ]

    for (spec, tv, rv, sv, xl, yl, lab, note) in panels:
        ax  = fig.add_subplot(spec)
        r2_all, rmse_all = compute_metrics(tv, rv)

        for arch in ARCHITECTURES:
            mask = arch_arr == arch
            ax.errorbar(tv[mask], rv[mask], yerr=sv[mask],
                        fmt=MARKERS[arch], color=COLORS[arch],
                        markersize=3.2, linewidth=0, elinewidth=0.55,
                        capsize=1.4, alpha=0.65, label=LABELS[arch])

        lo = min(tv.min(), rv.min()); hi = max(tv.max(), rv.max())
        if lo > 0:
            lo *= 0.82; hi *= 1.22
            ax.set_xscale("log"); ax.set_yscale("log")
        else:
            pad = (hi-lo)*0.09; lo -= pad; hi += pad

        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, zorder=5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(xl, fontsize=8.5); ax.set_ylabel(yl, fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, which="both", linewidth=0.35, alpha=0.45)
        ax.text(0.04, 0.96,
                f"$R^2={r2_all:.4f}$\nRMSE$={rmse_all:.2e}$",
                transform=ax.transAxes, fontsize=7.5, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#cccccc", linewidth=0.6))
        ax.text(-0.16, 1.05, lab, transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", va="top")
        if note:
            ax.text(0.04, 0.58, note, transform=ax.transAxes, fontsize=6.5,
                    color="#555555", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#fffbe6",
                              edgecolor="#cccc88", linewidth=0.5))

    handles, leg_labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=3,
               fontsize=7.5, frameon=True, bbox_to_anchor=(0.5, 0.005),
               handletextpad=0.4, columnspacing=0.8, framealpha=0.9, edgecolor="#cccccc")

    budget = shot_budget(ARCHITECTURES[0])
    fig.suptitle(
        f"Fig. 3 — Realistic Hardware Performance\n"
        f"(Per-arch operating mode: SC/TI live cal ±15%, NA arch-default, "
        f"$N={N_INSTANCES}$/arch, seed={SEED+1}, {budget:,} shots/qubit)",
        fontsize=8.5, y=0.98)

    out = FIGURES_DIR / "fig3_mismatch.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"[{ts}] Fig. 2 — Parity Experiment")
    print(f"  Protocol : 3-time XY Ramsey (Shnaiderov+) + 3-point T1 + adaptive gate-rep N")
    print(f"  Shots    : T1={SHOTS_T1}/circuit  Ramsey={SHOTS_RAMSEY}/circuit  Gate={SHOTS_GATE}/circuit")
    print(f"  Budget   : {shot_budget('superconducting'):,} shots/qubit (all architectures)")
    print(f"  Sampling (Fig. 2 & 3):")
    print(f"    superconducting: T1=[80,400]µs  T2=[40,200]µs  ε=[1e-4,2e-3]")
    print(f"    trapped_ion:     T1=[100,10000]s  T2=[0.1,3]s  ε=[1e-4,2e-3]")
    print(f"    neutral_atom:    T1=[1,100]s  T2=[0.1,3]s  ε=[1e-3,1e-2]")
    print(f"  Fig. 2 uses true-parameter priors (ideal, Strategy A delays)")
    print(f"  Fig. 3 operating mode per architecture:")
    print(f"    superconducting : live calibration prior ±15% + TLS/SPAM noise")
    print(f"    trapped_ion     : live calibration prior ±15% + B-field/motion noise")
    print(f"    neutral_atom    : arch-default prior (no live cal) + laser/tweezer noise")
    print()
    for arch in ARCHITECTURES:
        p_ideal = inv.BackendProfile.from_true_params(
            arch, inv.ARCH_DEFAULTS[arch]["T1_s"], inv.ARCH_DEFAULTS[arch]["T2_s"])
        p_real  = inv.BackendProfile.from_architecture(arch)
        print(f"  {arch:16s}: [ideal] T1 delays={[f'{d:.3g}s' for d in p_ideal.t1_delays_s]}  "
              f"Ramsey={[f'{d:.3g}s' for d in p_ideal.ramsey_delays_s]}")
        print(f"  {'':16s}  [real ] T1 delays={[f'{d:.3g}s' for d in p_real.t1_delays_s]}  "
              f"Ramsey={[f'{d:.3g}s' for d in p_real.ramsey_delays_s]}")
    print()

    rec, fit_failure_rate = run_experiment(use_true_prior=True)
    csv_path = save_csv(rec)
    fig_path = make_figure(rec)

    print(f"  Results (n={N_INSTANCES}/arch):")
    arch_arr = np.array(rec["arch"])
    all_pass = True
    for arch in ARCHITECTURES:
        mask = arch_arr == arch
        c = inv.ARCH_DEFAULTS[arch]
        s = c["time_scale"]; u = c["display_unit"]
        T1_r2, T1_rmse = compute_metrics(np.array(rec["T1_true"])[mask]*s, np.array(rec["T1_rec"])[mask]*s)
        T2_r2, T2_rmse = compute_metrics(np.array(rec["T2_true"])[mask]*s, np.array(rec["T2_rec"])[mask]*s)
        dw_r2, dw_rmse = compute_metrics(np.array(rec["dw_true"])[mask],   np.array(rec["dw_rec"])[mask])
        ep_r2, ep_rmse = compute_metrics(np.array(rec["eps_true"])[mask],  np.array(rec["eps_rec"])[mask])
        vals = [T1_r2, T2_r2, dw_r2, ep_r2]
        ok   = "✓" if all(v >= 0.94 for v in vals) else "✗"
        all_pass &= all(v >= 0.94 for v in vals)
        print(f"\n    {ok} {arch} (shots={shot_budget(arch)}, fit-failure rate={fit_failure_rate[arch]:.2%})")
        print(f"      T1 : R²={T1_r2:.4f}  RMSE={T1_rmse:.3e} {u}")
        print(f"      T2 : R²={T2_r2:.4f}  RMSE={T2_rmse:.3e} {u}")
        print(f"      Δω : R²={dw_r2:.4f}  RMSE={dw_rmse:.3e} rad/s")
        print(f"      ε  : R²={ep_r2:.4f}  RMSE={ep_rmse:.3e}")
        T2_ram_r2, _  = compute_metrics(np.array(rec["T2_true"])[mask]*s, np.array(rec["T2_ramsey_rec"])[mask]*s)
        T2_echo_r2, _ = compute_metrics(np.array(rec["T2_true"])[mask]*s, np.array(rec["T2_echo_rec"])[mask]*s)
        print(f"      T2 (combined)={T2_r2:.4f}  T2 (ramsey-only)={T2_ram_r2:.4f}  T2 (echo-only)={T2_echo_r2:.4f}")
        t1_chi2_arr  = np.array(rec["t1_chi2"])[mask]
        ram_chi2_arr = np.array(rec["ramsey_chi2"])[mask]
        gate_chi2_arr = np.array(rec["gate_chi2"])[mask]
        echo_chi2_arr = np.array(rec["echo_chi2"])[mask]
        print(f"      χ²/dof: T1={t1_chi2_arr.mean():.2f}±{t1_chi2_arr.std():.2f}  "
              f"Ramsey={ram_chi2_arr.mean():.2f}±{ram_chi2_arr.std():.2f}  "
              f"Gate={gate_chi2_arr.mean():.2f}±{gate_chi2_arr.std():.2f}  "
              f"Echo={echo_chi2_arr.mean():.2f}±{echo_chi2_arr.std():.2f}")

    print(f"\n  All R²≥0.94: {'YES ✓' if all_pass else 'NO ✗'}")
    print(f"\n  Data  : {csv_path}")
    print(f"  Figure: {fig_path}")

    print()
    print("Fig. 3 — Per-arch real operating mode (SC/TI live cal +/-15%, NA arch-default)")
    mismatch_rec, mismatch_fail_rate = run_realistic_mismatch_experiment()
    mismatch_csv = save_mismatch_csv(mismatch_rec)
    mismatch_fig = make_mismatch_figure(mismatch_rec)

    mismatch_arch_arr = np.array(mismatch_rec["arch"])
    for arch in ARCHITECTURES:
        mask = mismatch_arch_arr == arch
        c = inv.ARCH_DEFAULTS[arch]
        s = c["time_scale"]; u = c["display_unit"]
        T1_r2, T1_rmse = compute_metrics(np.array(mismatch_rec["T1_true"])[mask]*s, np.array(mismatch_rec["T1_rec"])[mask]*s)
        T2_r2, T2_rmse = compute_metrics(np.array(mismatch_rec["T2_true"])[mask]*s, np.array(mismatch_rec["T2_rec"])[mask]*s)
        dw_r2, dw_rmse = compute_metrics(np.array(mismatch_rec["dw_true"])[mask],   np.array(mismatch_rec["dw_rec"])[mask])
        ep_r2, ep_rmse = compute_metrics(np.array(mismatch_rec["eps_true"])[mask],  np.array(mismatch_rec["eps_rec"])[mask])
        vals = [T1_r2, T2_r2, dw_r2, ep_r2]
        ok   = "✓" if all(v >= 0.90 for v in vals) else "✗"
        print(f"\n    {ok} {arch} (fit-failure rate={mismatch_fail_rate[arch]:.2%})")
        print(f"      T1 : R²={T1_r2:.4f}  RMSE={T1_rmse:.3e} {u}")
        print(f"      T2 : R²={T2_r2:.4f}  RMSE={T2_rmse:.3e} {u}")
        print(f"      Δω : R²={dw_r2:.4f}  RMSE={dw_rmse:.3e} rad/s")
        print(f"      ε  : R²={ep_r2:.4f}  RMSE={ep_rmse:.3e}")
        T2_ram_r2, _  = compute_metrics(np.array(mismatch_rec["T2_true"])[mask]*s, np.array(mismatch_rec["T2_ramsey_rec"])[mask]*s)
        T2_echo_r2, _ = compute_metrics(np.array(mismatch_rec["T2_true"])[mask]*s, np.array(mismatch_rec["T2_echo_rec"])[mask]*s)
        print(f"      T2 (combined)={T2_r2:.4f}  T2 (ramsey-only)={T2_ram_r2:.4f}  T2 (echo-only)={T2_echo_r2:.4f}")
        t1_chi2_arr   = np.array(mismatch_rec["t1_chi2"])[mask]
        ram_chi2_arr  = np.array(mismatch_rec["ramsey_chi2"])[mask]
        gate_chi2_arr = np.array(mismatch_rec["gate_chi2"])[mask]
        echo_chi2_arr = np.array(mismatch_rec["echo_chi2"])[mask]
        print(f"      χ²/dof: T1={t1_chi2_arr.mean():.2f}±{t1_chi2_arr.std():.2f}  "
              f"Ramsey={ram_chi2_arr.mean():.2f}±{ram_chi2_arr.std():.2f}  "
              f"Gate={gate_chi2_arr.mean():.2f}±{gate_chi2_arr.std():.2f}  "
              f"Echo={echo_chi2_arr.mean():.2f}±{echo_chi2_arr.std():.2f}")

    print(f"\n  Mismatch data  : {mismatch_csv}")
    print(f"  Mismatch figure: {mismatch_fig}")