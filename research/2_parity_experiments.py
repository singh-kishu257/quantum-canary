import importlib.util, pathlib, sys, csv, os
import multiprocessing as mp
import numpy as np
from datetime import datetime, timezone

_spec = importlib.util.spec_from_file_location(
    "inversion", pathlib.Path(__file__).parent / "1_inversion.py")
inv = importlib.util.module_from_spec(_spec)
sys.modules["inversion"] = inv
_spec.loader.exec_module(inv)

SCRIPT_DIR  = pathlib.Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

N_INSTANCES   = 300
SHOTS_T1      = 300
SHOTS_RAMSEY  = 1000
SHOTS_GATE    = 500
SHOTS_ECHO    = 500
SEED          = 43

ARCHITECTURES = ["superconducting", "trapped_ion", "neutral_atom"]
_arch_env = os.environ.get("ARCH", "").strip()
if _arch_env in ARCHITECTURES:
    ARCHITECTURES = [_arch_env]

_n_env = os.environ.get("N_INSTANCES", "").strip()
if _n_env.isdigit():
    N_INSTANCES = int(_n_env)

rng = np.random.default_rng(SEED)


def shot_budget(arch_name: str) -> int:
    return 3*SHOTS_T1 + 6*SHOTS_RAMSEY + 3*SHOTS_GATE + 3*SHOTS_ECHO


TRUE_PARAM_RANGES = {
    "superconducting": {
        "T1_s": (80e-6, 400e-6),
        "T2_s": (40e-6, 200e-6),
        "eps":  (1e-4, 2e-3),
    },
    "trapped_ion": {
        "T1_s": (100.0, 10000.0),
        "T2_s": (0.1, 3.0),
        "eps":  (1e-4, 2e-3),
    },
    "neutral_atom": {
        "T1_s": (1.0, 100.0),
        "T2_s": (0.3, 3.0),
        "eps":  (1e-3, 1e-2),
    },
}

REALISTIC_NOISE = {
    "superconducting": {
        "sigma_T1_lognormal":  0.15,
        "T2_reduction_max":    0.12,
        "sigma_coherent_rad":  0.010,
        "sigma_SPAM_frac":     0.05,
        "sigma_dw_rad_s":      0.0,
    },
    "trapped_ion": {
        "sigma_T1_lognormal":  0.00,
        "T2_reduction_max":    0.10,
        "sigma_coherent_rad":  0.005,
        "sigma_SPAM_frac":     0.03,
        "sigma_dw_rad_s":      0.0,
    },
    "neutral_atom": {
        "sigma_T1_lognormal":  0.03,
        "T2_reduction_max":    0.15,
        "sigma_coherent_rad":  0.015,
        "sigma_SPAM_frac":     0.08,
        "sigma_dw_rad_s":      0.0,
    },
}


def _log_uniform(r, lo: float, hi: float) -> float:
    return float(10 ** r.uniform(np.log10(lo), np.log10(hi)))


def sample_dw(dw_max: float, rng_obj=None):
    r = rng_obj if rng_obj is not None else rng
    return r.choice([-1,1]) * r.uniform(0.2*dw_max, dw_max)


def _spawn_probe_rngs(base_seed: int, *keys, n_streams: int = 4):
    """Independent, reproducible child RNGs for the t1/ramsey/gate/echo probe
    families (n_streams=4), plus an optional 5th stream (n_streams=5) for
    the ±15% live-calibration prior jitter used in Fig. 3's live-cal arms.
    The extra stream is appended so the first 4 streams are unaffected."""
    ss = np.random.SeedSequence([base_seed, *keys])
    streams = ss.spawn(n_streams)
    rngs = [np.random.default_rng(s) for s in streams]
    return tuple(rngs)


def simulate_inversion(arch_name, T1_true, T2_true, dw_true, eps_true,
                       instance_id, use_true_prior=True):
    if use_true_prior:
        profile = inv.BackendProfile.from_true_params(arch_name, T1_true, T2_true)
        custom_arch = dict(inv.ARCH_DEFAULTS[arch_name])
        custom_arch["eps_typical"] = eps_true
        profile.custom_arch = custom_arch
    else:
        profile = inv.BackendProfile.from_architecture(arch_name)
    arch = inv.ARCH_DEFAULTS[arch_name]
    circuits, meta = inv.build_probe_circuits(profile)
    counts_list = inv.run_probe_circuits_aer(
        circuits, meta,
        T1_true, T2_true, eps_true,
        arch["p0_given_1"], arch["p1_given_0"],
        profile.dt_ns,
        SHOTS_T1, SHOTS_RAMSEY, SHOTS_GATE, SHOTS_ECHO,
        dw_s=dw_true,
    )
    return inv.lindblad_inversion(
        counts_list, meta, profile,
        shots_t1=SHOTS_T1, shots_ramsey=SHOTS_RAMSEY,
        shots_gate=SHOTS_GATE, shots_echo=SHOTS_ECHO,
        qubit_id=0,
        timestamp=datetime.now(timezone.utc).isoformat())


def compute_metrics(true_vals, rec_vals):
    t = np.array(true_vals); r = np.array(rec_vals)
    ss_res = np.sum((r-t)**2); ss_tot = np.sum((t-np.mean(t))**2)
    r2   = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((r-t)**2)))
    return r2, rmse


def _worker_fig2(args):
    arch, T1_t, T2_t, dw_t, eps_t, instance_id, use_true_prior = args
    try:
        r = simulate_inversion(arch, T1_t, T2_t, dw_t, eps_t, instance_id,
                               use_true_prior=use_true_prior)
    except (RuntimeError, ValueError):
        return (arch, T1_t, T2_t, dw_t, eps_t, None)
    if not (np.isfinite(r.T1_s) and np.isfinite(r.T2_s)):
        return (arch, T1_t, T2_t, dw_t, eps_t, None)
    return (arch, T1_t, T2_t, dw_t, eps_t, r)


def run_experiment(use_true_prior: bool = True, n_workers: int = None):
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
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
        attempt_counter = 0
        while success < N_INSTANCES:
            n_needed = N_INSTANCES - success
            batch_size = max(n_workers, int(n_needed * 1.05))
            batch_args = []
            for _ in range(batch_size):
                T1_t, T2_t, eps_t = sample_true_t1_t2_eps_realistic(arch, rng)
                if use_true_prior:
                    dw_max = inv.BackendProfile.from_true_params(
                        arch, T1_t, T2_t).dw_max_rad_s
                else:
                    dw_max = arch_default_dw_max
                dw_t = sample_dw(dw_max)
                attempt_counter += 1
                batch_args.append((arch, T1_t, T2_t, dw_t, eps_t,
                                   attempt_counter, use_true_prior))
            fit_attempts[arch] += len(batch_args)
            with mp.Pool(n_workers) as pool:
                results = pool.map(_worker_fig2, batch_args)
            for _, T1_t, T2_t, dw_t, eps_t, r in results:
                if r is None:
                    fit_failures[arch] += 1
                    continue
                if success >= N_INSTANCES:
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
            print(f"  [{arch}] {success}/{N_INSTANCES} complete", flush=True)
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
                                 instance_id, extra_seed):
    """Fig. 3 simulation: genuine model-mismatch experiment.
    The AerSimulator runs with EFFECTIVE hardware parameters that include
    unmodeled physical noise (TLS drift, 1/f dephasing, coherent over-
    rotation, quasi-static detuning offsets, SPAM calibration drift).
    Canary's inversion is blind to all of these — it uses a ±15% prior
    jitter around T1_true/T2_true and the device's nominal calibrated SPAM.
    The mismatch has two independent layers:
      Layer 1 — prior error: Canary's delay grid is centred on T1_prior ≠
                T1_eff (already present in the original code).
      Layer 2 — physics mismatch (NEW): the simulator runs physics outside
                Canary's Lindblad forward models. This breaks the Lindblad-
                vs-Lindblad tautology and tests genuine robustness.
    Returns (result, T1_eff, T2_eff, abs(dw_eff), eps_eff) so that R² in
    Fig. 3 is computed against the effective parameters — i.e., what the
    hardware actually had during this measurement window — not T1_true,
    which the hardware drifted away from.
    """
    arch     = inv.ARCH_DEFAULTS[arch_name]
    arch_idx = ARCHITECTURES.index(arch_name)
    noise_cfg = REALISTIC_NOISE[arch_name]
    rng_param, rng_noise = _spawn_probe_rngs(
        SEED + 1, arch_idx, instance_id, extra_seed, n_streams=2)
    if noise_cfg["sigma_T1_lognormal"] > 0.0:
        T1_eff = float(np.clip(
            T1_true * np.exp(rng_noise.normal(0.0, noise_cfg["sigma_T1_lognormal"])),
            arch["T1_min_s"], arch["T1_max_s"]))
    else:
        T1_eff = T1_true
    T2_reduction = rng_noise.uniform(0.0, noise_cfg["T2_reduction_max"])
    T2_eff = float(np.clip(
        T2_true * (1.0 - T2_reduction),
        arch["T2_min_s"], min(2.0 * T1_eff, arch["T1_max_s"])))
    delta_coh  = rng_noise.normal(0.0, noise_cfg["sigma_coherent_rad"])
    eps_eff    = float(np.clip(
        eps_true + delta_coh ** 2 / 4.0,
        0.0, arch["eps_max"]))
    dw_eff = dw_true + rng_noise.normal(0.0, noise_cfg["sigma_dw_rad_s"])
    p0g1_nom = arch["p0_given_1"]
    p1g0_nom = arch["p1_given_0"]
    p0g1_eff = float(np.clip(
        p0g1_nom * (1.0 + rng_noise.normal(0.0, noise_cfg["sigma_SPAM_frac"])),
        0.0, 0.30))
    p1g0_eff = float(np.clip(
        p1g0_nom * (1.0 + rng_noise.normal(0.0, noise_cfg["sigma_SPAM_frac"])),
        0.0, 0.30))
    if arch_name in ("superconducting", "trapped_ion"):
        T1_prior = float(np.clip(
            T1_true * (1.0 + rng_param.uniform(-0.15, 0.15)),
            arch["T1_min_s"], arch["T1_max_s"]))
        T2_prior = float(np.clip(
            T2_true * (1.0 + rng_param.uniform(-0.15, 0.15)),
            arch["T2_min_s"], min(2.0 * T1_prior, arch["T1_max_s"])))
        profile = inv.BackendProfile.from_true_params(arch_name, T1_prior, T2_prior)
        eps_prior = float(np.clip(
            eps_true * (1.0 + rng_param.uniform(-0.15, 0.15)),
            arch["eps_typical"] * 0.01, arch["eps_max"]))
        custom_arch = dict(arch)
        custom_arch["eps_typical"] = eps_prior
        profile.custom_arch = custom_arch
    else:
        profile = inv.BackendProfile.from_architecture(arch_name)
    circuits, meta = inv.build_probe_circuits(profile)
    counts_list = inv.run_probe_circuits_aer(
        circuits, meta,
        T1_eff, T2_eff, eps_eff,
        p0g1_eff, p1g0_eff,
        profile.dt_ns,
        SHOTS_T1, SHOTS_RAMSEY, SHOTS_GATE, SHOTS_ECHO,
        dw_s=dw_eff,
    )
    result = inv.lindblad_inversion(
        counts_list, meta, profile,
        shots_t1=SHOTS_T1, shots_ramsey=SHOTS_RAMSEY,
        shots_gate=SHOTS_GATE, shots_echo=SHOTS_ECHO,
        qubit_id=0,
        timestamp=datetime.now(timezone.utc).isoformat())
    return result, T1_eff, T2_eff, abs(dw_eff), eps_eff


def _worker_fig3(args):
    arch, T1_t, T2_t, dw_t, eps_t, instance_id, extra_seed = args
    try:
        r, T1_eff, T2_eff, dw_eff, eps_eff = simulate_realistic_inversion(
            arch, T1_t, T2_t, dw_t, eps_t, instance_id, extra_seed)
    except (RuntimeError, ValueError):
        return (arch, T1_t, T2_t, abs(dw_t), eps_t, None, None, None, None, None)
    if not (np.isfinite(r.T1_s) and np.isfinite(r.T2_s)):
        return (arch, T1_t, T2_t, abs(dw_t), eps_t, None, None, None, None, None)
    return (arch, T1_t, T2_t, abs(dw_t), eps_t, r, T1_eff, T2_eff, dw_eff, eps_eff)


def run_realistic_mismatch_experiment(n_workers: int = None):
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
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
        attempt_counter = 0
        while success < N_INSTANCES:
            n_needed = N_INSTANCES - success
            batch_size = max(n_workers, int(n_needed * 1.05))
            batch_args = []
            for _ in range(batch_size):
                T1_t, T2_t, eps_t = sample_true_t1_t2_eps_realistic(arch, rng_mismatch)
                if arch in ("superconducting", "trapped_ion"):
                    dw_max = inv.BackendProfile.from_true_params(
                        arch, T1_t, T2_t).dw_max_rad_s
                else:
                    dw_max = arch_default_dw_max
                dw_t = sample_dw(dw_max, rng_mismatch)
                extra_seed = int(rng_mismatch.integers(1, 2**31 - 1))
                attempt_counter += 1
                batch_args.append((arch, T1_t, T2_t, dw_t, eps_t,
                                   attempt_counter, extra_seed))
            fit_attempts[arch] += len(batch_args)
            with mp.Pool(n_workers) as pool:
                results = pool.map(_worker_fig3, batch_args)
            for _, T1_t, T2_t, dw_t, eps_t, r, T1_eff, T2_eff, dw_eff, eps_eff in results:
                if r is None:
                    fit_failures[arch] += 1
                    continue
                if success >= N_INSTANCES:
                    continue
                rec["arch"].append(arch)
                rec["T1_true"].append(T1_eff);  rec["T1_rec"].append(r.T1_s);       rec["T1_sig"].append(r.T1_sigma_s)
                rec["T2_true"].append(T2_eff);  rec["T2_rec"].append(r.T2_s);       rec["T2_sig"].append(r.T2_sigma_s)
                rec["T2_ramsey_rec"].append(r.T2_ramsey_s); rec["T2_ramsey_sig"].append(r.T2_ramsey_sigma_s)
                rec["T2_echo_rec"].append(r.T2_echo_s);     rec["T2_echo_sig"].append(r.T2_echo_sigma_s)
                rec["dw_true"].append(dw_eff);  rec["dw_rec"].append(abs(r.delta_omega)); rec["dw_sig"].append(r.delta_omega_sigma)
                rec["eps_true"].append(eps_eff); rec["eps_rec"].append(r.epsilon_sx); rec["eps_sig"].append(r.epsilon_sx_sigma)
                rec["t1_chi2"].append(r.t1_chi2_dof)
                rec["ramsey_chi2"].append(r.ramsey_chi2_dof)
                rec["gate_chi2"].append(r.gate_chi2_dof)
                rec["echo_chi2"].append(r.echo_chi2_dof)
                success += 1
            print(f"  [{arch}] {success}/{N_INSTANCES} complete", flush=True)
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


if __name__ == "__main__":
    N_WORKERS = int(os.environ.get("N_WORKERS", max(1, mp.cpu_count() - 1)))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"[{ts}] Fig. 2 — Parity Experiment")
    print(f"  Protocol : 3-time XY Ramsey (Shnaiderov+) + 3-point T1 + adaptive gate-rep N")
    print(f"  Shots    : T1={SHOTS_T1}/circuit  Ramsey={SHOTS_RAMSEY}/circuit  Gate={SHOTS_GATE}/circuit")
    print(f"  Budget   : {shot_budget('superconducting'):,} shots/qubit (all architectures)")
    print(f"  Sampling (Fig. 2 & 3, unified): T1 ~ TRUE_PARAM_RANGES;  "
          f"T2 = 1/(1/(2*T1) + 1/T_phi), T_phi ~ TRUE_PARAM_RANGES T2 range")
    print(f"    superconducting: T1=[80,400]µs  T_phi=[40,200]µs  ε=[1e-4,2e-3]")
    print(f"    trapped_ion:     T1=[100,10000]s  T_phi=[0.1,3]s  ε=[1e-4,2e-3]")
    print(f"    neutral_atom:    T1=[1,100]s  T_phi=[0.3,3]s  ε=[1e-3,1e-2]")
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
        print(f"  {arch:16s}: [ideal] T1 delays={[f'{d:.3g}s' for d in p_ideal.t1_delays_s]}   "
              f"Ramsey={[f'{d:.3g}s' for d in p_ideal.ramsey_delays_s]}")
        print(f"  {'':16s}  [real ] T1 delays={[f'{d:.3g}s' for d in p_real.t1_delays_s]}   "
              f"Ramsey={[f'{d:.3g}s' for d in p_real.ramsey_delays_s]}")
    print()
    rec, fit_failure_rate = run_experiment(use_true_prior=True, n_workers=N_WORKERS)
    csv_path = save_csv(rec)
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
    print()
    print("Fig. 3 — Per-arch real operating mode (SC/TI live cal +/-15%, NA arch-default)")
    mismatch_rec, mismatch_fail_rate = run_realistic_mismatch_experiment(n_workers=N_WORKERS)
    mismatch_csv = save_mismatch_csv(mismatch_rec)
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