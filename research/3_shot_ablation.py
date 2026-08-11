import importlib.util, pathlib, sys, csv, warnings
import multiprocessing as mp
import numpy as np
from datetime import datetime, timezone
import os as _os

_spec = importlib.util.spec_from_file_location(
    "inversion", pathlib.Path(__file__).parent / "1_inversion.py")
inv = importlib.util.module_from_spec(_spec)
sys.modules["inversion"] = inv
_spec.loader.exec_module(inv)

SCRIPT_DIR  = pathlib.Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

SEED          = 45
N_INSTANCES   = 300
ARCHITECTURES = ["superconducting", "trapped_ion", "neutral_atom"]

_arch_env = _os.environ.get("ARCH", "").strip()
if _arch_env in ARCHITECTURES:
    ARCHITECTURES = [_arch_env]

N_WORKERS = int(_os.environ.get("N_WORKERS",
                                max(1, mp.cpu_count() - 1)))

DEFAULT_SHOTS_T1     = 300
DEFAULT_SHOTS_RAMSEY = 1000
DEFAULT_SHOTS_GATE   = 500
DEFAULT_SHOTS_ECHO   = 500
DEFAULT_TOTAL        = (3*DEFAULT_SHOTS_T1 + 6*DEFAULT_SHOTS_RAMSEY +
                        3*DEFAULT_SHOTS_GATE + 3*DEFAULT_SHOTS_ECHO)  # 9900

MIN_SHOTS_PER_CIRCUIT = 10

# 9900 inserted explicitly so the deployed budget always has a data point.
# Remaining levels are every 500 shots from 1000 to 15000.
_base_levels = list(range(1000, 15500, 500))
BUDGET_LEVELS = sorted(set(_base_levels + [DEFAULT_TOTAL]))

TRUE_PARAM_RANGES = {
    "superconducting": {
        "T1_s": (80e-6,  400e-6),
        "T2_s": (40e-6,  200e-6),
        "eps":  (1e-4,   2e-3),
    },
    "trapped_ion": {
        "T1_s": (100.0,  10000.0),
        "T2_s": (0.1,    3.0),
        "eps":  (1e-4,   2e-3),
    },
    "neutral_atom": {
        "T1_s": (1.0,    100.0),
        "T2_s": (0.3,    3.0),
        "eps":  (1e-3,   1e-2),
    },
}

# Architecture-specific unmodeled hardware noise — identical to scripts
# 2_parity_experiments.py and 7_benchmark_experiments.py so all figures
# share the same noise model and are directly comparable.
#
# T1  : log-normal TLS fluctuator (Klimov et al. PRL 2018;
#                                   Carroll et al. npj QI 2022)
# T2  : uniform fractional reduction from quasi-static 1/f dephasing
# eps : coherent over-rotation delta ~ N(0, sigma_coh), adds
#       eps_coh = delta^2/4 to depolarising (Rol et al. PRL 2019)
# SPAM: calibration drift, multiplicative fractional noise
#
# dw has no perturbation: sigma_dw interacts with Canary's hard arctan2
# bounds in a way that depends on the per-instance profile dw_max and
# cannot be safely bounded by a fixed sigma.  Omitting it is consistent
# with both scripts 2 and 7.
REALISTIC_NOISE = {
    "superconducting": {
        "sigma_T1_lognormal": 0.15,
        "T2_reduction_max":   0.12,
        "sigma_coherent_rad": 0.010,
        "sigma_SPAM_frac":    0.05,
    },
    "trapped_ion": {
        "sigma_T1_lognormal": 0.00,
        "T2_reduction_max":   0.10,
        "sigma_coherent_rad": 0.005,
        "sigma_SPAM_frac":    0.03,
    },
    "neutral_atom": {
        "sigma_T1_lognormal": 0.03,
        "T2_reduction_max":   0.15,
        "sigma_coherent_rad": 0.015,
        "sigma_SPAM_frac":    0.08,
    },
}


def _log_uniform(r, lo, hi):
    return float(10 ** r.uniform(np.log10(lo), np.log10(hi)))


def sample_true_t1_t2_eps_realistic(arch_name, rng_obj):
    ranges = TRUE_PARAM_RANGES[arch_name]
    T1_lo, T1_hi = ranges["T1_s"]
    T2_lo, T2_hi = ranges["T2_s"]
    eps_lo, eps_hi = ranges["eps"]
    T1 = (_log_uniform(rng_obj, T1_lo, T1_hi) if T1_hi/T1_lo > 10.0
          else rng_obj.uniform(T1_lo, T1_hi))
    T_phi = rng_obj.uniform(T2_lo, T2_hi)
    T2    = 1.0 / (1.0/(2.0*T1) + 1.0/T_phi)
    T2    = min(T2, 2.0*T1)
    eps   = _log_uniform(rng_obj, eps_lo, eps_hi)
    return T1, T2, eps


def _perturb(arch_name, T1_true, T2_true, eps_true, rng_noise):
    """Apply architecture-specific unmodeled hardware physics.

    Returns effective (T1_eff, T2_eff, eps_eff, p0g1_eff, p1g0_eff).
    These are fed to AerSimulator so the simulation represents what the
    hardware actually has during the measurement window, not the nominal
    calibrated value.  The inversion is blind to all perturbations —
    it builds its prior from the ±15% jitter around T1_true/T2_true
    (superconducting/trapped_ion) or from arch defaults (neutral_atom).

    R² in Fig. 4 is computed against T1_eff/T2_eff/eps_eff (what the
    hardware actually had) rather than T1_true/T2_true/eps_true, which
    is the correct ground truth for a realistic measurement scenario.
    """
    arch = inv.ARCH_DEFAULTS[arch_name]
    cfg  = REALISTIC_NOISE[arch_name]

    if cfg["sigma_T1_lognormal"] > 0.0:
        T1_eff = float(np.clip(
            T1_true * np.exp(rng_noise.normal(0.0, cfg["sigma_T1_lognormal"])),
            arch["T1_min_s"], arch["T1_max_s"]))
    else:
        T1_eff = T1_true

    T2_eff = float(np.clip(
        T2_true * (1.0 - rng_noise.uniform(0.0, cfg["T2_reduction_max"])),
        arch["T2_min_s"], min(2.0 * T1_eff, arch["T1_max_s"])))

    delta_coh = rng_noise.normal(0.0, cfg["sigma_coherent_rad"])
    eps_eff   = float(np.clip(eps_true + delta_coh**2 / 4.0,
                              0.0, arch["eps_max"]))

    p0g1_nom = arch["p0_given_1"]
    p1g0_nom = arch["p1_given_0"]
    p0g1_eff = float(np.clip(
        p0g1_nom * (1.0 + rng_noise.normal(0.0, cfg["sigma_SPAM_frac"])),
        0.0, 0.30))
    p1g0_eff = float(np.clip(
        p1g0_nom * (1.0 + rng_noise.normal(0.0, cfg["sigma_SPAM_frac"])),
        0.0, 0.30))

    return T1_eff, T2_eff, eps_eff, p0g1_eff, p1g0_eff


def shots_from_budget(budget):
    scale = budget / DEFAULT_TOTAL
    st1  = max(MIN_SHOTS_PER_CIRCUIT, round(DEFAULT_SHOTS_T1     * scale))
    sram = max(MIN_SHOTS_PER_CIRCUIT, round(DEFAULT_SHOTS_RAMSEY  * scale))
    sg   = max(MIN_SHOTS_PER_CIRCUIT, round(DEFAULT_SHOTS_GATE    * scale))
    se   = max(MIN_SHOTS_PER_CIRCUIT, round(DEFAULT_SHOTS_ECHO    * scale))
    return st1, sram, sg, se


def simulate_realistic_inversion(arch_name, T1_true, T2_true, dw_true,
                                  T1_eff, T2_eff, eps_eff,
                                  p0g1_eff, p1g0_eff,
                                  instance_id,
                                  shots_t1, shots_ramsey, shots_gate,
                                  shots_echo):
    """Fig. 4 simulation: shot-efficiency sweep under realistic noise.

    Two independent mismatch layers:
      Layer 1 — prior error: delay schedule built from T1_prior/T2_prior
                (±15% of T1_true/T2_true for SC/TI; arch-default for NA).
                Canary's inversion is blind to the drift.
      Layer 2 — physics mismatch: AerSimulator runs with effective
                parameters (T1_eff, T2_eff, eps_eff, perturbed SPAM)
                from the REALISTIC_NOISE model, not the nominal values.
                Same perturbation model as 2_parity_experiments.py
                (Fig. 3) and 7_benchmark_experiments.py (Fig. 7).

    R² is computed against T1_eff/T2_eff/eps_eff — the ground truth is
    what the hardware actually had, not the nominal calibrated value.
    dw is not perturbed (consistent with scripts 2 and 7; see REALISTIC_NOISE
    comment for rationale).
    """
    arch = inv.ARCH_DEFAULTS[arch_name]

    # Build prior profile — Canary is blind to the effective parameters
    if arch_name in ("superconducting", "trapped_ion"):
        rng_param = np.random.default_rng(
            np.random.SeedSequence([SEED + 1,
                                    ARCHITECTURES.index(arch_name),
                                    instance_id]).generate_state(4)[0])
        T1_prior = float(np.clip(
            T1_true * (1.0 + rng_param.uniform(-0.15, 0.15)),
            arch["T1_min_s"], arch["T1_max_s"]))
        T2_prior = float(np.clip(
            T2_true * (1.0 + rng_param.uniform(-0.15, 0.15)),
            arch["T2_min_s"], min(2.0*T1_prior, arch["T1_max_s"])))
        profile = inv.BackendProfile.from_true_params(arch_name, T1_prior, T2_prior)
        custom_arch = dict(arch)
        custom_arch["eps_typical"] = float(np.clip(
            (eps_eff) * (1.0 + rng_param.uniform(-0.15, 0.15)),
            arch.get("eps_typical", 1e-3) * 0.01, arch["eps_max"]))
        profile.custom_arch = custom_arch
    else:
        profile = inv.BackendProfile.from_architecture(arch_name)

    circuits, meta = inv.build_probe_circuits(profile)

    # Simulate with EFFECTIVE parameters (what the hardware actually has).
    # dw_true is used directly — no perturbation (see REALISTIC_NOISE comment).
    counts_list = inv.run_probe_circuits_aer(
        circuits, meta,
        T1_eff, T2_eff, eps_eff,
        p0g1_eff, p1g0_eff,
        profile.dt_ns,
        shots_t1, shots_ramsey, shots_gate, shots_echo,
        dw_s=dw_true,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return inv.lindblad_inversion(
            counts_list, meta, profile,
            shots_t1=shots_t1, shots_ramsey=shots_ramsey,
            shots_gate=shots_gate, shots_echo=shots_echo,
            qubit_id=0,
            timestamp=datetime.now(timezone.utc).isoformat())


def compute_r2(true_vals, rec_vals):
    t = np.asarray(true_vals, dtype=float)
    r = np.asarray(rec_vals,  dtype=float)
    ss_res = np.sum((r - t)**2)
    ss_tot = np.sum((t - t.mean())**2)
    return float(1.0 - ss_res/ss_tot) if ss_tot > 0 else 0.0


def _worker(args):
    warnings.filterwarnings("ignore")
    (arch, T1_t, T2_t, dw_t, eps_t,
     T1_eff, T2_eff, eps_eff, p0g1_eff, p1g0_eff,
     attempt_id, st1, sram, sg, se) = args
    try:
        r = simulate_realistic_inversion(
            arch, T1_t, T2_t, dw_t,
            T1_eff, T2_eff, eps_eff, p0g1_eff, p1g0_eff,
            attempt_id, st1, sram, sg, se)
        if not (np.isfinite(r.T1_s) and np.isfinite(r.T2_s)):
            return None
        # Return recovered values alongside effective ground truth
        return (r.T1_s, r.T2_s, abs(r.delta_omega), r.epsilon_sx,
                T1_eff, T2_eff, abs(dw_t), eps_eff)
    except Exception:
        return None


def run_sweep():
    rows = []
    n_levels = len(BUDGET_LEVELS)

    tag  = ARCHITECTURES[0] if len(ARCHITECTURES) == 1 else "all"
    ckpt = DATA_DIR / f"fig4_shot_ablation_{tag}.csv"

    already_done = set()
    if ckpt.exists():
        existing = list(csv.DictReader(open(ckpt)))
        rows = existing
        for r in existing:
            already_done.add((int(r["shots"]), r["architecture"]))
        print(f"Resuming from checkpoint: {len(rows)} rows already saved, "
              f"{len(already_done)} (budget,arch) pairs complete.")

    for level_idx, budget in enumerate(BUDGET_LEVELS):
        st1, sram, sg, se = shots_from_budget(budget)
        actual = 3*st1 + 6*sram + 3*sg + 3*se

        for arch in ARCHITECTURES:
            if (budget, arch) in already_done:
                continue

            print(f"[{level_idx+1:3d}/{n_levels}] budget={budget:6d} "
                  f"actual={actual:6d} arch={arch} "
                  f"T1={st1} Ramsey={sram} Gate={sg} Echo={se}", flush=True)

            rng_inst = np.random.default_rng(
                SEED + 1 + level_idx * 13 + ARCHITECTURES.index(arch) * 7)
            rng_noise = np.random.default_rng(
                SEED + 2 + level_idx * 17 + ARCHITECTURES.index(arch) * 11)

            arch_default_dw_max = (
                inv.BackendProfile.from_true_params(
                    arch,
                    inv.ARCH_DEFAULTS[arch]["T1_s"],
                    inv.ARCH_DEFAULTS[arch]["T2_s"]).dw_max_rad_s
                if arch in ("superconducting", "trapped_ion")
                else inv.BackendProfile.from_architecture(arch).dw_max_rad_s)

            batch_args = []
            attempt_id = 0
            while len(batch_args) < int(N_INSTANCES * 1.2):
                attempt_id += 1
                T1_t, T2_t, eps_t = sample_true_t1_t2_eps_realistic(
                    arch, rng_inst)
                dw_max = (inv.BackendProfile.from_true_params(
                              arch, T1_t, T2_t).dw_max_rad_s
                          if arch in ("superconducting", "trapped_ion")
                          else arch_default_dw_max)
                dw_t = (rng_inst.choice([-1, 1]) *
                        rng_inst.uniform(0.2 * dw_max, dw_max))

                # Perturb in the main process so RNG state is deterministic
                # and reproducible regardless of worker scheduling order.
                T1_eff, T2_eff, eps_eff, p0g1_eff, p1g0_eff = _perturb(
                    arch, T1_t, T2_t, eps_t, rng_noise)

                batch_args.append((arch, T1_t, T2_t, dw_t, eps_t,
                                   T1_eff, T2_eff, eps_eff,
                                   p0g1_eff, p1g0_eff,
                                   attempt_id, st1, sram, sg, se))

            # Ground truth: effective values (what hardware actually had)
            rec = {"T1_true":[], "T1_rec":[],
                   "T2_true":[], "T2_rec":[],
                   "dw_true":[], "dw_rec":[],
                   "eps_true":[], "eps_rec":[]}

            with mp.Pool(N_WORKERS) as pool:
                for res in pool.imap(_worker, batch_args):
                    if res is None or len(rec["T1_true"]) >= N_INSTANCES:
                        continue
                    (T1_rec, T2_rec, dw_rec, eps_rec,
                     T1_eff, T2_eff, dw_eff, eps_eff) = res
                    rec["T1_true"].append(T1_eff);  rec["T1_rec"].append(T1_rec)
                    rec["T2_true"].append(T2_eff);  rec["T2_rec"].append(T2_rec)
                    rec["dw_true"].append(dw_eff);  rec["dw_rec"].append(dw_rec)
                    rec["eps_true"].append(eps_eff); rec["eps_rec"].append(eps_rec)

            r2_T1  = compute_r2(rec["T1_true"],  rec["T1_rec"])
            r2_T2  = compute_r2(rec["T2_true"],  rec["T2_rec"])
            r2_dw  = compute_r2(rec["dw_true"],  rec["dw_rec"])
            r2_eps = compute_r2(rec["eps_true"],  rec["eps_rec"])
            n      = len(rec["T1_true"])

            print(f"         n={n}/{N_INSTANCES} | "
                  f"T1={r2_T1:.4f} T2={r2_T2:.4f} "
                  f"dw={r2_dw:.4f} eps={r2_eps:.4f}", flush=True)

            for param, r2 in [("T1", r2_T1), ("T2", r2_T2),
                               ("delta_omega", r2_dw), ("epsilon_sx", r2_eps)]:
                rows.append({"shots": budget, "architecture": arch,
                              "parameter": param, "r2": f"{r2:.6f}"})

            with open(ckpt, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["shots", "architecture",
                                                   "parameter", "r2"])
                w.writeheader()
                w.writerows(rows)

    return rows


def save_csv(rows):
    tag  = ARCHITECTURES[0] if len(ARCHITECTURES) == 1 else "all"
    path = DATA_DIR / f"fig4_shot_ablation_{tag}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["shots", "architecture",
                                          "parameter", "r2"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {path}  ({len(rows)} rows)")
    return path


if __name__ == "__main__":
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"[{ts}] Shot ablation sweep")
    print(f"  Budget range : {BUDGET_LEVELS[0]:,} – {BUDGET_LEVELS[-1]:,} "
          f"(step 500 + {DEFAULT_TOTAL:,} native, {len(BUDGET_LEVELS)} levels)")
    print(f"  N instances  : {N_INSTANCES}/arch per level")
    print(f"  Noise model  : Fig. 3 realistic (TLS/1f/coherent/SPAM perturbations; "
          f"SC/TI prior ±15%, NA arch-default)")
    print(f"  Ground truth : effective params (T1_eff, T2_eff, eps_eff) "
          f"not nominal")
    print(f"  Seed         : {SEED}")
    print(f"  Workers      : {N_WORKERS}")
    print()

    rows = run_sweep()
    save_csv(rows)