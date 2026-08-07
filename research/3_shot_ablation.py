import importlib.util, pathlib, sys, csv, warnings
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

SEED          = 45
N_INSTANCES   = 150
ARCHITECTURES = ["superconducting", "trapped_ion", "neutral_atom"]

DEFAULT_SHOTS_T1     = 300
DEFAULT_SHOTS_RAMSEY = 1000
DEFAULT_SHOTS_GATE   = 500
DEFAULT_SHOTS_ECHO   = 500
DEFAULT_TOTAL        = (3*DEFAULT_SHOTS_T1 + 6*DEFAULT_SHOTS_RAMSEY +
                        3*DEFAULT_SHOTS_GATE + 3*DEFAULT_SHOTS_ECHO)  # 9900

MIN_SHOTS_PER_CIRCUIT = 10

BUDGET_LEVELS = list(range(1000, 15100, 100))

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


def _spawn_probe_rngs(base_seed, *keys, n_streams=4):
    ss = np.random.SeedSequence([base_seed, *keys])
    return tuple(np.random.default_rng(s) for s in ss.spawn(n_streams))


def shots_from_budget(budget):
    scale = budget / DEFAULT_TOTAL
    st1  = max(MIN_SHOTS_PER_CIRCUIT, round(DEFAULT_SHOTS_T1     * scale))
    sram = max(MIN_SHOTS_PER_CIRCUIT, round(DEFAULT_SHOTS_RAMSEY  * scale))
    sg   = max(MIN_SHOTS_PER_CIRCUIT, round(DEFAULT_SHOTS_GATE    * scale))
    se   = max(MIN_SHOTS_PER_CIRCUIT, round(DEFAULT_SHOTS_ECHO    * scale))
    return st1, sram, sg, se


def simulate_realistic_inversion(arch_name, T1_true, T2_true, dw_true,
                                  eps_true, instance_id, rng_mismatch,
                                  shots_t1, shots_ramsey, shots_gate,
                                  shots_echo):
    arch     = inv.ARCH_DEFAULTS[arch_name]
    arch_idx = ARCHITECTURES.index(arch_name)
    extra_seed = int(rng_mismatch.integers(1, 2**31 - 1))
    rng_t1, rng_ramsey, rng_gate, rng_echo, rng_param = _spawn_probe_rngs(
        SEED + 1, arch_idx, instance_id, extra_seed, n_streams=5)

    if arch_name in ("superconducting", "trapped_ion"):
        T1_prior = float(np.clip(
            T1_true * (1.0 + rng_param.uniform(-0.15, 0.15)),
            arch["T1_min_s"], arch["T1_max_s"]))
        T2_prior = float(np.clip(
            T2_true * (1.0 + rng_param.uniform(-0.15, 0.15)),
            arch["T2_min_s"], min(2.0*T1_prior, arch["T1_max_s"])))
        profile = inv.BackendProfile.from_true_params(arch_name, T1_prior, T2_prior)
        custom_arch = dict(arch)
        custom_arch["eps_typical"] = eps_true
        profile.custom_arch = custom_arch
    else:
        profile = inv.BackendProfile.from_architecture(arch_name)

    _, meta       = inv.build_probe_circuits(profile)
    t1_delays     = meta["t1_delays_s"]
    ramsey_delays = meta["ramsey_delays_s"]
    echo_delays   = meta["echo_delays_s"]
    Nv            = np.array(meta["gate_rep_N"], dtype=float)

    def c1(r, p, sh):
        n1 = int(r.binomial(sh, float(np.clip(p, 0, 1))))
        return {"0": sh-n1, "1": n1}

    def c0(r, p, sh):
        n0 = int(r.binomial(sh, float(np.clip(p, 0, 1))))
        return {"0": n0, "1": sh-n0}

    T2_echo_true = min(T2_true, 2.0*T1_true)

    if arch_name == "superconducting":
        p0_base    = arch["p0_given_1"]
        p1_base    = arch["p1_given_0"]
        spam_drift = float(np.clip(rng_t1.normal(0, 0.005), 0.0, 0.05))
        p0_eff     = float(np.clip(p0_base + spam_drift, 0.0, 0.05))
        p1_eff     = float(np.clip(p1_base + spam_drift, 0.0, 0.05))
        def spam(p): return p*(1.0 - p0_eff) + (1.0 - p)*p1_eff

        T1_local = float(np.clip(T1_true * (1.0 + rng_t1.normal(0, 0.08)),
                                 arch["T1_min_s"], arch["T1_max_s"]))
        T2_local = float(np.clip(T2_true * (1.0 + rng_ramsey.normal(0, 0.08)),
                                 arch["T2_min_s"],
                                 min(2.0*T1_local, arch["T1_max_s"])))

        t1_counts = [c1(rng_t1, spam(float(inv.forward_t1(d, T1_local))),
                        shots_t1) for d in t1_delays]

        ramsey_counts = []
        for t in ramsey_delays:
            decay = np.exp(-t / T2_local)
            px = 0.5*(1.0 - decay*np.cos(dw_true*t))
            py = 0.5*(1.0 - decay*np.sin(dw_true*t))
            ramsey_counts.append(c1(rng_ramsey, spam(px), shots_ramsey))
            ramsey_counts.append(c1(rng_ramsey, spam(py), shots_ramsey))

        p0_gate    = inv.forward_gate(Nv, eps_true)
        gate_counts = [c0(rng_gate, spam(p), shots_gate) for p in p0_gate]

        T2_echo_local = min(T2_local, 2.0*T1_local)
        echo_counts = [c1(rng_echo,
                          spam(float(inv.forward_echo(t, T2_echo_local))),
                          shots_echo) for t in echo_delays]

    elif arch_name == "trapped_ion":
        t1_counts = []
        for d in t1_delays:
            T1_local = T1_true * np.exp(-d / (50.0*T1_true))
            t1_counts.append(c1(rng_t1,
                                float(inv.forward_t1(d, T1_local)),
                                shots_t1))

        ramsey_counts = []
        for t in ramsey_delays:
            dw_local = dw_true + rng_ramsey.normal(0, 0.05*abs(dw_true))
            px, py   = inv.forward_ramsey_xy(t, T2_true, dw_local)
            px_eff   = px * (1.0 + 0.03*np.cos(2*np.pi*t/T2_true))
            py_eff   = py * (1.0 + 0.03*np.sin(2*np.pi*t/T2_true))
            ramsey_counts.append(c1(rng_ramsey, px_eff, shots_ramsey))
            ramsey_counts.append(c1(rng_ramsey, py_eff, shots_ramsey))

        p0_gate    = inv.forward_gate(Nv, eps_true)
        gate_counts = [c0(rng_gate, p, shots_gate) for p in p0_gate]

        echo_counts = [c1(rng_echo,
                          float(inv.forward_echo(t, T2_echo_true)),
                          shots_echo) for t in echo_delays]

    else:  # neutral_atom
        t1_counts = [c1(rng_t1,
                        float(inv.forward_t1(d, T1_true)),
                        shots_t1) for d in t1_delays]

        dw_local = dw_true + rng_ramsey.normal(0, 0.08*abs(dw_true))
        ramsey_counts = []
        for t in ramsey_delays:
            px, py = inv.forward_ramsey_xy(t, T2_true, dw_local)
            ramsey_counts.append(c1(rng_ramsey, px, shots_ramsey))
            ramsey_counts.append(c1(rng_ramsey, py, shots_ramsey))

        eps_max     = arch["eps_max"]
        gate_counts = []
        for N in Nv:
            eps_local = float(np.clip(
                eps_true * (1.0 + rng_gate.normal(0, 0.10)), 0.0, eps_max))
            p0 = float(inv.forward_gate(np.array([N]), eps_local)[0])
            gate_counts.append(c0(rng_gate, p0, shots_gate))

        echo_counts = [c1(rng_echo,
                          float(inv.forward_echo(t, T2_echo_true)),
                          shots_echo) for t in echo_delays]

    counts_list = t1_counts + ramsey_counts + gate_counts + echo_counts

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


def run_sweep():
    rows = []
    n_levels = len(BUDGET_LEVELS)

    for level_idx, budget in enumerate(BUDGET_LEVELS):
        st1, sram, sg, se = shots_from_budget(budget)
        actual = 3*st1 + 6*sram + 3*sg + 3*se

        print(f"[{level_idx+1:3d}/{n_levels}] budget={budget:6d} "
              f"actual={actual:6d} | "
              f"T1={st1} Ramsey={sram} Gate={sg} Echo={se}")

        for arch in ARCHITECTURES:
            rng_mismatch = np.random.default_rng(SEED + 1 + level_idx * 13 +
                                                  ARCHITECTURES.index(arch) * 7)

            arch_default_dw_max = (
                inv.BackendProfile.from_true_params(arch,
                    inv.ARCH_DEFAULTS[arch]["T1_s"],
                    inv.ARCH_DEFAULTS[arch]["T2_s"]).dw_max_rad_s
                if arch in ("superconducting", "trapped_ion")
                else inv.BackendProfile.from_architecture(arch).dw_max_rad_s)

            rec = {k: [] for k in [
                "T1_true","T1_rec","T2_true","T2_rec",
                "dw_true","dw_rec","eps_true","eps_rec"]}

            success  = 0
            attempts = 0
            while success < N_INSTANCES:
                attempts += 1
                try:
                    T1_t, T2_t, eps_t = sample_true_t1_t2_eps_realistic(
                        arch, rng_mismatch)

                    if arch in ("superconducting", "trapped_ion"):
                        dw_max = inv.BackendProfile.from_true_params(
                            arch, T1_t, T2_t).dw_max_rad_s
                    else:
                        dw_max = arch_default_dw_max

                    dw_t = rng_mismatch.choice([-1, 1]) * \
                           rng_mismatch.uniform(0.2*dw_max, dw_max)

                    r = simulate_realistic_inversion(
                        arch, T1_t, T2_t, dw_t, eps_t,
                        attempts, rng_mismatch,
                        st1, sram, sg, se)

                    if not (np.isfinite(r.T1_s) and np.isfinite(r.T2_s)):
                        continue

                    rec["T1_true"].append(T1_t);   rec["T1_rec"].append(r.T1_s)
                    rec["T2_true"].append(T2_t);   rec["T2_rec"].append(r.T2_s)
                    rec["dw_true"].append(abs(dw_t)); rec["dw_rec"].append(abs(r.delta_omega))
                    rec["eps_true"].append(eps_t); rec["eps_rec"].append(r.epsilon_sx)
                    success += 1
                except Exception:
                    continue

            r2_T1  = compute_r2(rec["T1_true"],  rec["T1_rec"])
            r2_T2  = compute_r2(rec["T2_true"],  rec["T2_rec"])
            r2_dw  = compute_r2(rec["dw_true"],  rec["dw_rec"])
            r2_eps = compute_r2(rec["eps_true"],  rec["eps_rec"])

            print(f"         {arch:18s} | "
                  f"T1={r2_T1:.4f} T2={r2_T2:.4f} "
                  f"dw={r2_dw:.4f} eps={r2_eps:.4f}")

            for param, r2 in [("T1", r2_T1), ("T2", r2_T2),
                               ("delta_omega", r2_dw), ("epsilon_sx", r2_eps)]:
                rows.append({
                    "shots":        budget,
                    "architecture": arch,
                    "parameter":    param,
                    "r2":           f"{r2:.6f}",
                })

    return rows


def save_csv(rows):
    path = DATA_DIR / "fig4_shot_ablation.csv"
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
          f"(step 100, {len(BUDGET_LEVELS)} levels)")
    print(f"  N instances  : {N_INSTANCES}/arch per level")
    print(f"  Noise model  : Fig. 3 realistic (SC/TI live-cal ±15%, "
          f"NA arch-default)")
    print(f"  Seed         : {SEED}")
    print()

    rows = run_sweep()
    save_csv(rows)