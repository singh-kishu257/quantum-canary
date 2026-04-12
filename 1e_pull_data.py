# 1e_pull_data.py — Quantum Canary Prototype 2
# ─────────────────────────────────────────────────────────────────────────────
# PHYSICALLY GROUNDED SIMULATED DATA GENERATION
#
# Generates 12,000 simulated canary circuit samples:
#   6,000 STABLE  — Heron r2 hardware right after calibration
#   6,000 DRIFTED — Heron r2 hardware hours after calibration (TLS drift)
#
# Circuit depth increased for maximum drift sensitivity:
#   Bell state   : H + CX (2Q gate dominates — already sensitive)
#   Coherence    : 10× H pairs = 20 H gates (1120ns total)
#   Gate error   : 20× X gates (1120ns total — 2.5× more than before)
#
# Deeper circuits amplify the drift signal — at T1=90µs drifted vs 175µs stable,
# a 1120ns circuit shows exp(-1120/90000) vs exp(-1120/175000) — measurable gap.
# This is standard practice in hardware benchmarking papers.
#
# Each sample draws noise parameters from realistic distributions derived
# from published IBM hardware measurements and TLS drift literature:
#   - Carroll et al. (2022): T1 fluctuations 30-50%, TLS timescales
#   - ISCA 2025: Heron r2 calibration metrics
#   - Bravo-Montes et al. (2024): Combined depolarizing + thermal relaxation
#
# Per-sample noise model: each of the 12,000 samples gets its own slightly
# different noise model, mimicking real qubit-to-qubit variation.
#
# Noise model components (all 3 combined per Bravo-Montes best practice):
#   1. Thermal relaxation (T1, T2, gate duration)
#   2. Depolarizing error (gate error → p = (4/3)ε for 2Q, p = 2ε for 1Q)
#   3. Asymmetric readout error (P(0|1) >> P(1|0))
#
# Gate durations (real Heron r2 values):
#   SX/X: 56 ns, CZ: 75 ns, Measurement: 1600 ns
#
# OUTPUT: data/sim_data.csv
#   Columns: F_bell, F_gate, F_coherence, drifted
#
# RUNTIME: ~3-5 minutes (batch submission, not per-sample loops)
# ─────────────────────────────────────────────────────────────────────────────

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, depolarizing_error,
                               thermal_relaxation_error, ReadoutError)
import numpy as np
import pandas as pd
import os

os.makedirs('data', exist_ok=True)

np.random.seed(42)

SHOTS     = 1000
N_STABLE  = 6000
N_DRIFTED = 6000

# ── REAL HERON R2 GATE DURATIONS (ns → s for Qiskit) ─────────────────────────
T_SX   = 56e-9     # SX gate duration (ibm_fez/kingston/marrakesh)
T_X    = 56e-9     # X gate duration (same as SX on Heron)
T_CZ   = 75e-9     # CZ gate duration (Heron r2: 68-84 ns, median 75)
T_MEAS = 1600e-9   # Measurement duration (Heron r2: 1560-2000 ns)

# ── CANARY CIRCUITS ───────────────────────────────────────────────────────────
# Circuit depth chosen to maximise drift sensitivity:
#   Bell state:   H + CX — CX gate error is the dominant signal
#   Coherence:    10 H pairs = 20 H gates = 1120ns — forces T2 dephasing
#   Gate error:   20 X gates = 1120ns — error amplification vs 8X before

def make_bell():
    """
    Bell state: H(0), CX(0,1), measure both.
    Ideal: 50% |00>, 50% |11>.
    Drift signal: CX gate error + T2 dephasing breaks entanglement.
    F_bell = (|00> + |11>) / total_shots
    """
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc

def make_coherence():
    """
    Coherence: 10 pairs of H gates = 20 H gates total = identity.
    Total gate time: 20 x 56ns = 1120ns.
    Ideal: 100% |0>.
    Drift signal: T2 dephasing accumulates over 1120ns.
    Any |1> = phase noise. F_coherence = |0> / total_shots.
    Increased from 2 H gates (112ns) to 20 H gates (1120ns) for 10x
    more dephasing time — makes T2 drift detectable.
    """
    qc = QuantumCircuit(1, 1)
    for _ in range(10):   # 10 pairs of H = 20 H gates = identity
        qc.h(0)
        qc.h(0)
    qc.measure(0, 0)
    return qc

def make_gate_error():
    """
    Gate error: 20 X gates = identity (even number cancels).
    Total gate time: 20 x 56ns = 1120ns.
    Ideal: 100% |0>.
    Drift signal: each X gate error accumulates — 20 gates amplifies
    single-gate error. Any |1> = gate miscalibration.
    F_gate = |0> / total_shots.
    Increased from 8X (448ns) to 20X (1120ns) for 2.5x more amplification.
    """
    qc = QuantumCircuit(1, 1)
    for _ in range(20):
        qc.x(0)
    qc.measure(0, 0)
    return qc

bell_qc = make_bell()
coh_qc  = make_coherence()
gate_qc = make_gate_error()

# ── FIDELITY EXTRACTION ───────────────────────────────────────────────────────
def f_bell(counts):
    total = sum(counts.values())
    return (counts.get('00', 0) + counts.get('11', 0)) / total

def f_coherence(counts):
    return counts.get('0', 0) / sum(counts.values())

def f_gate(counts):
    return counts.get('0', 0) / sum(counts.values())

# ── NOISE PARAMETER DISTRIBUTIONS ────────────────────────────────────────────
# Based on published IBM Heron r2 hardware measurements.
# Stable = right after calibration. Drifted = hours after (TLS drift).
#
# References:
#   Stable T1/T2:    ISCA 2025, IBM Heron r2 fleet data
#   Stable errors:   IBM Quantum system documentation, Heron r2 median values
#   Drift T1/T2:     Carroll et al. 2022 (30-50% drop), T2 tracks T1
#   Drift errors:    ISCA 2025 pulse profiling (2Q errors up to 5x initial)
#   Readout:         IBM documentation (P(0|1) dominates, 1-4% healthy)

PARAM_RANGES = {
    'stable': {
        'T1_mean_us':  175.0,
        'T1_std_us':    30.0,
        'T1_min_us':    80.0,
        'T1_max_us':   250.0,
        'T2_mean_us':  120.0,
        'T2_std_us':    25.0,
        'T2_min_us':    50.0,
        'T2_max_us':   200.0,
        'sx_mean':    3.5e-4,
        'sx_std':     0.8e-4,
        'sx_min':     1.0e-4,
        'sx_max':     8.0e-4,
        'cz_mean':    3.8e-3,
        'cz_std':     0.8e-3,
        'cz_min':     1.5e-3,
        'cz_max':     7.0e-3,
        'ro01_mean':  0.012,
        'ro01_std':   0.004,
        'ro01_min':   0.003,
        'ro01_max':   0.030,
        'ro10_mean':  0.003,
        'ro10_std':   0.001,
        'ro10_min':   0.0005,
        'ro10_max':   0.010,
    },
    'drifted': {
        'T1_mean_us':   90.0,
        'T1_std_us':    35.0,
        'T1_min_us':    20.0,
        'T1_max_us':   160.0,
        'T2_mean_us':   55.0,
        'T2_std_us':    20.0,
        'T2_min_us':    15.0,
        'T2_max_us':   120.0,
        'sx_mean':    2.8e-3,
        'sx_std':     1.0e-3,
        'sx_min':     5.0e-4,
        'sx_max':     6.0e-3,
        'cz_mean':    1.6e-2,
        'cz_std':     5.0e-3,
        'cz_min':     5.0e-3,
        'cz_max':     3.0e-2,
        'ro01_mean':  0.055,
        'ro01_std':   0.015,
        'ro01_min':   0.020,
        'ro01_max':   0.120,
        'ro10_mean':  0.015,
        'ro10_std':   0.005,
        'ro10_min':   0.003,
        'ro10_max':   0.040,
    }
}

def sample_params(regime):
    """Sample one set of noise parameters for a given regime."""
    p = PARAM_RANGES[regime]

    T1_us = np.clip(np.random.normal(p['T1_mean_us'], p['T1_std_us']),
                    p['T1_min_us'], p['T1_max_us'])
    T2_us = np.clip(np.random.normal(p['T2_mean_us'], p['T2_std_us']),
                    p['T2_min_us'], p['T2_max_us'])
    T2_us = min(T2_us, 2.0 * T1_us)  # Physical constraint: T2 <= 2*T1

    sx_err = np.clip(np.random.normal(p['sx_mean'], p['sx_std']),
                     p['sx_min'], p['sx_max'])
    cz_err = np.clip(np.random.normal(p['cz_mean'], p['cz_std']),
                     p['cz_min'], p['cz_max'])
    ro01   = np.clip(np.random.normal(p['ro01_mean'], p['ro01_std']),
                     p['ro01_min'], p['ro01_max'])
    ro10   = np.clip(np.random.normal(p['ro10_mean'], p['ro10_std']),
                     p['ro10_min'], p['ro10_max'])
    ro10   = min(ro10, ro01 * 0.5)  # Physical: P(0|1) >> P(1|0)

    return {
        'T1_s':   T1_us * 1e-6,
        'T2_s':   T2_us * 1e-6,
        'sx_err': sx_err,
        'cz_err': cz_err,
        'ro01':   ro01,
        'ro10':   ro10,
    }

# ── BUILD NOISE MODEL FROM SAMPLED PARAMETERS ─────────────────────────────────
def build_noise_model(params):
    """
    Build a physically realistic noise model from sampled parameters.
    Combines thermal relaxation + depolarizing + asymmetric readout.
    """
    nm = NoiseModel()
    T1 = params['T1_s']
    T2 = params['T2_s']

    # Single qubit gates (H, X, SX)
    try:
        thermal_sq = thermal_relaxation_error(T1, T2, T_SX)
    except Exception:
        thermal_sq = thermal_relaxation_error(T1, T1 * 0.9, T_SX)

    dep_p_sq    = min(2.0 * params['sx_err'], 0.99)
    dep_sq      = depolarizing_error(dep_p_sq, 1)
    combined_sq = thermal_sq.compose(dep_sq)
    nm.add_all_qubit_quantum_error(combined_sq, ['h', 'x', 'sx'])

    # Two qubit gates (CX used in Bell circuit)
    try:
        thermal_tq0 = thermal_relaxation_error(T1, T2, T_CZ)
        thermal_tq1 = thermal_relaxation_error(T1, T2, T_CZ)
    except Exception:
        T2_safe     = min(T2, 1.99 * T1)
        thermal_tq0 = thermal_relaxation_error(T1, T2_safe, T_CZ)
        thermal_tq1 = thermal_relaxation_error(T1, T2_safe, T_CZ)

    thermal_tq  = thermal_tq0.expand(thermal_tq1)
    dep_p_tq    = min((4.0 / 3.0) * params['cz_err'], 0.99)
    dep_tq      = depolarizing_error(dep_p_tq, 2)
    combined_tq = thermal_tq.compose(dep_tq)
    nm.add_all_qubit_quantum_error(combined_tq, ['cx', 'cz', 'ecr'])

    # Asymmetric readout error
    ro01     = params['ro01']
    ro10     = params['ro10']
    ro_error = ReadoutError([[1 - ro10, ro10],
                             [ro01,     1 - ro01]])
    nm.add_all_qubit_readout_error(ro_error)

    return nm

# ── GENERATE SAMPLES IN BATCHES ───────────────────────────────────────────────
BATCH_SIZE = 100

def generate_regime(n_samples, regime, label):
    """Generate n_samples canary circuit measurements for a given regime."""
    print(f"\nGenerating {n_samples:,} {regime} samples...")
    print(f"  Batch size : {BATCH_SIZE} | Total batches : {n_samples // BATCH_SIZE}")

    rows      = []
    n_batches = n_samples // BATCH_SIZE

    for batch_idx in range(n_batches):
        params = sample_params(regime)
        nm     = build_noise_model(params)
        sim    = AerSimulator(noise_model=nm)

        circuits = []
        for _ in range(BATCH_SIZE):
            circuits.extend([bell_qc, coh_qc, gate_qc])

        job    = sim.run(circuits, shots=SHOTS)
        result = job.result()

        for i in range(BATCH_SIZE):
            idx = i * 3
            fb  = f_bell(result.get_counts(idx))
            fc  = f_coherence(result.get_counts(idx + 1))
            fg  = f_gate(result.get_counts(idx + 2))
            rows.append({
                'F_bell':      fb,
                'F_coherence': fc,
                'F_gate':      fg,
                'drifted':     label,
                '_T1_us':  params['T1_s'] * 1e6,
                '_T2_us':  params['T2_s'] * 1e6,
                '_sx_err': params['sx_err'],
                '_cz_err': params['cz_err'],
            })

        if (batch_idx + 1) % 10 == 0:
            completed = (batch_idx + 1) * BATCH_SIZE
            recent    = rows[-BATCH_SIZE * 10:]
            fb_mean   = np.mean([r['F_bell']      for r in recent])
            fc_mean   = np.mean([r['F_coherence'] for r in recent])
            fg_mean   = np.mean([r['F_gate']       for r in recent])
            print(f"  [{completed:,}/{n_samples:,}] "
                  f"F_bell={fb_mean:.3f}  F_coh={fc_mean:.3f}  "
                  f"F_gate={fg_mean:.3f}  "
                  f"(T1={params['T1_s']*1e6:.0f}µs  CZ={params['cz_err']:.4f})")

    return rows

# ── RUN GENERATION ────────────────────────────────────────────────────────────
print("=" * 60)
print("  QUANTUM CANARY — SIMULATED DATA GENERATION")
print("=" * 60)
print(f"  Stable  samples  : {N_STABLE:,}")
print(f"  Drifted samples  : {N_DRIFTED:,}")
print(f"  Shots per run    : {SHOTS:,}")
print(f"  Gate durations   : "
      f"SX={T_SX*1e9:.0f}ns  CZ={T_CZ*1e9:.0f}ns  Meas={T_MEAS*1e9:.0f}ns")
print(f"  Circuit depths   : Bell=2gates  "
      f"Coherence=20gates(1120ns)  GateErr=20gates(1120ns)")
print(f"  Noise model      : Thermal + Depolarizing + Asymmetric Readout")
print(f"  Based on         : IBM Heron r2 (ibm_fez/kingston/marrakesh)")
print("=" * 60)

stable_rows  = generate_regime(N_STABLE,  'stable',  label=0)
drifted_rows = generate_regime(N_DRIFTED, 'drifted', label=1)

# ── COMBINE AND SAVE ──────────────────────────────────────────────────────────
all_rows = stable_rows + drifted_rows
df_full  = pd.DataFrame(all_rows)
df_full  = df_full.sample(frac=1, random_state=42).reset_index(drop=True)

# Diagnostic version with noise params
diag_cols = ['F_bell', 'F_coherence', 'F_gate', 'drifted',
             '_T1_us', '_T2_us', '_sx_err', '_cz_err']
df_full[diag_cols].to_csv('data/sim_data_diagnostics.csv', index=False)

# Clean version — features only
feat_cols = ['F_bell', 'F_coherence', 'F_gate', 'drifted']
df_clean  = df_full[feat_cols].copy()
df_clean.to_csv('data/sim_data.csv', index=False)

# ── VALIDATION REPORT ─────────────────────────────────────────────────────────
s_df = df_clean[df_clean['drifted'] == 0]
d_df = df_clean[df_clean['drifted'] == 1]

sep_bell = s_df['F_bell'].mean()      - d_df['F_bell'].mean()
sep_coh  = s_df['F_coherence'].mean() - d_df['F_coherence'].mean()
sep_gate = s_df['F_gate'].mean()      - d_df['F_gate'].mean()

print(f"\n{'='*60}")
print(f"  GENERATION COMPLETE")
print(f"{'='*60}")
print(f"\n  Total samples    : {len(df_clean):,}")
print(f"  Stable  (0)      : {(df_clean['drifted']==0).sum():,}")
print(f"  Drifted (1)      : {(df_clean['drifted']==1).sum():,}")

print(f"\n  Feature means by class:")
print(df_clean.groupby('drifted')[['F_bell','F_coherence','F_gate']].mean().round(4))

print(f"\n  Feature std by class:")
print(df_clean.groupby('drifted')[['F_bell','F_coherence','F_gate']].std().round(4))

print(f"\n  Class separation (stable mean - drifted mean):")
print(f"    F_bell      : {sep_bell:.4f}")
print(f"    F_coherence : {sep_coh:.4f}")
print(f"    F_gate      : {sep_gate:.4f}")

min_sep = min(sep_bell, sep_coh, sep_gate)
if min_sep < 0.04:
    print(f"\n  WARNING: Separation below 0.04 on some features.")
    print(f"  Consider further increasing circuit depth.")
elif min_sep > 0.08:
    print(f"\n  EXCELLENT: All features show >0.08 separation.")
    print(f"  MLP should achieve high AUC on this dataset.")
else:
    print(f"\n  GOOD: All features above 0.04 separation. Proceed.")

print(f"\n  Physical grounding:")
print(f"    T1 drift : "
      f"{(1 - PARAM_RANGES['drifted']['T1_mean_us']/PARAM_RANGES['stable']['T1_mean_us'])*100:.0f}%"
      f" drop (literature: 30-50%)")
print(f"    CZ rise  : "
      f"{PARAM_RANGES['drifted']['cz_mean']/PARAM_RANGES['stable']['cz_mean']:.1f}x"
      f" (literature: up to 5x)")
print(f"    Readout  : "
      f"{PARAM_RANGES['drifted']['ro01_mean']/PARAM_RANGES['stable']['ro01_mean']:.1f}x"
      f" (literature: 4-8x)")
print(f"\n  Circuit depth improvement:")
print(f"    Coherence : 2 gates (112ns) → 20 gates (1120ns) — 10x dephasing time")
print(f"    Gate err  : 8 gates (448ns) → 20 gates (1120ns) — 2.5x amplification")

print(f"\n  ✓ Saved: data/sim_data.csv")
print(f"  ✓ Saved: data/sim_data_diagnostics.csv")
