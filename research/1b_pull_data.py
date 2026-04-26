# 1b_pull_data.py — Quantum Canary 2
# ─────────────────────────────────────────────────────────────────────────────
# PHYSICALLY GROUNDED SIMULATED DATA GENERATION
#
# Generates 15,000 simulated canary circuit samples across 3 regimes:
#   5,000 STABLE      — Heron r2 right after calibration (label=0)
#   5,000 BORDERLINE  — 2-4 hours after calibration, early TLS drift (label=1)
#   5,000 DRIFTED     — 6-12 hours after calibration, significant drift (label=1)
#
# WHY THREE REGIMES:
#   Real IBM hardware degrades continuously — not as a binary jump.
#   Stable and drifted are the extremes. Borderline is the hard region:
#   T1 has dropped 10-25%, CZ has risen moderately. Fidelity values
#   overlap with the stable class. A simple mean-threshold classifier
#   cannot reliably separate borderline from stable — it requires the
#   MLP's nonlinear decision boundary to catch these early drift events.
#   This is physically motivated: Carroll et al. (2022) documents TLS
#   fluctuations beginning within 1-2 hours of calibration.
#
# LABELING:
#   Borderline = drifted=1. A 15-25% T1 drop meets the labeling threshold
#   from 2_label_data.py (T1 drop >= 15%). These are genuine drift events —
#   just subtle ones that threshold classifiers miss.
#
# Circuit depth chosen for maximum drift sensitivity:
#   Bell state   : H + CX (2Q gate dominates)
#   Coherence    : 10× H pairs = 20 H gates (1120ns total)
#   Gate error   : 20× X gates (1120ns total)
#
# References:
#   Carroll et al. (2022): T1 fluctuations 30-50%, TLS timescales
#   ISCA 2025: Heron r2 calibration metrics
#   Bravo-Montes et al. (2024): Combined depolarizing + thermal relaxation
#
# OUTPUT: data/sim_data.csv
#   Columns: F_bell, F_gate, F_coherence, drifted
#
# RUNTIME: ~4-6 minutes
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

SHOTS        = 1000
N_STABLE     = 5000
N_BORDERLINE = 5000   # the hard ambiguous zone — this is what breaks simple thresholds
N_DRIFTED    = 5000

# ── REAL HERON R2 GATE DURATIONS ─────────────────────────────────────────────
T_SX   = 56e-9
T_X    = 56e-9
T_CZ   = 75e-9
T_MEAS = 1600e-9

# ── CANARY CIRCUITS ───────────────────────────────────────────────────────────
def make_bell():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc

def make_coherence():
    qc = QuantumCircuit(1, 1)
    for _ in range(10):
        qc.h(0)
        qc.h(0)
    qc.measure(0, 0)
    return qc

def make_gate_error():
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
# Three physically distinct regimes:
#
# STABLE     : right after calibration. Peak hardware health.
# BORDERLINE : 2-4 hours post-calibration. T1 dropped 10-25%, CZ risen
#              moderately. Fidelity values overlap with stable class.
#              This is the regime a threshold classifier cannot reliably catch.
# DRIFTED    : 6-12 hours post-calibration. Significant TLS drift.
#              T1 dropped 30-50% (Carroll et al. 2022), CZ up to 5x.

PARAM_RANGES = {
    'stable': {
        'T1_mean_us':  175.0,
        'T1_std_us':    25.0,
        'T1_min_us':   120.0,   # tightened min — truly healthy qubits
        'T1_max_us':   250.0,
        'T2_mean_us':  120.0,
        'T2_std_us':    20.0,
        'T2_min_us':    80.0,
        'T2_max_us':   200.0,
        'sx_mean':    3.5e-4,
        'sx_std':     0.6e-4,
        'sx_min':     1.5e-4,
        'sx_max':     6.0e-4,
        'cz_mean':    3.8e-3,
        'cz_std':     0.6e-3,
        'cz_min':     2.0e-3,
        'cz_max':     6.0e-3,
        'ro01_mean':  0.012,
        'ro01_std':   0.003,
        'ro01_min':   0.004,
        'ro01_max':   0.025,
        'ro10_mean':  0.003,
        'ro10_std':   0.001,
        'ro10_min':   0.0005,
        'ro10_max':   0.008,
    },
    'borderline': {
        # T1 dropped 10-25% from stable mean (175µs → ~130-158µs)
        # CZ risen 30-80% from stable mean
        # Fidelity values deliberately overlap with stable class
        # Label = drifted=1 (meets the >=15% T1 drop threshold)
        'T1_mean_us':  140.0,
        'T1_std_us':    25.0,
        'T1_min_us':    95.0,
        'T1_max_us':   175.0,   # upper bound touches stable lower bound
        'T2_mean_us':   95.0,
        'T2_std_us':    20.0,
        'T2_min_us':    55.0,
        'T2_max_us':   140.0,
        'sx_mean':    8.0e-4,
        'sx_std':     2.0e-4,
        'sx_min':     3.5e-4,
        'sx_max':     1.8e-3,
        'cz_mean':    7.5e-3,
        'cz_std':     1.5e-3,
        'cz_min':     4.0e-3,
        'cz_max':     1.2e-2,
        'ro01_mean':  0.028,
        'ro01_std':   0.008,
        'ro01_min':   0.012,
        'ro01_max':   0.060,
        'ro10_mean':  0.007,
        'ro10_std':   0.002,
        'ro10_min':   0.001,
        'ro10_max':   0.018,
    },
    'drifted': {
        # T1 dropped 30-50% (Carroll et al. 2022)
        # CZ up to 5x initial value (ISCA 2025)
        'T1_mean_us':   90.0,
        'T1_std_us':    25.0,
        'T1_min_us':    20.0,
        'T1_max_us':   125.0,   # tightened max — clearly degraded
        'T2_mean_us':   55.0,
        'T2_std_us':    18.0,
        'T2_min_us':    15.0,
        'T2_max_us':    95.0,
        'sx_mean':    2.8e-3,
        'sx_std':     0.8e-3,
        'sx_min':     1.2e-3,
        'sx_max':     6.0e-3,
        'cz_mean':    1.6e-2,
        'cz_std':     4.0e-3,
        'cz_min':     8.0e-3,
        'cz_max':     3.0e-2,
        'ro01_mean':  0.055,
        'ro01_std':   0.012,
        'ro01_min':   0.025,
        'ro01_max':   0.120,
        'ro10_mean':  0.015,
        'ro10_std':   0.004,
        'ro10_min':   0.003,
        'ro10_max':   0.040,
    }
}

def sample_params(regime):
    p = PARAM_RANGES[regime]
    T1_us = np.clip(np.random.normal(p['T1_mean_us'], p['T1_std_us']),
                    p['T1_min_us'], p['T1_max_us'])
    T2_us = np.clip(np.random.normal(p['T2_mean_us'], p['T2_std_us']),
                    p['T2_min_us'], p['T2_max_us'])
    T2_us = min(T2_us, 2.0 * T1_us)   # physical constraint

    sx_err = np.clip(np.random.normal(p['sx_mean'], p['sx_std']),
                     p['sx_min'], p['sx_max'])
    cz_err = np.clip(np.random.normal(p['cz_mean'], p['cz_std']),
                     p['cz_min'], p['cz_max'])
    ro01   = np.clip(np.random.normal(p['ro01_mean'], p['ro01_std']),
                     p['ro01_min'], p['ro01_max'])
    ro10   = np.clip(np.random.normal(p['ro10_mean'], p['ro10_std']),
                     p['ro10_min'], p['ro10_max'])
    ro10   = min(ro10, ro01 * 0.5)     # physical: P(0|1) >> P(1|0)

    return {
        'T1_s':   T1_us * 1e-6,
        'T2_s':   T2_us * 1e-6,
        'sx_err': sx_err,
        'cz_err': cz_err,
        'ro01':   ro01,
        'ro10':   ro10,
    }

# ── BUILD NOISE MODEL ─────────────────────────────────────────────────────────
def build_noise_model(params):
    nm = NoiseModel()
    T1 = params['T1_s']
    T2 = params['T2_s']

    try:
        thermal_sq = thermal_relaxation_error(T1, T2, T_SX)
    except Exception:
        thermal_sq = thermal_relaxation_error(T1, T1 * 0.9, T_SX)

    dep_sq      = depolarizing_error(min(2.0 * params['sx_err'], 0.99), 1)
    combined_sq = thermal_sq.compose(dep_sq)
    nm.add_all_qubit_quantum_error(combined_sq, ['h', 'x', 'sx'])

    try:
        thermal_tq = thermal_relaxation_error(T1, T2, T_CZ).expand(
                     thermal_relaxation_error(T1, T2, T_CZ))
    except Exception:
        T2s = min(T2, 1.99 * T1)
        thermal_tq = thermal_relaxation_error(T1, T2s, T_CZ).expand(
                     thermal_relaxation_error(T1, T2s, T_CZ))

    dep_tq      = depolarizing_error(min((4.0/3.0) * params['cz_err'], 0.99), 2)
    combined_tq = thermal_tq.compose(dep_tq)
    nm.add_all_qubit_quantum_error(combined_tq, ['cx', 'cz', 'ecr'])

    ro_error = ReadoutError([[1 - params['ro10'], params['ro10']],
                             [params['ro01'],     1 - params['ro01']]])
    nm.add_all_qubit_readout_error(ro_error)

    return nm

# ── GENERATE SAMPLES ──────────────────────────────────────────────────────────
BATCH_SIZE = 100

def generate_regime(n_samples, regime, label):
    print(f"\nGenerating {n_samples:,} {regime} samples (label={label})...")
    print(f"  Batches: {n_samples // BATCH_SIZE}")

    rows      = []
    n_batches = n_samples // BATCH_SIZE

    for batch_idx in range(n_batches):
        params = sample_params(regime)
        nm     = build_noise_model(params)
        sim    = AerSimulator(noise_model=nm)

        circuits = []
        for _ in range(BATCH_SIZE):
            circuits.extend([bell_qc, coh_qc, gate_qc])

        result = sim.run(circuits, shots=SHOTS).result()

        for i in range(BATCH_SIZE):
            idx = i * 3
            rows.append({
                'F_bell':      f_bell(result.get_counts(idx)),
                'F_coherence': f_coherence(result.get_counts(idx + 1)),
                'F_gate':      f_gate(result.get_counts(idx + 2)),
                'drifted':     label,
                '_T1_us':      params['T1_s'] * 1e6,
                '_T2_us':      params['T2_s'] * 1e6,
                '_sx_err':     params['sx_err'],
                '_cz_err':     params['cz_err'],
            })

        if (batch_idx + 1) % 10 == 0:
            completed = (batch_idx + 1) * BATCH_SIZE
            recent    = rows[-BATCH_SIZE * 10:]
            print(f"  [{completed:,}/{n_samples:,}] "
                  f"F_bell={np.mean([r['F_bell'] for r in recent]):.3f}  "
                  f"F_coh={np.mean([r['F_coherence'] for r in recent]):.3f}  "
                  f"F_gate={np.mean([r['F_gate'] for r in recent]):.3f}  "
                  f"T1={params['T1_s']*1e6:.0f}µs  CZ={params['cz_err']:.4f}")

    return rows

# ── RUN ───────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  QUANTUM CANARY — SIMULATED DATA GENERATION")
print("=" * 60)
print(f"  Stable     : {N_STABLE:,}  (label=0)")
print(f"  Borderline : {N_BORDERLINE:,}  (label=1) ← the hard ambiguous zone")
print(f"  Drifted    : {N_DRIFTED:,}  (label=1)")
print(f"  Total      : {N_STABLE + N_BORDERLINE + N_DRIFTED:,}")
print(f"  Shots/run  : {SHOTS:,}")
print(f"  Noise      : Thermal + Depolarizing + Asymmetric Readout")
print("=" * 60)

stable_rows     = generate_regime(N_STABLE,     'stable',     label=0)
borderline_rows = generate_regime(N_BORDERLINE, 'borderline', label=1)
drifted_rows    = generate_regime(N_DRIFTED,    'drifted',    label=1)

# ── COMBINE AND SAVE ──────────────────────────────────────────────────────────
all_rows = stable_rows + borderline_rows + drifted_rows
df_full  = pd.DataFrame(all_rows)
df_full  = df_full.sample(frac=1, random_state=42).reset_index(drop=True)

diag_cols = ['F_bell', 'F_coherence', 'F_gate', 'drifted',
             '_T1_us', '_T2_us', '_sx_err', '_cz_err']
df_full[diag_cols].to_csv('data/sim_data_diagnostics.csv', index=False)

feat_cols = ['F_bell', 'F_coherence', 'F_gate', 'drifted']
df_clean  = df_full[feat_cols].copy()
df_clean.to_csv('data/sim_data.csv', index=False)

# ── VALIDATION REPORT ─────────────────────────────────────────────────────────
s_df  = df_clean[df_clean['drifted'] == 0]
d_df  = df_clean[df_clean['drifted'] == 1]
bl_df = df_full[df_full['_T1_us'].between(
    PARAM_RANGES['borderline']['T1_min_us'],
    PARAM_RANGES['borderline']['T1_max_us']
)]

print(f"\n{'='*60}")
print(f"  GENERATION COMPLETE")
print(f"{'='*60}")
print(f"\n  Total : {len(df_clean):,}")
print(f"  Stable (0)  : {(df_clean['drifted']==0).sum():,}")
print(f"  Drifted (1) : {(df_clean['drifted']==1).sum():,}")
print(f"  Drift rate  : {round(df_clean['drifted'].mean()*100,1)}%")

print(f"\n  Feature means by class:")
print(df_clean.groupby('drifted')[['F_bell','F_coherence','F_gate']].mean().round(4))

print(f"\n  Borderline regime fidelity means (the hard zone):")
print(f"    F_bell      : {bl_df['F_bell'].mean():.4f}  "
      f"(stable: {s_df['F_bell'].mean():.4f})")
print(f"    F_coherence : {bl_df['F_coherence'].mean():.4f}  "
      f"(stable: {s_df['F_coherence'].mean():.4f})")
print(f"    F_gate      : {bl_df['F_gate'].mean():.4f}  "
      f"(stable: {s_df['F_gate'].mean():.4f})")

print(f"\n  Physical grounding:")
print(f"    Stable T1    : {PARAM_RANGES['stable']['T1_mean_us']}µs")
print(f"    Borderline T1: {PARAM_RANGES['borderline']['T1_mean_us']}µs  "
      f"({round((1-PARAM_RANGES['borderline']['T1_mean_us']/PARAM_RANGES['stable']['T1_mean_us'])*100)}% drop)")
print(f"    Drifted T1   : {PARAM_RANGES['drifted']['T1_mean_us']}µs  "
      f"({round((1-PARAM_RANGES['drifted']['T1_mean_us']/PARAM_RANGES['stable']['T1_mean_us'])*100)}% drop)")

print(f"\n  ✓ Saved: data/sim_data.csv")
print(f"  ✓ Saved: data/sim_data_diagnostics.csv")
print(f"\n  NEXT: python 2_label_data.py")
