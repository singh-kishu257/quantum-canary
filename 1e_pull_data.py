# 1e_pull_data.py — Quantum Canary Prototype 2
# Generates synthetic training data using Qiskit Aer.
# Builds noise models from scratch for full control over separation.
# Stable: typical IBM noise levels
# Drifted: severely degraded noise levels

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, depolarizing_error,
                               thermal_relaxation_error, ReadoutError)
import numpy as np
import pandas as pd
import os

os.makedirs('data', exist_ok=True)

SHOTS     = 1000
N_STABLE  = 5000
N_DRIFTED = 5000

# ── CANARY CIRCUITS ───────────────────────────────────────────────────────────
def make_bell():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc

def make_coherence():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.barrier()
    qc.h(0)
    qc.measure(0, 0)
    return qc

def make_gate_error():
    qc = QuantumCircuit(1, 1)
    for _ in range(8):
        qc.x(0)
    qc.measure(0, 0)
    return qc

bell_qc      = make_bell()
coherence_qc = make_coherence()
gate_qc      = make_gate_error()

# ── NOISE MODEL BUILDER ───────────────────────────────────────────────────────
# Stable:  typical healthy IBM backend noise levels
# Drifted: severely degraded — 5-10x worse, mimics real drift events
#
# Parameters chosen to match real hardware observations:
#   Stable   → F_bell ~0.95-0.99, F_coherence ~0.95-0.99, F_gate ~0.94-0.99
#   Drifted  → F_bell ~0.80-0.93, F_coherence ~0.75-0.93, F_gate ~0.75-0.92

NOISE_PARAMS = {
    'stable': {
        't1':           100e-6,   # 100 µs  — typical healthy T1
        't2':            80e-6,   #  80 µs  — typical healthy T2
        'sq_error':    0.0005,    # 0.05%   — single qubit gate error
        'tq_error':    0.005,     # 0.5%    — two qubit gate error
        'readout':     0.01,      # 1%      — readout error
        'gate_time_sq': 50e-9,    # 50 ns   — single qubit gate time
        'gate_time_tq': 300e-9,   # 300 ns  — two qubit gate time
    },
    'drifted': {
        't1':           30e-6,    #  30 µs  — severely degraded T1 (-70%)
        't2':           20e-6,    #  20 µs  — severely degraded T2 (-75%)
        'sq_error':    0.008,     # 0.8%    — 16x worse single qubit error
        'tq_error':    0.05,      # 5%      — 10x worse two qubit error
        'readout':     0.08,      # 8%      — 8x worse readout
        'gate_time_sq': 50e-9,
        'gate_time_tq': 300e-9,
    }
}

def build_noise_model(params):
    noise_model = NoiseModel()
    t1  = params['t1']
    t2  = params['t2']
    t2  = min(t2, 2 * t1)   # T2 <= 2*T1 always

    # Single qubit thermal relaxation
    thermal_sq = thermal_relaxation_error(t1, t2, params['gate_time_sq'])
    # Single qubit depolarizing
    dep_sq     = depolarizing_error(params['sq_error'], 1)
    # Combined single qubit error
    combined_sq = thermal_sq.compose(dep_sq)
    noise_model.add_all_qubit_quantum_error(combined_sq, ['h', 'x', 'sx'])

    # Two qubit thermal relaxation (each qubit)
    thermal_tq0 = thermal_relaxation_error(t1, t2, params['gate_time_tq'])
    thermal_tq1 = thermal_relaxation_error(t1, t2, params['gate_time_tq'])
    thermal_tq  = thermal_tq0.expand(thermal_tq1)
    # Two qubit depolarizing
    dep_tq      = depolarizing_error(params['tq_error'], 2)
    # Combined two qubit error
    combined_tq = thermal_tq.compose(dep_tq)
    noise_model.add_all_qubit_quantum_error(combined_tq, ['cx', 'ecr', 'cz'])

    # Readout error
    p_ro = params['readout']
    ro_error = ReadoutError([[1 - p_ro, p_ro],
                             [p_ro,     1 - p_ro]])
    noise_model.add_all_qubit_readout_error(ro_error)

    return noise_model

# ── FIDELITY EXTRACTION ───────────────────────────────────────────────────────
def f_bell(counts):
    total = sum(counts.values())
    return (counts.get('00', 0) + counts.get('11', 0)) / total

def f_coherence(counts):
    return counts.get('0', 0) / sum(counts.values())

def f_gate(counts):
    return counts.get('0', 0) / sum(counts.values())

# ── BATCH SIMULATE ────────────────────────────────────────────────────────────
def generate_samples(simulator, n, label):
    # Submit all circuits in one batch job
    circuits = []
    for _ in range(n):
        circuits.extend([bell_qc, coherence_qc, gate_qc])

    print(f'  Submitting {len(circuits):,} circuits in one batch job...')
    job     = simulator.run(circuits, shots=SHOTS)
    result  = job.result()
    print(f'  ✓ Job complete. Extracting fidelities...')

    rows = []
    for i in range(n):
        idx = i * 3
        fb  = f_bell(result.get_counts(idx))
        fc  = f_coherence(result.get_counts(idx + 1))
        fg  = f_gate(result.get_counts(idx + 2))
        rows.append({'F_bell': fb, 'F_coherence': fc, 'F_gate': fg, 'drifted': label})

    return rows

# ── GENERATE STABLE ───────────────────────────────────────────────────────────
print(f'Generating {N_STABLE:,} stable samples...')
print(f'  T1={NOISE_PARAMS["stable"]["t1"]*1e6:.0f}µs  '
      f'T2={NOISE_PARAMS["stable"]["t2"]*1e6:.0f}µs  '
      f'sq_err={NOISE_PARAMS["stable"]["sq_error"]}  '
      f'tq_err={NOISE_PARAMS["stable"]["tq_error"]}  '
      f'ro_err={NOISE_PARAMS["stable"]["readout"]}')

sim_stable  = AerSimulator(noise_model=build_noise_model(NOISE_PARAMS['stable']))
stable_rows = generate_samples(sim_stable, N_STABLE, label=0)

fb_s = np.mean([r['F_bell']       for r in stable_rows])
fc_s = np.mean([r['F_coherence']  for r in stable_rows])
fg_s = np.mean([r['F_gate']       for r in stable_rows])
print(f'  Stable means → F_bell={fb_s:.4f}  F_coh={fc_s:.4f}  F_gate={fg_s:.4f}\n')

# ── GENERATE DRIFTED ──────────────────────────────────────────────────────────
print(f'Generating {N_DRIFTED:,} drifted samples...')
print(f'  T1={NOISE_PARAMS["drifted"]["t1"]*1e6:.0f}µs  '
      f'T2={NOISE_PARAMS["drifted"]["t2"]*1e6:.0f}µs  '
      f'sq_err={NOISE_PARAMS["drifted"]["sq_error"]}  '
      f'tq_err={NOISE_PARAMS["drifted"]["tq_error"]}  '
      f'ro_err={NOISE_PARAMS["drifted"]["readout"]}')

sim_drifted  = AerSimulator(noise_model=build_noise_model(NOISE_PARAMS['drifted']))
drifted_rows = generate_samples(sim_drifted, N_DRIFTED, label=1)

fb_d = np.mean([r['F_bell']       for r in drifted_rows])
fc_d = np.mean([r['F_coherence']  for r in drifted_rows])
fg_d = np.mean([r['F_gate']       for r in drifted_rows])
print(f'  Drifted means → F_bell={fb_d:.4f}  F_coh={fc_d:.4f}  F_gate={fg_d:.4f}\n')

# ── SEPARATION CHECK ──────────────────────────────────────────────────────────
print(f'Separation (stable - drifted):')
print(f'  F_bell:      {fb_s - fb_d:.4f}')
print(f'  F_coherence: {fc_s - fc_d:.4f}')
print(f'  F_gate:      {fg_s - fg_d:.4f}')
if min(fb_s - fb_d, fc_s - fc_d, fg_s - fg_d) < 0.05:
    print('  WARNING: separation may be too small — consider increasing drift parameters')
else:
    print('  ✓ Separation looks good')

# ── SAVE ──────────────────────────────────────────────────────────────────────
df = pd.DataFrame(stable_rows + drifted_rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('data/synthetic_data.csv', index=False)

print(f'\n{"="*50}')
print(f'  DONE')
print(f'{"="*50}')
print(f'  Total samples : {len(df):,}')
print(f'  Stable  (0)   : {(df["drifted"]==0).sum():,}')
print(f'  Drifted (1)   : {(df["drifted"]==1).sum():,}')
print(f'\n  Feature means by class:')
print(df.groupby('drifted')[['F_bell','F_coherence','F_gate']].mean().round(4))
print(f'\n  ✓ Saved: data/synthetic_data.csv')
print(f'\n  NEXT: python 2_label_data.py')