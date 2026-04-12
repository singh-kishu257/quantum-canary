# 1e_pull_data.py — Quantum Canary Prototype 2
# Generates synthetic training data using Qiskit Aer.
# Batches all circuits into 2 jobs (stable + drifted) for speed.

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np
import pandas as pd
import os

os.makedirs('data', exist_ok=True)

SHOTS    = 1000
N_STABLE  = 5000
N_DRIFTED = 5000

# ── CREDENTIALS ───────────────────────────────────────────────────────────────
with open('api_token.txt',    'r') as f: token    = f.read().strip()
with open('crn_instance.txt', 'r') as f: instance = f.read().strip()

service = QiskitRuntimeService(channel='ibm_cloud', token=token, instance=instance)
print('✓ Connected to IBM Quantum')

# Pick first available backend
available = [b.name for b in service.backends()]
print(f'  Available backends: {available}')
BACKEND_NAME = available[0]
backend = service.backend(BACKEND_NAME)
print(f'✓ Using backend: {BACKEND_NAME}\n')

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
def build_noise_model(backend, drift=False):
    noise_model = NoiseModel.from_backend(backend)

    if drift:
        props = backend.properties()
        for qubit in range(min(5, backend.num_qubits)):
            t1 = props.t1(qubit) * 0.70
            t2 = props.t2(qubit) * 0.60
            t2 = min(t2, 2 * t1)

            thermal_err = thermal_relaxation_error(t1, t2, 50e-9)
            noise_model.add_quantum_error(thermal_err, ['h', 'x'], [qubit])

            sx_error = props.gate_error('sx', qubit)
            dep_err  = depolarizing_error(min(sx_error * 1.5, 0.15), 1)
            noise_model.add_quantum_error(dep_err, ['h', 'x'], [qubit])

        for gate in props.gates:
            if gate.gate in ['cx', 'ecr', 'cz'] and len(gate.qubits) == 2:
                q0, q1       = gate.qubits
                original_err = props.gate_error(gate.gate, gate.qubits)
                degraded_err = min(original_err * 1.5, 0.20)
                dep_err_2q   = depolarizing_error(degraded_err, 2)
                noise_model.add_quantum_error(dep_err_2q, [gate.gate], [q0, q1])

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
    # Build one big list: [bell, coherence, gate, bell, coherence, gate, ...]
    # 3 circuits per sample, all submitted in a single job
    circuits = []
    for _ in range(n):
        circuits.extend([bell_qc, coherence_qc, gate_qc])

    print(f'  Submitting {len(circuits):,} circuits in one batch job...')
    job     = simulator.run(circuits, shots=SHOTS)
    results = job.result()
    print(f'  ✓ Job complete. Extracting fidelities...')

    rows = []
    for i in range(n):
        idx    = i * 3
        fb     = f_bell(results.get_counts(idx))
        fc     = f_coherence(results.get_counts(idx + 1))
        fg     = f_gate(results.get_counts(idx + 2))
        rows.append({'F_bell': fb, 'F_coherence': fc, 'F_gate': fg, 'drifted': label})

    return rows

# ── GENERATE ──────────────────────────────────────────────────────────────────
print(f'Generating {N_STABLE:,} stable samples...')
noise_stable = build_noise_model(backend, drift=False)
sim_stable   = AerSimulator(noise_model=noise_stable)
stable_rows  = generate_samples(sim_stable, N_STABLE, label=0)
print(f'  F_bell={np.mean([r["F_bell"] for r in stable_rows]):.3f}  '
      f'F_coh={np.mean([r["F_coherence"] for r in stable_rows]):.3f}  '
      f'F_gate={np.mean([r["F_gate"] for r in stable_rows]):.3f}\n')

print(f'Generating {N_DRIFTED:,} drifted samples...')
noise_drifted = build_noise_model(backend, drift=True)
sim_drifted   = AerSimulator(noise_model=noise_drifted)
drifted_rows  = generate_samples(sim_drifted, N_DRIFTED, label=1)
print(f'  F_bell={np.mean([r["F_bell"] for r in drifted_rows]):.3f}  '
      f'F_coh={np.mean([r["F_coherence"] for r in drifted_rows]):.3f}  '
      f'F_gate={np.mean([r["F_gate"] for r in drifted_rows]):.3f}\n')

# ── SAVE ──────────────────────────────────────────────────────────────────────
df = pd.DataFrame(stable_rows + drifted_rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('data/synthetic_data.csv', index=False)

print(f'{"="*50}')
print(f'  DONE')
print(f'{"="*50}')
print(f'  Total samples : {len(df):,}')
print(f'  Stable  (0)   : {(df["drifted"]==0).sum():,}')
print(f'  Drifted (1)   : {(df["drifted"]==1).sum():,}')
print(f'\n  Feature means by class:')
print(df.groupby('drifted')[['F_bell','F_coherence','F_gate']].mean().round(4))
print(f'\n  ✓ Saved: data/synthetic_data.csv')
print(f'\n  NEXT: python 2_label_data.py')