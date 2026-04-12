# 1e_pull_data.py — Quantum Canary Prototype 2
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Generate synthetic training data using Qiskit Aer noise models.
#   Produces 5,000 stable + 5,000 drifted samples from simulated canary circuits.
#
# APPROACH:
#   Stable  → AerSimulator with noise model from real IBM backend calibration
#   Drifted → Same but with manually degraded noise parameters
#
# CIRCUITS:
#   1. Bell State:   H(0), CX(0,1), measure → F_bell
#   2. Coherence:    H(0), barrier, H(0), measure → F_coherence
#   3. Gate Error:   X*8, measure → F_gate
#
# OUTPUT:
#   data/synthetic_data.csv
#   Columns: F_bell, F_coherence, F_gate, drifted
#
# SETUP:
#   pip install qiskit qiskit-aer qiskit-ibm-runtime
#   python 1e_pull_data.py
# ─────────────────────────────────────────────────────────────────────────────

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np
import pandas as pd
import os

os.makedirs('data', exist_ok=True)

SHOTS        = 1000
N_STABLE     = 5000
N_DRIFTED    = 5000
BACKEND_NAME = 'ibm_torino'  # change to any accessible backend

# ── CREDENTIALS ───────────────────────────────────────────────────────────────
with open('api_token.txt',    'r') as f: token    = f.read().strip()
with open('crn_instance.txt', 'r') as f: instance = f.read().strip()

service = QiskitRuntimeService(channel='ibm_cloud', token=token, instance=instance)
print(f'✓ Connected to IBM Quantum')

backend = service.backend(BACKEND_NAME)
print(f'✓ Loaded backend: {BACKEND_NAME}\n')

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

# ── FIDELITY EXTRACTION ───────────────────────────────────────────────────────
def f_bell(counts):
    total   = sum(counts.values())
    correct = counts.get('00', 0) + counts.get('11', 0)
    return correct / total

def f_coherence(counts):
    total = sum(counts.values())
    return counts.get('0', 0) / total

def f_gate(counts):
    total = sum(counts.values())
    return counts.get('0', 0) / total

# ── NOISE MODEL BUILDER ───────────────────────────────────────────────────────
def build_noise_model(backend, drift=False):
    """
    Build a noise model from real backend properties.
    If drift=True, degrade the noise parameters to simulate hardware drift:
      - Gate errors increased by 50%
      - T1 decreased by 30%
      - T2 decreased by 40%
      - Added depolarizing error on all single-qubit gates
    """
    noise_model = NoiseModel.from_backend(backend)

    if drift:
        props = backend.properties()

        for qubit in range(min(5, backend.num_qubits)):
            # Degraded T1 and T2
            t1 = props.t1(qubit) * 0.70   # 30% T1 reduction
            t2 = props.t2(qubit) * 0.60   # 40% T2 reduction
            t2 = min(t2, 2 * t1)          # T2 <= 2*T1 always

            # Gate time ~50ns for single qubit gates
            gate_time = 50e-9

            # Thermal relaxation on single qubit gates
            thermal_err = thermal_relaxation_error(t1, t2, gate_time)
            noise_model.add_quantum_error(thermal_err, ['h', 'x'], [qubit])

            # Extra depolarizing error on top (gate pulse miscalibration)
            sx_error = props.gate_error('sx', qubit)
            dep_err  = depolarizing_error(min(sx_error * 1.5, 0.15), 1)
            noise_model.add_quantum_error(dep_err, ['h', 'x'], [qubit])

        # Degrade 2-qubit gate errors
        for gate in props.gates:
            if gate.gate == 'cx' or gate.gate == 'ecr' or gate.gate == 'cz':
                if len(gate.qubits) == 2:
                    q0, q1 = gate.qubits
                    original_err = props.gate_error(gate.gate, gate.qubits)
                    degraded_err = min(original_err * 1.5, 0.20)
                    dep_err_2q   = depolarizing_error(degraded_err, 2)
                    noise_model.add_quantum_error(
                        dep_err_2q, [gate.gate], [q0, q1]
                    )

    return noise_model

# ── SIMULATE ONE SAMPLE ───────────────────────────────────────────────────────
def simulate_sample(simulator):
    results = {}
    for name, qc in [('bell', bell_qc), ('coherence', coherence_qc), ('gate', gate_qc)]:
        job    = simulator.run(qc, shots=SHOTS)
        counts = job.result().get_counts()
        results[name] = counts
    return (
        f_bell(results['bell']),
        f_coherence(results['coherence']),
        f_gate(results['gate'])
    )

# ── GENERATE STABLE SAMPLES ───────────────────────────────────────────────────
print(f'Generating {N_STABLE} stable samples...')
noise_stable  = build_noise_model(backend, drift=False)
sim_stable    = AerSimulator(noise_model=noise_stable)

stable_rows = []
for i in range(N_STABLE):
    fb, fc, fg    = simulate_sample(sim_stable)
    stable_rows.append({'F_bell': fb, 'F_coherence': fc, 'F_gate': fg, 'drifted': 0})
    if (i + 1) % 500 == 0:
        print(f'  [{i+1:,}/{N_STABLE:,}] F_bell={fb:.3f} F_coh={fc:.3f} F_gate={fg:.3f}')

print(f'✓ Stable samples done\n')

# ── GENERATE DRIFTED SAMPLES ──────────────────────────────────────────────────
print(f'Generating {N_DRIFTED} drifted samples...')
noise_drifted = build_noise_model(backend, drift=True)
sim_drifted   = AerSimulator(noise_model=noise_drifted)

drifted_rows = []
for i in range(N_DRIFTED):
    fb, fc, fg     = simulate_sample(sim_drifted)
    drifted_rows.append({'F_bell': fb, 'F_coherence': fc, 'F_gate': fg, 'drifted': 1})
    if (i + 1) % 500 == 0:
        print(f'  [{i+1:,}/{N_DRIFTED:,}] F_bell={fb:.3f} F_coh={fc:.3f} F_gate={fg:.3f}')

print(f'✓ Drifted samples done\n')

# ── COMBINE AND SAVE ──────────────────────────────────────────────────────────
df = pd.DataFrame(stable_rows + drifted_rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
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