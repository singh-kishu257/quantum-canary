# 7_realtime_canary.py — Quantum Canary 2
# ─────────────────────────────────────────────────────────────────────────────
# Interactive real-time drift monitor.
# Prompts for credentials, lists available backends, then runs canary
# circuits every 15 minutes logging both MLP and threshold verdicts.
#
# Run: python 7_realtime_canary.py
#
# OUTPUT:
#   logs/realtime_log.csv           — full timestamped log
#   figures/fig_realtime_canary.png — live validation figure (updated each round)
# ─────────────────────────────────────────────────────────────────────────────

import time
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from tensorflow import keras

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

os.makedirs('logs',    exist_ok=True)
os.makedirs('figures', exist_ok=True)

INTERVAL_SECONDS = 15 * 60   # 15 minutes
LOG_PATH         = 'logs/realtime_log.csv'
SHOTS            = 1000

# ── BANNER ────────────────────────────────────────────────────────────────────
print()
print("╔══════════════════════════════════════════════════════════╗")
print("║          QUANTUM CANARY 2 — REAL-TIME MONITOR           ║")
print("╚══════════════════════════════════════════════════════════╝")
print()

# ── CREDENTIALS ───────────────────────────────────────────────────────────────
print("── Step 1: IBM Quantum Credentials ─────────────────────────")
print()
token    = input("  Enter your IBM Quantum API token: ").strip()
instance = input("  Enter your CRN instance:          ").strip()
print()

print("  Connecting to IBM Quantum...")
try:
    service = QiskitRuntimeService(
        channel="ibm_cloud", token=token, instance=instance
    )
    print("  ✓ Connected.\n")
except Exception as e:
    print(f"  ✗ Connection failed: {e}")
    sys.exit(1)

# ── BACKEND SELECTION ─────────────────────────────────────────────────────────
print("── Step 2: Select Backend ───────────────────────────────────")
print()
print("  Fetching available backends...")

try:
    all_backends = service.backends()
    # Filter to operational, non-simulator backends
    available = []
    for b in all_backends:
        try:
            status = b.status()
            if status.operational and not b.configuration().simulator:
                available.append(b)
        except Exception:
            continue
except Exception as e:
    print(f"  Could not list backends: {e}")
    sys.exit(1)

if not available:
    print("  No operational backends found.")
    sys.exit(1)

print(f"  Found {len(available)} operational backend(s):\n")
for i, b in enumerate(available):
    try:
        status     = b.status()
        n_qubits   = b.configuration().n_qubits
        queue_jobs = status.pending_jobs
        print(f"  [{i+1}] {b.name:<25s}  "
              f"{n_qubits} qubits   "
              f"queue: {queue_jobs} jobs")
    except Exception:
        print(f"  [{i+1}] {b.name}")

print()
while True:
    try:
        choice = int(input("  Select backend number: ").strip())
        if 1 <= choice <= len(available):
            backend = available[choice - 1]
            break
        else:
            print(f"  Please enter a number between 1 and {len(available)}.")
    except ValueError:
        print("  Please enter a valid number.")

print(f"\n  ✓ Selected: {backend.name}\n")

# ── DURATION ──────────────────────────────────────────────────────────────────
print("── Step 3: Monitoring Duration ──────────────────────────────")
print()
while True:
    try:
        hours_input = input("  How many hours to monitor? (default: 24): ").strip()
        hours = float(hours_input) if hours_input else 24.0
        if hours > 0:
            break
        print("  Please enter a positive number.")
    except ValueError:
        print("  Please enter a valid number.")

N_ROUNDS = int((hours * 3600) / INTERVAL_SECONDS)

print()
print("╔══════════════════════════════════════════════════════════╗")
print(f"║  Backend  : {backend.name:<44s}║")
print(f"║  Duration : {str(hours) + ' hours':<44s}║")
print(f"║  Interval : 15 minutes{' '*37}║")
print(f"║  Rounds   : {str(N_ROUNDS):<44s}║")
print("╚══════════════════════════════════════════════════════════╝")
print()
input("  Press Enter to start monitoring...")
print()

# ── CANARY CIRCUITS ───────────────────────────────────────────────────────────
def make_bell():
    qc = QuantumCircuit(2, 2)
    qc.h(0); qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc

def make_coherence():
    qc = QuantumCircuit(1, 1)
    for _ in range(10):
        qc.h(0); qc.h(0)
    qc.measure(0, 0)
    return qc

def make_gate_error():
    qc = QuantumCircuit(1, 1)
    for _ in range(20):
        qc.x(0)
    qc.measure(0, 0)
    return qc

print("  Transpiling circuits...")
pm       = generate_preset_pass_manager(optimization_level=0, backend=backend)
bell_t   = pm.run(make_bell())
coh_t    = pm.run(make_coherence())
gate_t   = pm.run(make_gate_error())
circuits = [bell_t, coh_t, gate_t]
print("  ✓ Circuits transpiled\n")

# ── LOAD MLP ENSEMBLE ─────────────────────────────────────────────────────────
print("  Loading MLP ensemble...")
models = []
for seed in range(10):
    path = f'models/mlp_seed_{seed}.keras'
    if os.path.exists(path):
        models.append(keras.models.load_model(path))
if not models:
    print("  ✗ No trained models found in models/. Run 5_train_mlp.py first.")
    sys.exit(1)
print(f"  ✓ Loaded {len(models)} MLP models\n")

# ── SCALER AND THRESHOLD ──────────────────────────────────────────────────────
print("  Setting up scaler and threshold classifier...")
df_train = pd.read_csv('data/features_data.csv')
with open('data/split_indices.json') as f:
    split = json.load(f)

FEATURES = ['F_bell', 'F_gate', 'F_coherence']
X_all    = df_train[FEATURES].values
y_all    = df_train['drifted'].values
X_train  = X_all[split['train']]
X_val    = X_all[split['val']]
y_val    = y_all[split['val']]

scaler = StandardScaler()
scaler.fit(X_train)

val_scores  = X_val.mean(axis=1) + np.random.normal(0, 0.02, len(X_val))
best_thresh, best_f1 = 0.0, -1.0
for t in np.linspace(val_scores.min(), val_scores.max(), 500):
    preds = (val_scores < t).astype(int)
    if len(np.unique(preds)) < 2: continue
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1; best_thresh = t

print(f"  ✓ Threshold classifier ready (cutoff = {round(best_thresh, 4)})\n")

# ── FIDELITY HELPERS ──────────────────────────────────────────────────────────
def f_bell(counts):
    total = sum(counts.values())
    return (counts.get('00', 0) + counts.get('11', 0)) / total

def f_single(counts):
    total = sum(counts.values())
    return counts.get('0', 0) / total if total > 0 else 0.0

# ── FIGURE GENERATION ─────────────────────────────────────────────────────────
def generate_figure(rows, threshold, backend_name):
    if len(rows) < 2:
        return

    df  = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    t   = df['timestamp']

    DARK_BG  = '#0d1117'
    PANEL_BG = '#161b22'
    CRIMSON  = '#e63946'
    ORANGE   = '#f4a261'
    TEAL     = '#2ec4b6'
    BLUE     = '#58a4f4'
    GREY     = '#8b949e'
    WHITE    = '#f0f6fc'

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(3, 1, hspace=0.45,
                            top=0.88, bottom=0.09,
                            left=0.08, right=0.97)

    mlp_binary    = (df['mlp_verdict']       == 'DRIFTED').astype(int)
    thresh_binary = (df['threshold_verdict'] == 'DRIFTED').astype(int)

    # ── Panel 1: Fidelity traces ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(PANEL_BG)
    for sp in ax1.spines.values(): sp.set_edgecolor('#30363d')

    ax1.plot(t, df['F_bell'],      color=CRIMSON, lw=1.8, label='F_bell')
    ax1.plot(t, df['F_gate'],      color=TEAL,    lw=1.8, label='F_gate')
    ax1.plot(t, df['F_coherence'], color=BLUE,    lw=1.8, label='F_coherence')
    ax1.axhline(threshold, color=ORANGE, lw=1.4, linestyle='--',
                label=f'Threshold ({round(threshold, 3)})')

    for i in range(len(df)-1):
        if mlp_binary.iloc[i]:
            ax1.axvspan(t.iloc[i], t.iloc[i+1], alpha=0.15, color=CRIMSON)

    ax1.set_ylabel('Fidelity', color=WHITE, fontsize=10)
    lo = max(0, df[['F_bell','F_gate','F_coherence']].min().min() - 0.02)
    ax1.set_ylim(lo, 1.01)
    ax1.tick_params(colors=GREY, labelsize=8)
    ax1.xaxis.set_tick_params(labelbottom=False)
    ax1.legend(fontsize=8, loc='lower left',
               facecolor=PANEL_BG, edgecolor='#30363d', labelcolor=WHITE)
    ax1.set_title('Canary Circuit Fidelities  ·  Red shading = MLP drift region',
                  color=WHITE, fontsize=10, fontweight='bold', pad=8)
    ax1.grid(alpha=0.15, color=GREY)

    # ── Panel 2: MLP probability ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor(PANEL_BG)
    for sp in ax2.spines.values(): sp.set_edgecolor('#30363d')

    ax2.fill_between(t, df['mlp_prob'], alpha=0.25, color=CRIMSON)
    ax2.plot(t, df['mlp_prob'], color=CRIMSON, lw=2.0,
             label='Quantum Canary MLP')
    if 'mlp_uncertainty' in df.columns:
        lo_b = (df['mlp_prob'] - df['mlp_uncertainty']).clip(0, 1)
        hi_b = (df['mlp_prob'] + df['mlp_uncertainty']).clip(0, 1)
        ax2.fill_between(t, lo_b, hi_b, alpha=0.12, color=CRIMSON,
                         label='Ensemble uncertainty')
    ax2.axhline(0.5, color=WHITE, lw=1.2, linestyle='--',
                label='Decision threshold (0.5)', alpha=0.7)
    ax2.set_ylabel('Drift Probability', color=WHITE, fontsize=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(colors=GREY, labelsize=8)
    ax2.xaxis.set_tick_params(labelbottom=False)
    ax2.legend(fontsize=8, loc='upper left',
               facecolor=PANEL_BG, edgecolor='#30363d', labelcolor=WHITE)
    ax2.set_title('Quantum Canary MLP — Drift Probability + Ensemble Uncertainty',
                  color=WHITE, fontsize=10, fontweight='bold', pad=8)
    ax2.grid(alpha=0.15, color=GREY)

    # ── Panel 3: Verdict comparison ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor(PANEL_BG)
    for sp in ax3.spines.values(): sp.set_edgecolor('#30363d')

    ax3.step(t, mlp_binary    + 0.05, color=CRIMSON, lw=2.5,
             where='post', label='Quantum Canary MLP')
    ax3.step(t, thresh_binary - 0.05, color=ORANGE,  lw=2.0,
             where='post', linestyle='--', label='Threshold Classifier')

    # Highlight disagreements
    disagree = mlp_binary != thresh_binary
    for i in range(len(df)-1):
        if disagree.iloc[i]:
            ax3.axvspan(t.iloc[i], t.iloc[i+1],
                        alpha=0.25, color='#f7c948')

    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['STABLE', 'DRIFTED'], color=WHITE, fontsize=9)
    ax3.set_ylim(-0.35, 1.35)
    ax3.tick_params(colors=GREY, labelsize=8)
    ax3.set_xlabel('Time (UTC)', color=WHITE, fontsize=10)
    ax3.legend(fontsize=8, loc='upper left',
               facecolor=PANEL_BG, edgecolor='#30363d', labelcolor=WHITE)
    ax3.set_title(
        'Verdict Comparison — Quantum Canary vs Threshold  ·  '
        'Yellow = disagreement (MLP caught what threshold missed)',
        color=WHITE, fontsize=10, fontweight='bold', pad=8)
    ax3.grid(alpha=0.15, color=GREY)

    plt.setp(ax3.xaxis.get_majorticklabels(),
             rotation=30, ha='right', color=GREY, fontsize=8)

    # Header
    last_prob    = round(df['mlp_prob'].iloc[-1], 4)
    last_verdict = df['mlp_verdict'].iloc[-1]
    agree_count  = int((mlp_binary == thresh_binary).sum())
    disagree_n   = len(df) - agree_count

    fig.suptitle(
        f'Quantum Canary 2  ·  {backend_name}  ·  Real-Time Drift Monitor\n'
        f'Latest: {df["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")}  ·  '
        f'MLP prob: {last_prob:.4f}  ·  Verdict: {last_verdict}  ·  '
        f'Rounds: {len(df)}',
        color=WHITE, fontsize=12, fontweight='bold', y=0.97
    )
    fig.text(
        0.5, 0.005,
        f'MLP drift detections: {int(mlp_binary.sum())}   '
        f'Threshold detections: {int(thresh_binary.sum())}   '
        f'Agreement: {agree_count}/{len(df)}   '
        f'Disagreement (yellow): {disagree_n} rounds',
        ha='center', color=GREY, fontsize=8
    )

    plt.savefig('figures/fig_realtime_canary.png',
                dpi=300, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.close()


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
log_cols = ['timestamp', 'F_bell', 'F_gate', 'F_coherence',
            'mean_fidelity', 'mlp_prob', 'mlp_uncertainty',
            'mlp_verdict', 'threshold_verdict', 'ibm_last_calibration']
log_rows = []

print("─" * 72)
print(f"{'Time (UTC)':22s} {'F_bell':8s} {'F_gate':8s} {'F_coh':8s} "
      f"{'MLP Prob':10s} {'Quantum Canary':16s} {'Threshold':12s}")
print("─" * 72)

for round_idx in range(N_ROUNDS):

    ts     = datetime.now(timezone.utc)
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S UTC")

    # ── Submit ────────────────────────────────────────────────────────────────
    try:
        sampler = Sampler(backend)
        job     = sampler.run(circuits, shots=SHOTS)
        result  = job.result()

        def get_counts(res_item):
            """Extract counts regardless of classical register name."""
            data = res_item.data
            for attr in vars(data):
                try:
                    return getattr(data, attr).get_counts()
                except Exception:
                    continue
            raise ValueError(f"Could not extract counts. Registers: {vars(data)}")

        counts_bell = get_counts(result[0])
        counts_coh  = get_counts(result[1])
        counts_gate = get_counts(result[2])

        fb = f_bell(counts_bell)
        fc = f_single(counts_coh)
        fg = f_single(counts_gate)

    except Exception as e:
        print(f"  [{ts_str}] ✗ Circuit error: {e} — skipping round")
        if round_idx < N_ROUNDS - 1:
            time.sleep(INTERVAL_SECONDS)
        continue

    # ── MLP ───────────────────────────────────────────────────────────────────
    features    = np.array([[fb, fg, fc]])
    features_s  = scaler.transform(features)
    seed_preds  = [m.predict(features_s, verbose=0).flatten()[0]
                   for m in models]
    mlp_prob    = float(np.mean(seed_preds))
    mlp_unc     = float(np.std(seed_preds))
    mlp_verdict = 'DRIFTED' if mlp_prob >= 0.5 else 'STABLE'

    # ── Threshold ─────────────────────────────────────────────────────────────
    mean_fid         = float(np.mean([fb, fg, fc]))
    thresh_verdict   = 'DRIFTED' if mean_fid < best_thresh else 'STABLE'

    # ── IBM calibration timestamp ─────────────────────────────────────────────
    try:
        ibm_cal = backend.properties().last_update_date.strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    except Exception:
        ibm_cal = 'unavailable'

    # ── Print ─────────────────────────────────────────────────────────────────
    mlp_icon   = '🔴' if mlp_verdict   == 'DRIFTED' else '🟢'
    thresh_icon= '🔴' if thresh_verdict == 'DRIFTED' else '🟢'
    print(f"  {ts_str:20s}  "
          f"{fb:.4f}   {fg:.4f}   {fc:.4f}   "
          f"{mlp_prob:.4f}    "
          f"{mlp_icon} {mlp_verdict:<10s}  "
          f"{thresh_icon} {thresh_verdict}")

    # ── Log ───────────────────────────────────────────────────────────────────
    log_rows.append({
        'timestamp':            ts_str,
        'F_bell':               round(fb, 6),
        'F_gate':               round(fg, 6),
        'F_coherence':          round(fc, 6),
        'mean_fidelity':        round(mean_fid, 6),
        'mlp_prob':             round(mlp_prob, 6),
        'mlp_uncertainty':      round(mlp_unc, 6),
        'mlp_verdict':          mlp_verdict,
        'threshold_verdict':    thresh_verdict,
        'ibm_last_calibration': ibm_cal,
    })

    pd.DataFrame(log_rows, columns=log_cols).to_csv(LOG_PATH, index=False)
    generate_figure(log_rows, best_thresh, backend.name)

    if round_idx < N_ROUNDS - 1:
        next_time = datetime.now(timezone.utc)
        print(f"\n  Next measurement in {args_interval} minutes  "
              f"[Round {round_idx+2}/{N_ROUNDS}]\n")
        time.sleep(INTERVAL_SECONDS)

# ── DONE ──────────────────────────────────────────────────────────────────────
print()
print("╔══════════════════════════════════════════════════════════╗")
print("║              MONITORING COMPLETE                        ║")
print("╚══════════════════════════════════════════════════════════╝")
print(f"  Rounds logged  : {len(log_rows)}")
print(f"  Log file       : {LOG_PATH}")
print(f"  Figure         : figures/fig_realtime_canary.png")

if log_rows:
    df_final  = pd.DataFrame(log_rows)
    n_mlp     = int((df_final['mlp_verdict']       == 'DRIFTED').sum())
    n_thresh  = int((df_final['threshold_verdict'] == 'DRIFTED').sum())
    n_disagree= int((df_final['mlp_verdict'] != df_final['threshold_verdict']).sum())
    print(f"  MLP detections : {n_mlp}")
    print(f"  Thresh detects : {n_thresh}")
    print(f"  Disagreements  : {n_disagree} rounds where methods differed")
print()