# 1_pull_data.py — Quantum Canary Prototype 2
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Pull 200 days of raw IBM calibration data from 4 backends.
#   Target: ~8,000 rows (200 days × 10 qubits × 4 backends).
#
# COLUMNS COLLECTED:
#   timestamp     — date of calibration snapshot (YYYY-MM-DD)
#   backend       — which IBM QPU
#   qubit_id      — qubit number (0–9)
#   T1_us         — T1 relaxation time in microseconds
#   T2_us         — T2 dephasing time in microseconds
#   sx_error      — SX gate error rate
#   x_error       — X gate error rate
#   cz_error      — 2-qubit gate error (CZ / ECR / CX — whichever IBM has)
#   readout_error — measurement error probability
#   t_sx_ns       — SX gate duration in nanoseconds  ← needed for fidelity formulas
#   t_x_ns        — X gate duration in nanoseconds   ← needed for fidelity formulas
#   t_cz_ns       — 2Q gate duration in nanoseconds  ← needed for fidelity formulas
#
# WHY GATE DURATIONS:
#   The analytical fidelity estimators in 2_label_data.py use T2-decay terms
#   like e^(-(t_sx + t_cz)/T2). These require real gate durations from IBM.
#   This is the core improvement over Prototype 1.
#
# BACKENDS: ibm_kingston, ibm_fez, ibm_marrakesh, ibm_sherbrooke
#   ibm_torino is retired. ibm_sherbrooke is the 4th backend — swap if needed.
#
# OUTPUT:
#   data/ibm_kingston_calibration.csv
#   data/ibm_fez_calibration.csv
#   data/ibm_marrakesh_calibration.csv
#   data/ibm_sherbrooke_calibration.csv
#   data/all_backends_raw.csv   ← master file used by all downstream scripts
#
# SETUP:
#   echo "YOUR_IBM_TOKEN" > api_token.txt
#   echo "YOUR_CRN"       > crn_instance.txt
#   Add both to .gitignore — never commit credentials.
#
#   pip install qiskit-ibm-runtime pandas
#   python 1_pull_data.py
# ─────────────────────────────────────────────────────────────────────────────

from qiskit_ibm_runtime import QiskitRuntimeService
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)

# ── CREDENTIALS ───────────────────────────────────────────────────────────────
with open(SCRIPT_DIR / "api_token.txt",    "r") as f: token    = f.read().strip()
with open(SCRIPT_DIR / "crn_instance.txt", "r") as f: instance = f.read().strip()

service = QiskitRuntimeService(channel="ibm_cloud", token=token, instance=instance)
print("✓ Connected to IBM Quantum\n")

# ── CONFIG ────────────────────────────────────────────────────────────────────
BACKENDS = [
    "ibm_kingston",
    "ibm_fez",
    "ibm_marrakesh",
    "ibm_sherbrooke",   # swap for any other accessible backend if needed
]
DAYS   = 200   # days of history per backend
QUBITS = 10    # qubits 0–9 per snapshot

# ── HELPER ────────────────────────────────────────────────────────────────────
def gate_duration_ns(hist, gate, qargs):
    """Return gate duration in nanoseconds, or None if unavailable."""
    try:
        d = hist[gate][qargs].duration
        return round(d * 1e9, 4) if d is not None else None
    except Exception:
        return None

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
all_dfs = []

for backend_name in BACKENDS:
    print(f"{'─'*55}")
    print(f"  {backend_name}")
    print(f"{'─'*55}")

    try:
        backend = service.backend(backend_name)
    except Exception as e:
        print(f"  Could not connect: {e}\n  Skipping.\n")
        continue

    rows    = []
    skipped = 0

    for i in range(DAYS):
        date = datetime.now() - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")

        try:
            hist = backend.target_history(datetime=date)

            for q in range(QUBITS):

                # Coherence times — IBM stores in seconds, convert to µs
                t1 = hist.qubit_properties[q].t1
                t2 = hist.qubit_properties[q].t2

                # SX gate
                try:    sx_err = round(hist["sx"][(q,)].error, 8)
                except: sx_err = None
                t_sx = gate_duration_ns(hist, "sx", (q,))

                # X gate (falls back to SX values if unavailable)
                try:    x_err = round(hist["x"][(q,)].error, 8)
                except: x_err = sx_err
                t_x = gate_duration_ns(hist, "x", (q,))
                if t_x is None and t_sx is not None:
                    t_x = round(t_sx * 2, 4)   # X ≈ 2×SX on IBM hardware

                # Readout
                try:    ro_err = round(hist["measure"][(q,)].error, 8)
                except: ro_err = None

                # 2Q gate — try CZ → ECR → CX (varies by backend)
                cz_err = None
                t_cz   = None
                if q + 1 < QUBITS:
                    pair = (q, q + 1)
                    for g in ["cz", "ecr", "cx"]:
                        try:
                            cz_err = round(hist[g][pair].error, 8)
                            t_cz   = gate_duration_ns(hist, g, pair)
                            break
                        except Exception:
                            continue

                rows.append({
                    "timestamp":     date_str,
                    "backend":       backend_name,
                    "qubit_id":      q,
                    "T1_us":         round(t1 * 1e6, 4) if t1 else None,
                    "T2_us":         round(t2 * 1e6, 4) if t2 else None,
                    "sx_error":      sx_err,
                    "x_error":       x_err,
                    "cz_error":      cz_err,
                    "readout_error": ro_err,
                    "t_sx_ns":       t_sx,
                    "t_x_ns":        t_x,
                    "t_cz_ns":       t_cz,
                })

            print(f"  [{i+1:3}/{DAYS}] {date_str} ✓")

        except Exception as e:
            skipped += 1
            print(f"  [{i+1:3}/{DAYS}] {date_str} — skipped: {e}")

    # Save this backend
    df = pd.DataFrame(rows)
    path = DATA_DIR / f"{backend_name}_calibration.csv"
    df.to_csv(path, index=False)
    print(f"\n  ✓ {len(df)} rows → {path}  (skipped {skipped} days)\n")
    all_dfs.append(df)

# ── COMBINE ───────────────────────────────────────────────────────────────────
if not all_dfs:
    print("ERROR: No data collected. Check credentials and backend names.")
else:
    master = pd.concat(all_dfs, ignore_index=True)
    master.to_csv(DATA_DIR / "all_backends_raw.csv", index=False)

    print(f"{'='*55}")
    print(f"  DONE")
    print(f"{'='*55}")
    print(f"  Total rows : {len(master)}")
    print(f"  Backends   : {list(master['backend'].unique())}")
    print(f"  Date range : {master['timestamp'].min()} → {master['timestamp'].max()}")
    print(f"\n  Missing values:")
    print(master.isnull().sum().to_string())
    print(f"\n  ✓ Saved: data/all_backends_raw.csv")
    