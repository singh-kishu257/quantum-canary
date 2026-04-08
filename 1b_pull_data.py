# 1b_pull_data.py — Quantum Canary Prototype 2 (Supplemental)
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Pull an additional ~3,000 rows from 3 backends using OLDER dates
#   (days 201–333 ago) that are completely non-overlapping with 1_pull_data.py.
#
#   1_pull_data.py  → days 0–199   (recent)
#   1b_pull_data.py → days 201–333 (older, ~133 days back)
#
#   133 days × 10 qubits × 3 backends = ~3,990 rows attempted
#   Expect ~1,000 per backend after skipped days → ~3,000 usable rows
#
#   Combined with 1_pull_data.py: ~9,000 rows total (target was 8,000 ✓)
#
# SAME COLUMNS AS 1_pull_data.py — appends cleanly to all_backends_raw.csv
#
# SETUP: Requires api_token.txt and crn_instance.txt (same as script 1)
#   python 1b_pull_data.py
# ─────────────────────────────────────────────────────────────────────────────

from qiskit_ibm_runtime import QiskitRuntimeService
from datetime import datetime, timedelta
import pandas as pd
import os

os.makedirs('data', exist_ok=True)

# ── CREDENTIALS ───────────────────────────────────────────────────────────────
with open("api_token.txt",    "r") as f: token    = f.read().strip()
with open("crn_instance.txt", "r") as f: instance = f.read().strip()

service = QiskitRuntimeService(channel="ibm_cloud", token=token, instance=instance)
print("✓ Connected to IBM Quantum\n")

# ── CONFIG ────────────────────────────────────────────────────────────────────
BACKENDS   = ["ibm_kingston", "ibm_fez", "ibm_marrakesh"]
DAY_START  = 201   # start here so no overlap with 1_pull_data.py (days 0–199)
DAY_END    = 334   # 133 days × 10 qubits × 3 backends ≈ 3,990 rows attempted
QUBITS     = 10    # same qubits 0–9 — different dates = different data

# ── HELPER ────────────────────────────────────────────────────────────────────
def gate_duration_ns(hist, gate, qargs):
    try:
        d = hist[gate][qargs].duration
        return round(d * 1e9, 4) if d is not None else None
    except Exception:
        return None

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
all_dfs = []

for backend_name in BACKENDS:
    print(f"{'─'*55}")
    print(f"  {backend_name}  (days {DAY_START}–{DAY_END-1} ago)")
    print(f"{'─'*55}")

    try:
        backend = service.backend(backend_name)
    except Exception as e:
        print(f"  Could not connect: {e}\n  Skipping.\n")
        continue

    rows    = []
    skipped = 0
    total   = DAY_END - DAY_START

    for i, day in enumerate(range(DAY_START, DAY_END)):
        date     = datetime.now() - timedelta(days=day)
        date_str = date.strftime("%Y-%m-%d")

        try:
            hist = backend.target_history(datetime=date)

            for q in range(QUBITS):

                # Coherence times
                t1 = hist.qubit_properties[q].t1
                t2 = hist.qubit_properties[q].t2

                # SX gate
                try:    sx_err = round(hist["sx"][(q,)].error, 8)
                except: sx_err = None
                t_sx = gate_duration_ns(hist, "sx", (q,))

                # X gate
                try:    x_err = round(hist["x"][(q,)].error, 8)
                except: x_err = sx_err
                t_x = gate_duration_ns(hist, "x", (q,))
                if t_x is None and t_sx is not None:
                    t_x = round(t_sx * 2, 4)

                # Readout
                try:    ro_err = round(hist["measure"][(q,)].error, 8)
                except: ro_err = None

                # 2Q gate: CZ → ECR → CX
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

            print(f"  [{i+1:3}/{total}] {date_str} ✓")

        except Exception as e:
            skipped += 1
            print(f"  [{i+1:3}/{total}] {date_str} — skipped: {e}")

    df = pd.DataFrame(rows)
    path = f"data/{backend_name}_calibration_extended.csv"
    df.to_csv(path, index=False)
    print(f"\n  ✓ {len(df)} rows → {path}  (skipped {skipped} days)\n")
    all_dfs.append(df)

# ── APPEND TO MASTER FILE ─────────────────────────────────────────────────────
if not all_dfs:
    print("ERROR: No data collected.")
else:
    new_data = pd.concat(all_dfs, ignore_index=True)

    master_path = "data/all_backends_raw.csv"
    if os.path.exists(master_path):
        existing = pd.read_csv(master_path)
        combined = pd.concat([existing, new_data], ignore_index=True)

        # Deduplicate on (timestamp, backend, qubit_id) — safety check
        before = len(combined)
        combined = combined.drop_duplicates(
            subset=["timestamp", "backend", "qubit_id"]
        ).reset_index(drop=True)
        dupes = before - len(combined)
        if dupes > 0:
            print(f"  Removed {dupes} duplicate rows (safety check).")
    else:
        combined = new_data

    combined.to_csv(master_path, index=False)

    print(f"{'='*55}")
    print(f"  DONE")
    print(f"{'='*55}")
    print(f"  New rows added : {len(new_data)}")
    print(f"  Total rows now : {len(combined)}")
    print(f"  Backends       : {list(combined['backend'].unique())}")
    print(f"  Date range     : {combined['timestamp'].min()} → {combined['timestamp'].max()}")
    print(f"\n  ✓ Updated: data/all_backends_raw.csv")
    