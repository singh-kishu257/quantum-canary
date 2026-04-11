# 1d_pull_data.py — Quantum Canary Prototype 2 (Qubits 20-39)
# Pulls qubits 20-39 from 3 backends, same 200-day window.
# Zero overlap with previous pulls (qubits 0-9, 10-19).
# Target: ~12,000 new rows

from qiskit_ibm_runtime import QiskitRuntimeService
from datetime import datetime, timedelta
import pandas as pd
import os

os.makedirs('data', exist_ok=True)

with open("api_token.txt",    "r") as f: token    = f.read().strip()
with open("crn_instance.txt", "r") as f: instance = f.read().strip()

service = QiskitRuntimeService(channel="ibm_cloud", token=token, instance=instance)
print("Connected to IBM Quantum\n")

BACKENDS    = ["ibm_kingston", "ibm_fez", "ibm_marrakesh"]
DAYS        = 200
QUBIT_START = 20
QUBIT_END   = 40   # qubits 20-39

def gate_duration_ns(hist, gate, qargs):
    try:
        d = hist[gate][qargs].duration
        return round(d * 1e9, 4) if d is not None else None
    except: return None

all_dfs = []

for backend_name in BACKENDS:
    print(f"{'─'*50}")
    print(f"  {backend_name}  (qubits {QUBIT_START}-{QUBIT_END-1})")
    print(f"{'─'*50}")

    try:
        backend = service.backend(backend_name)
    except Exception as e:
        print(f"  Could not connect: {e}\n  Skipping.\n")
        continue

    rows = []; skipped = 0

    for i in range(DAYS):
        date     = datetime.now() - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        try:
            hist = backend.target_history(datetime=date)
            for q in range(QUBIT_START, QUBIT_END):
                t1 = hist.qubit_properties[q].t1
                t2 = hist.qubit_properties[q].t2
                try:    sx_err = round(hist["sx"][(q,)].error, 8)
                except: sx_err = None
                t_sx = gate_duration_ns(hist, "sx", (q,))
                try:    x_err = round(hist["x"][(q,)].error, 8)
                except: x_err = sx_err
                t_x = gate_duration_ns(hist, "x", (q,))
                if t_x is None and t_sx is not None: t_x = round(t_sx*2, 4)
                try:    ro_err = round(hist["measure"][(q,)].error, 8)
                except: ro_err = None
                cz_err = None; t_cz = None
                if q+1 < QUBIT_END:
                    pair = (q, q+1)
                    for g in ["cz","ecr","cx"]:
                        try:
                            cz_err = round(hist[g][pair].error, 8)
                            t_cz   = gate_duration_ns(hist, g, pair)
                            break
                        except: continue
                rows.append({"timestamp": date_str, "backend": backend_name,
                    "qubit_id": q,
                    "T1_us": round(t1*1e6,4) if t1 else None,
                    "T2_us": round(t2*1e6,4) if t2 else None,
                    "sx_error": sx_err, "x_error": x_err,
                    "cz_error": cz_err, "readout_error": ro_err,
                    "t_sx_ns": t_sx, "t_x_ns": t_x, "t_cz_ns": t_cz})
            print(f"  [{i+1:3}/{DAYS}] {date_str} ✓")
        except Exception as e:
            skipped += 1
            print(f"  [{i+1:3}/{DAYS}] {date_str} — skipped: {e}")

    df = pd.DataFrame(rows)
    path = f"data/{backend_name}_qubits20-39.csv"
    df.to_csv(path, index=False)
    print(f"\n  {len(df)} rows -> {path}  (skipped {skipped})\n")
    all_dfs.append(df)

if all_dfs:
    new_data = pd.concat(all_dfs, ignore_index=True)
    master   = pd.read_csv("data/all_backends_raw.csv")
    combined = pd.concat([master, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp","backend","qubit_id"]).reset_index(drop=True)
    combined.to_csv("data/all_backends_raw.csv", index=False)
    print(f"{'='*50}")
    print(f"  New rows: {len(new_data):,}")
    print(f"  Total:    {len(combined):,}")
    print(f"  NEXT: python 2_label_data.py")