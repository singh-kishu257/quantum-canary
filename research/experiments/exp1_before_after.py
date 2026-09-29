"""Experiment 1: before vs after Canary on Trotterized Ising dynamics (simulated Forte-like device).

Baselines per device instance (exact, infinite-shot evaluation):
  naive      identity assignment (logical q -> ion q), native gates
  random     mean fidelity over 25 random assignments (arbitrary static mapping)
  debias     equal mixture of the same 25 assignments (qubit-assignment symmetrization, averaging)
  oracle     best assignment found with the true error tables
Canary runs B batches of S shots on a shot-sampled device and is scored on
  canary_final      its final recommended assignment
  canary_late       trusted runs of the last half of batches
  canary_cumulative trusted runs of all batches (includes exploration)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]

import numpy as np
import pandas as pd

from quantum_canary import CanaryConfig, LogicalProgram
from bench.circuits import ising_trotter
from bench.device import ExactDevice, sample_device
from bench.runner import Evaluator, oracle_assignment, random_assignments, run_canary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=40)
    ap.add_argument("--ions", type=int, default=12)
    ap.add_argument("--qubits", type=int, default=6)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--outlier-fraction", type=float, default=0.05)
    ap.add_argument("--anneal-steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(ROOT / "research" / "data" / "exp1"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    program = LogicalProgram.from_circuit(ising_trotter(args.qubits, args.steps))
    master = np.random.default_rng(args.seed)
    rows, traces = [], []
    t0 = time.time()
    for inst in range(args.instances):
        rng = np.random.default_rng(master.integers(2**31))
        truth = sample_device(args.ions, rng, outlier_fraction=args.outlier_fraction)
        device = ExactDevice(truth, seed=int(rng.integers(2**31)))
        ev = Evaluator(program, device)
        naive = tuple(range(args.qubits))
        rand = random_assignments(args.qubits, range(args.ions), 25, rng)
        f_rand = [ev.fidelity(a) for a in rand]
        oracle, f_oracle = oracle_assignment(program, truth, ev, rng)
        cfg = CanaryConfig(seed=int(rng.integers(2**31)), anneal_steps=args.anneal_steps)
        tr = run_canary(program, device, ev, args.batches, args.shots, cfg)
        half = len(tr.trusted_parts) // 2
        bad = truth.bad_pairs()
        used_bad_naive = sum(1 for (a, b) in program.interaction_weights if (min(naive[a], naive[b]), max(naive[a], naive[b])) in bad)
        row = {
            "instance": inst,
            "n_bad_pairs_device": len(bad),
            "naive_uses_bad_pair": used_bad_naive,
            "naive": ev.fidelity(naive),
            "random": float(np.mean(f_rand)),
            "random_worst": float(np.min(f_rand)),
            "debias": ev.mixture_fidelity([(a, 1.0) for a in rand]),
            "oracle": f_oracle,
            "canary_final": tr.recommended_fidelity[-1],
            "canary_late": ev.mixture_fidelity(tr.trusted_parts[half:]),
            "canary_cumulative": ev.mixture_fidelity(tr.trusted_parts),
            "canary_all_runs": ev.mixture_fidelity(tr.all_parts),
            "algorithm_shots": tr.algorithm_shots,
            "probe_shots": tr.probe_shots,
            "final_degraded": len(tr.degraded[-1]),
        }
        rows.append(row)
        traces.append(
            {
                "instance": inst,
                "batch_trusted_fidelity": tr.batch_fidelity,
                "batch_exploit_fidelity": tr.batch_exploit_fidelity,
                "recommended_fidelity": tr.recommended_fidelity,
                "degraded": tr.degraded,
                "report": tr.canary.report(),
                "true_bad_pairs": sorted(bad),
            }
        )
        print(
            f"[{inst + 1}/{args.instances}] naive={row['naive']:.4f} rand={row['random']:.4f} "
            f"debias={row['debias']:.4f} canary={row['canary_final']:.4f} oracle={row['oracle']:.4f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "instances.csv", index=False)
    (out / "traces.json").write_text(json.dumps(traces, default=float))
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(df.describe().T.to_string())


if __name__ == "__main__":
    main()
