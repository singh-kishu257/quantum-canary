"""Experiment 2: when does Canary help? Gain versus pair reuse (Trotter steps) and versus pair heterogeneity.

Sweep A: Trotter steps s in {1, 2, 4, 8, 12} at outlier fraction 0.05.
Sweep B: outlier fraction in {0, 0.02, 0.05, 0.10} at s = 8.
For every instance we also record the error-model prediction of the log-fidelity gain,
Delta = sum_pairs W (x_random - x_canary) + sum_ions (x_random - x_canary), averaged over random assignments.
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

from quantum_canary import AssignmentSolver, CanaryConfig, LogicalProgram
from bench.circuits import ising_trotter
from bench.device import ExactDevice, sample_device
from bench.runner import Evaluator, oracle_assignment, random_assignments, run_canary


def predicted_log_fidelity(program: LogicalProgram, truth, assignment) -> float:
    wp, wi = program.weights()
    iv, pv = truth.damping_tables()
    perm = np.asarray(assignment)
    return -AssignmentSolver.cost(wp, wi, pv, iv, perm)


def run_point(steps: int, outlier: float, instances: int, args, master: np.random.Generator, tag: str) -> list[dict]:
    program = LogicalProgram.from_circuit(ising_trotter(args.qubits, steps))
    rows = []
    for inst in range(instances):
        rng = np.random.default_rng(master.integers(2**31))
        truth = sample_device(args.ions, rng, outlier_fraction=outlier)
        device = ExactDevice(truth, seed=int(rng.integers(2**31)))
        ev = Evaluator(program, device)
        rand = random_assignments(args.qubits, range(args.ions), 25, rng)
        f_rand = [ev.fidelity(a) for a in rand]
        _, f_oracle = oracle_assignment(program, truth, ev, rng, restarts=2)
        cfg = CanaryConfig(seed=int(rng.integers(2**31)), anneal_steps=args.anneal_steps)
        tr = run_canary(program, device, ev, args.batches, args.shots, cfg)
        final = tr.recommended[-1]
        pred_rand = float(np.mean([predicted_log_fidelity(program, truth, a) for a in rand]))
        pred_can = predicted_log_fidelity(program, truth, final)
        rows.append(
            {
                "sweep": tag,
                "steps": steps,
                "outlier_fraction": outlier,
                "instance": inst,
                "rzz_per_pair": steps,
                "random": float(np.mean(f_rand)),
                "debias": ev.mixture_fidelity([(a, 1.0) for a in rand]),
                "canary_final": tr.recommended_fidelity[-1],
                "canary_cumulative": ev.mixture_fidelity(tr.trusted_parts),
                "oracle": f_oracle,
                "measured_log_gain": float(np.log(tr.recommended_fidelity[-1]) - np.mean(np.log(f_rand))),
                "predicted_log_gain": pred_can - pred_rand,
                "probe_shots": tr.probe_shots,
                "algorithm_shots": tr.algorithm_shots,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=12)
    ap.add_argument("--ions", type=int, default=12)
    ap.add_argument("--qubits", type=int, default=6)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--anneal-steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "research" / "data" / "exp2"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    master = np.random.default_rng(args.seed)
    rows: list[dict] = []
    t0 = time.time()
    for s in (1, 2, 4, 8, 12):
        rows += run_point(s, 0.05, args.instances, args, master, "reuse")
        print(f"reuse s={s} done ({time.time() - t0:.0f}s)", flush=True)
    for f in (0.0, 0.02, 0.10):
        rows += run_point(8, f, args.instances, args, master, "heterogeneity")
        print(f"heterogeneity f={f} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "instances.csv", index=False)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(df.groupby(["sweep", "steps", "outlier_fraction"])[["random", "debias", "canary_final", "oracle"]].mean().to_string())


if __name__ == "__main__":
    main()
