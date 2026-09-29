"""Experiment 4: algorithm families. Canary versus baselines on Ising dynamics and QAOA of different depths."""

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
from bench.circuits import ising_trotter, qaoa_ring
from bench.device import ExactDevice, sample_device
from bench.runner import Evaluator, oracle_assignment, random_assignments, run_canary


def workloads(n: int) -> dict:
    return {
        "ising_s2": ising_trotter(n, 2),
        "ising_s8": ising_trotter(n, 8),
        "qaoa_p1": qaoa_ring(n, 1),
        "qaoa_p4": qaoa_ring(n, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=12)
    ap.add_argument("--ions", type=int, default=12)
    ap.add_argument("--qubits", type=int, default=6)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--anneal-steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", default=str(ROOT / "research" / "data" / "exp4"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    master = np.random.default_rng(args.seed)
    devices = [int(master.integers(2**31)) for _ in range(args.instances)]
    rows = []
    t0 = time.time()
    for name, circ in workloads(args.qubits).items():
        program = LogicalProgram.from_circuit(circ)
        for inst, dseed in enumerate(devices):
            rng = np.random.default_rng(dseed)
            truth = sample_device(args.ions, rng, outlier_fraction=0.05)
            device = ExactDevice(truth, seed=int(rng.integers(2**31)))
            ev = Evaluator(program, device)
            rand = random_assignments(args.qubits, range(args.ions), 25, rng)
            f_rand = [ev.fidelity(a) for a in rand]
            _, f_oracle = oracle_assignment(program, truth, ev, rng, restarts=2)
            cfg = CanaryConfig(seed=int(rng.integers(2**31)), anneal_steps=args.anneal_steps)
            tr = run_canary(program, device, ev, args.batches, args.shots, cfg)
            rows.append(
                {
                    "workload": name,
                    "instance": inst,
                    "rzz_total": program.two_qubit_count,
                    "max_pair_reuse": max(program.interaction_weights.values()),
                    "naive": ev.fidelity(tuple(range(args.qubits))),
                    "random": float(np.mean(f_rand)),
                    "debias": ev.mixture_fidelity([(a, 1.0) for a in rand]),
                    "canary_final": tr.recommended_fidelity[-1],
                    "canary_cumulative": ev.mixture_fidelity(tr.trusted_parts),
                    "oracle": f_oracle,
                }
            )
        print(f"{name} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "instances.csv", index=False)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(df.groupby("workload")[["naive", "random", "debias", "canary_final", "oracle"]].mean().to_string())


if __name__ == "__main__":
    main()
