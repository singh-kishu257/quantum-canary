"""Experiment 3: a pair degrades mid-run. Detection latency and recovery versus static assignments.

Before batch E, one ion pair jumps to error EVENT_ERROR. --target static hits the most exposed pair of
the static best map (a pair a fixed calibration-based map depends on); --target canary hits the most
exposed pair of Canary's current recommendation (adversarial to Canary). Static baselines are fixed
from the pre-event truth:
  static_oracle   best assignment with perfect pre-event knowledge, never updated
  debias          fixed mixture of 25 random assignments
We report, per batch, the exact fidelity of what each method executes under the current truth.
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

from quantum_canary import CanaryConfig, LogicalProgram, pair
from bench.circuits import ising_trotter
from bench.device import ExactDevice, sample_device
from bench.runner import Evaluator, oracle_assignment, random_assignments, run_canary

EVENT_ERROR = 0.06


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=16)
    ap.add_argument("--ions", type=int, default=12)
    ap.add_argument("--qubits", type=int, default=6)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--event-batch", type=int, default=6)
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--anneal-steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--target", choices=("static", "canary"), default="static")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or ROOT / "research" / "data" / f"exp3_{args.target}")
    out.mkdir(parents=True, exist_ok=True)
    program = LogicalProgram.from_circuit(ising_trotter(args.qubits, args.steps))
    master = np.random.default_rng(args.seed)
    series, summary = [], []
    t0 = time.time()
    for inst in range(args.instances):
        rng = np.random.default_rng(master.integers(2**31))
        truth = sample_device(args.ions, rng, outlier_fraction=0.05)
        device = ExactDevice(truth, seed=int(rng.integers(2**31)))
        ev = Evaluator(program, device)
        static, _ = oracle_assignment(program, truth, ev, rng, restarts=2)
        rand = random_assignments(args.qubits, range(args.ions), 25, rng)
        event: dict = {}
        static_f, debias_f = [], []

        def before(b: int, canary) -> None:
            if b == args.event_batch:
                rec = static if args.target == "static" else canary.recommend(program)
                (u, v), _ = max(program.interaction_weights.items(), key=lambda kv: (kv[1], -kv[0][0]))
                i, j = sorted((rec[u], rec[v]))
                device.truth.set_pair(i, j, EVENT_ERROR)
                event.update(
                    {
                        "pair": (i, j),
                        "in_static": pair(i, j) in program.used_resources(static),
                        "in_canary": pair(i, j) in program.used_resources(canary.recommend(program)),
                    }
                )
            static_f.append(ev.fidelity(static))
            debias_f.append(ev.mixture_fidelity([(a, 1.0) for a in rand]))

        cfg = CanaryConfig(seed=int(rng.integers(2**31)), anneal_steps=args.anneal_steps)
        tr = run_canary(program, device, ev, args.batches, args.shots, cfg, before_batch=before)
        key = ":".join(map(str, ("pair", *event["pair"])))
        detect = next((b for b in range(args.event_batch, args.batches) if key in tr.degraded[b]), None)
        avoid = next(
            (b for b in range(args.event_batch, args.batches) if pair(*event["pair"]) not in program.used_resources(tr.recommended[b])),
            None,
        )
        for b in range(args.batches):
            series.append(
                {
                    "instance": inst,
                    "batch": b,
                    "canary_trusted": tr.batch_fidelity[b],
                    "canary_exploit": tr.batch_exploit_fidelity[b],
                    "canary_recommended": tr.recommended_fidelity[b],
                    "static_oracle": static_f[b],
                    "debias": debias_f[b],
                }
            )
        summary.append(
            {
                "instance": inst,
                "event_pair": list(event["pair"]),
                "event_pair_in_static": bool(event["in_static"]),
                "event_pair_in_canary": bool(event["in_canary"]),
                "detect_latency_batches": None if detect is None else detect - args.event_batch + 1,
                "reroute_latency_batches": None if avoid is None else avoid - args.event_batch + 1,
            }
        )
        print(f"[{inst + 1}/{args.instances}] detect={summary[-1]['detect_latency_batches']} reroute={summary[-1]['reroute_latency_batches']} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(series).to_csv(out / "series.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
