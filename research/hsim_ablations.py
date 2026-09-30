"""Ablations of the information-acquisition changes on the Hamiltonian-simulation experiment (first 10 sessions).

Each variant runs research/hsim_before_after.py with --no-extra and CanaryConfig overrides that switch changes off;
the original rows (identity, delivered, final map, ...) are computed exactly as in the acceptance run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OFF_D = {"exploit_fraction": 0.25, "explore_typical": False, "trust_requires_vetting": False}
VARIANTS = {
    "before (all changes off)": {
        "probe_pack": 1, "coverage_weight": 0.0, "screening": False, "explore_mode": "thompson", **OFF_D,
    },
    "A": {"probe_pack": 4, "coverage_weight": 0.0, "screening": False, "explore_mode": "thompson", **OFF_D},
    "A+B": {"coverage_weight": 1.0, "screening": False, "explore_mode": "thompson", **OFF_D},
    "A+B+C": {**OFF_D},
    "A+B+C+D (default)": {},
    "default, explore_mode=thompson": {"explore_mode": "thompson"},
    "default, screen_calibration=False": {"screen_calibration": False},
    "default, screening=False": {"screening": False},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument("--forte-dir", default="/tmp/forte")
    ap.add_argument("--out-dir", default=str(ROOT / "research" / "ablations"))
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, (name, overrides) in enumerate(VARIANTS.items()):
        if args.only and name not in args.only:
            continue
        out = out_dir / f"variant_{k}.json"
        cmd = [
            sys.executable, str(ROOT / "research" / "hsim_before_after.py"), "--sessions", str(args.sessions),
            "--no-extra", "--forte-dir", args.forte_dir, "--config", json.dumps(overrides), "--out", str(out),
        ]
        print(f"== {name}: {json.dumps(overrides)}", flush=True)
        subprocess.run(cmd, check=True)
        data = json.loads(out.read_text())
        data["variant"] = name
        data["overrides"] = overrides
        out.write_text(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
