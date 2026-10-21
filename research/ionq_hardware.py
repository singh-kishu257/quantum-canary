"""Quantum-Canary gate error on ten fixed IonQ ions, priors from IonQ characterization.
Usage: python research/ionq_hardware.py TAG [BACKEND] [MAX_COST]   e.g.  A   |   B qpu.forte-1 3000"""
import sys, pathlib, quantum_canary as canary
from provenance import write_meta

tag, name, cap = sys.argv[1], (sys.argv[2:3] or ["simulator"])[0], (sys.argv[3:4] or [None])[0]

if name.startswith("qpu.") and cap is None:
    sys.exit("a QPU run needs MAX_COST (checked against IonQ's estimate)")

backend = canary.credentials.ionq_provider(pathlib.Path(__file__).parents[1] / "ionq_api_token.txt").get_backend(name, gateset="native")
report = canary.Report(canary.ionq.to_canary_report(canary.run_ionq(backend, [0, 3, 6, 9, 12, 14, 18, 21, 27, 33], max_cost=float(cap) if cap else None)))
report.print()
out = pathlib.Path(__file__).parent / "data" / "hardware" / f"ionq_{name}_{tag}_{report.timestamp[:19].replace(':', '')}.json"

report.save(out)
write_meta(out)
