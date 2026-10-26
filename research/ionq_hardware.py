"""Quantum-Canary gate error on ten fixed IonQ ions, priors from IonQ's published characterization.
Usage: python research/ionq_hardware.py TAG [BACKEND] [MAX_COST]   e.g.  A   |   A qpu.forte-1 3000
       python research/ionq_hardware.py estimate [BACKEND]          IonQ's prior, sequence lengths and cost estimate;
                                                                    nothing is submitted
BACKEND sim:forte-1 is IonQ's simulator with the forte-1 noise model, priors from qpu.forte-1's characterization,
on qubits 0-9 (a simulator simulates every qubit up to the highest one used, and cannot run 34).
A run saves the Canary report and IonQ's full report (IonQ's cost estimates, the prior, sequence lengths, job ids,
raw counts, start time and wall time)."""
import json, sys, time, pathlib, quantum_canary as canary
from datetime import datetime, timezone

IONS = [0, 3, 6, 9, 12, 14, 18, 21, 27, 33]
tag, name, cap = sys.argv[1], (sys.argv[2:3] or ["simulator"])[0], (sys.argv[3:4] or [None])[0]

if name.startswith("qpu.") and cap is None and tag != "estimate":
    sys.exit("a QPU run needs MAX_COST (checked against IonQ's estimate)")

provider = canary.credentials.ionq_provider(pathlib.Path(__file__).resolve().parents[1] / "ionq_api_token.txt")
noise = name.split(":", 1)[1] if name.startswith("sim:") else None
backend = provider.get_backend("simulator" if noise else name, gateset="native")
priors = {}
if noise:
    published = canary.ionq.ionq_priors(provider.get_backend(f"qpu.{noise}", gateset="native")) or {}
    priors = {"eps_prior": published.get("eps_prior"), "spam": published.get("spam")}

if tag == "estimate":
    prior = canary.ionq.ionq_priors(backend) or {}
    arch = canary.ionq._arch(prior.get("eps_prior"), prior.get("spam"))
    lengths = canary.ionq._profile(arch).rb_lengths
    batches, circuits = canary.ionq.ionq_circuits(IONS, lengths, seed=0)
    n = len(circuits) // len(batches)
    est = [canary.ionq.ionq_estimate(backend, circuits[i * n:(i + 1) * n], canary.ionq.DEFAULT_SHOTS) for i in range(len(batches))]
    gates = sum(v for qc in circuits for k, v in qc.count_ops().items() if k.startswith("gpi"))
    print(f"IonQ characterization {prior.get('date')}: eps_prior {arch['eps_typical']:.3g} (1 - median 1q fidelity), spam {prior.get('spam')}")
    print(f"RB lengths {lengths}; {n} circuits x {canary.ionq.DEFAULT_SHOTS} shots per job; {gates * canary.ionq.DEFAULT_SHOTS:,.0f} gate-shots per run")
    for b, e in zip(batches, est):
        print(f"  job, ions {b}: " + (f"{e['cost']:,.2f} {e['cost_unit']} (per circuit {min(e['per_circuit_cost']):.2f}-"
                                      f"{max(e['per_circuit_cost']):.2f}), execution {e['execution_time']:,.0f} (IonQ's units, ms)"
                                      if e and e.get("cost") is not None else f"no cost from IonQ; IonQ returned {e and e.get('raw')}"))
    if all(e and e.get("cost") is not None for e in est):
        print(f"one run: {sum(e['cost'] for e in est):,.2f} {est[0]['cost_unit']}  (A + B: {2 * sum(e['cost'] for e in est):,.2f})")
    sys.exit()

start, t0 = datetime.now(timezone.utc).isoformat(), time.perf_counter()
ions = list(range(len(IONS))) if noise else IONS  # simulators run the circuit's full width: keep it 10 qubits
raw = canary.run_ionq(backend, ions, max_cost=float(cap) if cap else None,
                      run_options={"noise_model": noise} if noise else None, **priors)
wall = time.perf_counter() - t0
report = canary.Report(canary.ionq.to_canary_report(raw))
report.print()
stem = f"ionq_{name.replace(':', '-')}_{tag}_{report.timestamp[:19].replace(':', '')}"
out = pathlib.Path(__file__).resolve().parent / "data" / "hardware" / f"{stem}.json"
report.save(out)
full = out.parent / f"{stem}_ionq.json"
full.write_text(json.dumps(dict(raw.to_dict(), start_utc=start, wall_s=wall), indent=1, default=str))
