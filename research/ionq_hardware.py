"""Quantum-Canary gate error on ten fixed IonQ ions, priors from IonQ's published characterization.
Usage: python research/ionq_hardware.py submit TAGS [BACKEND] [MAX_COST]   e.g.  submit A,B,C,D qpu.forte-1 1950
           every run's jobs submitted at once, without waiting; each run's plan (job ids, which circuits each job
           holds, seed, lengths, prior, IonQ's estimate) saved after every job: ionq_<backend>_<tag>_<time>_plan.json
       python research/ionq_hardware.py collect [BACKEND]       fit and save every submitted run whose jobs are all
           done (the Canary report and IonQ's full report); report the status of the others. Run it again later.
       python research/ionq_hardware.py TAG [BACKEND] [MAX_COST]   one run, waiting for its jobs
       python research/ionq_hardware.py estimate [BACKEND]          IonQ's prior, sequence lengths and cost estimate;
                                                                    nothing is submitted
MAX_COST (per run) is checked against IonQ's estimate before a run is submitted; a QPU needs one.
BACKEND sim:forte-1 is IonQ's simulator with the forte-1 noise model, priors from qpu.forte-1's characterization,
on qubits 0-9 (a simulator simulates every qubit up to the highest one used, and cannot run 34)."""
import json, sys, time, pathlib, quantum_canary as canary
from datetime import datetime, timezone

IONS = [0, 3, 6, 9, 12, 14, 18, 21, 27, 33]
OUT = pathlib.Path(__file__).resolve().parent / "data" / "hardware"
cmd = sys.argv[1]
if cmd == "submit":
    tags, name, cap = sys.argv[2].split(","), (sys.argv[3:4] or ["simulator"])[0], (sys.argv[4:5] or [None])[0]
else:
    tags, name, cap = [cmd], (sys.argv[2:3] or ["simulator"])[0], (sys.argv[3:4] or [None])[0]
if name.startswith("qpu.") and cap is None and cmd not in ("estimate", "collect"):
    sys.exit("a QPU run needs MAX_COST (checked against IonQ's estimate)")

provider = canary.credentials.ionq_provider(pathlib.Path(__file__).resolve().parents[1] / "ionq_api_token.txt")
noise = name.split(":", 1)[1] if name.startswith("sim:") else None
backend = provider.get_backend("simulator" if noise else name, gateset="native")
safe = name.replace(":", "-")
priors, options = {}, {}
if noise:
    published = canary.ionq.ionq_priors(provider.get_backend(f"qpu.{noise}", gateset="native")) or {}
    priors = {"eps_prior": published.get("eps_prior"), "spam": published.get("spam")}
    options = {"run_options": {"noise_model": noise}}
ions = list(range(len(IONS))) if noise else IONS  # simulators run the circuit's full width: keep it 10 qubits


def save(raw, stem, extra):
    report = canary.Report(canary.ionq.to_canary_report(raw))
    report.print()
    report.save(OUT / f"{stem}.json")
    (OUT / f"{stem}_ionq.json").write_text(json.dumps(dict(raw.to_dict(), **extra), indent=1, default=str))
    print("saved:", OUT / f"{stem}.json")


if cmd == "estimate":
    prior = canary.ionq.ionq_priors(backend) or {}
    arch = canary.ionq._arch(prior.get("eps_prior"), prior.get("spam"))
    lengths = canary.ionq._profile(arch).rb_lengths
    batches, circuits = canary.ionq.ionq_circuits(IONS, lengths, seed=0)
    n = len(circuits) // len(batches)
    est = [canary.ionq.ionq_estimate(backend, circuits[i * n:(i + 1) * n], canary.ionq.DEFAULT_SHOTS) for i in range(len(batches))]
    gates = sum(v for qc in circuits for k, v in qc.count_ops().items() if k.startswith("gpi"))
    print(f"IonQ characterization {prior.get('date')}: eps_prior {arch['eps_typical']:.3g} (1 - median 1q fidelity), spam {prior.get('spam')}")
    print(f"RB lengths {lengths}; {n} circuits x {canary.ionq.DEFAULT_SHOTS} shots per run; {gates * canary.ionq.DEFAULT_SHOTS:,.0f} gate-shots per run")
    for b, e in zip(batches, est):
        print(f"  ions {b}: " + (f"{e['cost']:,.2f} {e['cost_unit']} (per circuit {min(e['per_circuit_cost']):.2f}-"
                                 f"{max(e['per_circuit_cost']):.2f}), execution {e['execution_time']:,.0f} (IonQ's units, ms)"
                                 if e and e.get("cost") is not None else f"no cost from IonQ; IonQ returned {e and e.get('raw')}"))
    if all(e and e.get("cost") is not None for e in est):
        print(f"one run: {sum(e['cost'] for e in est):,.2f} {est[0]['cost_unit']}")

elif cmd == "submit":
    for tag in tags:
        stem = f"ionq_{safe}_{tag}_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%S')}"
        path = OUT / f"{stem}_plan.json"

        def write(plan, path=path, tag=tag):
            jobs = [{k: v for k, v in r.items() if k != "_job"} for r in plan["jobs"]]
            path.write_text(json.dumps(dict(plan, jobs=jobs, tag=tag, request=name), indent=1, default=str))

        plan = canary.ionq.submit_ionq(backend, ions, max_cost=float(cap) if cap else None, on_job=write,
                                       **priors, **options)
        print(f"run {tag}: {len(plan['jobs'])} jobs submitted -> {path.name}")

elif cmd == "collect":
    for path in sorted(OUT.glob(f"ionq_{safe}_*_plan.json")):
        stem = path.name[:-len("_plan.json")]
        if (OUT / f"{stem}_ionq.json").exists():
            print(f"{stem}: already collected")
            continue
        plan = json.loads(path.read_text())
        status = [str(backend.retrieve_job(r["job_id"]).status()) for r in plan["jobs"]]
        if not all(s.endswith("DONE") for s in status):
            print(f"{stem}: {sum(s.endswith('DONE') for s in status)}/{len(status)} jobs done ({', '.join(status)})")
            continue
        ionq_jobs = [backend.client.retrieve_job(r["job_id"]) for r in plan["jobs"]]
        save(canary.ionq.collect_ionq(backend, plan), stem,
             {"submitted_utc": [r["submitted_utc"] for r in plan["jobs"]], "collected_utc": datetime.now(timezone.utc).isoformat(),
              "ionq_job_times": [{k: j.get(k) for k in ("submitted_at", "started_at", "completed_at", "execution_duration_ms")}
                                 for j in ionq_jobs]})

else:
    start, t0 = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    raw = canary.run_ionq(backend, ions, max_cost=float(cap) if cap else None, **priors, **options)
    save(raw, f"ionq_{safe}_{cmd}_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%S')}",
         {"start_utc": start, "wall_s": time.perf_counter() - t0})
