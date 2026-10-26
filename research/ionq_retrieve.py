"""Retrieve a finished IonQ Canary job (e.g. one whose run was interrupted before it was saved) and fit it exactly as
run_ionq does. Sequence lengths and ions come from the circuit names IonQ stored with the job
(canary_rb_<s>_<j>_m<length>_ions<ions>); the priors, recorded with the fit, from IonQ's characterization now.
Usage: python research/ionq_retrieve.py JOB_ID TAG [BACKEND]     e.g.  5c3895b6-....  ion0_isolated"""
import json, re, sys, pathlib, numpy as np, quantum_canary as canary
from datetime import datetime, timezone
from scipy.stats import chi2
from quantum_canary import inversion as inv, ionq

job_id, tag, name = sys.argv[1], sys.argv[2], (sys.argv[3:4] or ["qpu.forte-1"])[0]
backend = canary.credentials.ionq_provider(pathlib.Path(__file__).resolve().parents[1] / "ionq_api_token.txt").get_backend(name, gateset="native")
result = backend.retrieve_job(job_id).result()
names = [r["header"].get("name", "") for r in result.to_dict()["results"]]
parsed = [re.fullmatch(r"canary_rb_(\d+)_(\d+)_m(\d+)_ions([\d-]+)", n) for n in names]
if not all(parsed):
    sys.exit(f"not a Canary job: circuit names {names[:3]} ...")
length_of = {int(m.group(1)): int(m.group(3)) for m in parsed}
rb_lengths = [length_of[s] for s in sorted(length_of)]
ions = [int(x) for x in parsed[0].group(4).split("-")]
position = {(int(m.group(1)), int(m.group(2))): k for k, m in enumerate(parsed)}
order = [(s, j) for s, k in enumerate(inv._RB_SEQUENCES) for j in range(k)]
if sorted(position) != sorted(order):
    sys.exit(f"circuits {sorted(position)[:3]} ... are not the engine's {len(order)} sequences")

prior = ionq.ionq_priors(backend) or {}
arch = ionq._arch(prior.get("eps_prior"), prior.get("spam"))
counts = [result.get_counts(position[sj]) for sj in order]
shots = int(sum(counts[0].values()))
dof = len(rb_lengths) - 2
results = []
for bit, q in enumerate(ions):
    p0 = np.array([z / t for z, t in (ionq._p0(c, bit) for c in counts)])
    eps, sigma, chi2_dof, _ = inv._invert_rb(p0, rb_lengths, inv._RB_SEQUENCES, arch)
    results.append(ionq.IonResult(ion=q, epsilon_sx=float(eps), epsilon_sx_sigma=float(sigma), chi2_dof=chi2_dof,
                                  chi2_p=float(chi2.sf(chi2_dof * dof, dof)), rb_lengths=rb_lengths,
                                  p0=[float(x) for x in p0], shots=[shots] * len(order)))
raw = [{"job_id": job_id, "batch": ions, "circuit": i, "name": names[position[sj]], "counts": dict(c), "probabilities": None}
       for i, (sj, c) in enumerate(zip(order, counts))]
report_raw = ionq.IonQReport(backend=name, ions=ions, shots=shots, rb_lengths=rb_lengths,
                             eps_prior=float(arch["eps_typical"]), spam=(float(arch["p0_given_1"]), float(arch["p1_given_0"])),
                             batches=[ions], job_ids=[job_id], ionq_estimates=[None], results=results, raw=raw)
report_raw.print()
report = canary.Report(ionq.to_canary_report(report_raw))
stem = f"ionq_{name}_{tag}_{report.timestamp[:19].replace(':', '')}"
out = pathlib.Path(__file__).resolve().parent / "data" / "hardware" / f"{stem}.json"
report.save(out)
(out.parent / f"{stem}_ionq.json").write_text(json.dumps(dict(report_raw.to_dict(), retrieved_utc=datetime.now(timezone.utc).isoformat(),
                                                               prior_date=prior.get("date")), indent=1, default=str))
print("saved:", out)
