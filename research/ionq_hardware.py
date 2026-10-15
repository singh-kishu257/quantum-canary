"""IonQ hardware runs for the paper: a gate-error map of ten ions on Forte-1.

On trapped ions only the gate-repetition probe is run (T1 and T2 are seconds
long). Each ion is probed on its own (several ions per circuit gave
crosstalk in August 2026), and every finished ion is saved, so a run that
stops can be resumed. Run A and run B measure the same ions; the analysis
checks that each ion's epsilon_sx repeats within its error bars and sets
the median against IonQ's published device-average single-qubit error.

  estimate   repetition counts, gates and shots per ion, and IonQ's own
             estimate for them; nothing is sent
  run        one run (--tag A or B); resumes; on hardware refused if IonQ's
             own estimate of the remaining ions is above --max-cost
  analyze    A vs B per ion, and the map vs IonQ's published numbers

--simulator forte-1 runs on IonQ's free cloud simulator with that noise
model. Ions are fixed in advance: 3, 6 and 14 were the highest in the
August 2026 untwirled run, the rest are spread along the chain.

Fixed before any data: A and B pass if, pooled over the ions, a one-sided
binomial test does not reject |z| <= 2 at rate 0.954 (p < 0.05). An ion
counts as worse than the device if it sits more than 2 sigma above the
median in both runs.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import binom

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_SRC = SCRIPT_DIR.parent / "src"
if REPO_SRC.is_dir():
    sys.path.insert(0, str(REPO_SRC))

from quantum_canary import ionq

DATA_DIR = SCRIPT_DIR / "data" / "hardware"
IONS = [0, 3, 6, 9, 12, 14, 18, 21, 27, 33]
TAGS = ("A", "B")
SPLIT_TARGET, ALPHA = 0.9545, 0.05


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def token() -> str:
    key = os.environ.get("IONQ_API_KEY")
    if not key:
        raise SystemExit("set IONQ_API_KEY")
    return key


def ssl_setup() -> None:
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass


def characterization(name: str) -> Dict[str, Any]:
    """IonQ's published characterization of qpu.<name> (ionq.ionq_priors): the
    device single-qubit error and the readout errors split as the engine does."""
    ssl_setup()
    from qiskit_ionq import IonQProvider
    pri = ionq.ionq_priors(IonQProvider(token()).get_backend(f"qpu.{name}", gateset="native"))
    if not pri:
        return {}
    out = {"fetched": now(), "date": pri["date"], "raw": pri["raw"], "stat": pri.get("eps_stat")}
    if pri.get("eps_prior") is not None:
        out["eps_1q"] = pri["eps_prior"]
    if pri.get("eps_stderr") is not None:
        out["eps_1q_stderr"] = pri["eps_stderr"]
    if pri.get("spam") is not None:
        out["spam"] = pri["spam"]
    return out


def backend_for(args):
    ssl_setup()
    from qiskit_ionq import IonQProvider
    provider = IonQProvider(token())
    if args.simulator:
        b = provider.get_backend("simulator", gateset="native")
        b.set_options(noise_model=args.simulator)
        return b
    return provider.get_backend(f"qpu.{args.backend}", gateset="native")


def priors(args) -> Dict[str, Any]:
    char = characterization(args.backend) if os.environ.get("IONQ_API_KEY") else {}
    eps = args.eps_prior if args.eps_prior is not None else char.get("eps_1q")
    spam = tuple(args.spam) if args.spam else char.get("spam")
    return {"characterization": char, "eps_prior": eps, "spam": spam}


def out_dir(args) -> pathlib.Path:
    name = f"ionq_sim_{args.simulator}" if args.simulator else f"ionq_{args.backend}"
    d = pathlib.Path(args.out) if args.out else DATA_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def ion_estimates(backend, ions, n, shots):
    """IonQ's own estimate for each ion's job, or None per ion if IonQ gives none."""
    return {q: ionq.ionq_estimate(backend, ionq.ionq_circuits([q], n)[1], shots) for q in ions}


def show_estimates(est) -> Optional[float]:
    if any(e is None or e.get("cost") is None for e in est.values()):
        print("IonQ returned no estimate; check the cost on your IonQ account before running")
        return None
    unit = next(iter(est.values()))["cost_unit"]
    total = sum(float(e["cost"]) for e in est.values())
    times = [e.get("execution_time_s") for e in est.values()]
    print(f"IonQ's estimate: {total:,.2f} {unit} for {len(est)} ions"
          + (f", execution {sum(times) / 3600:.1f} h" if all(t is not None for t in times) else "")
          + " (IonQ's numbers; actual charges are on your IonQ account)")
    return total


def cmd_estimate(args, d):
    p = priors(args)
    arch = ionq._arch(p["eps_prior"], p["spam"])
    n = ionq._profile(arch).gate_rep_n
    _, circuits = ionq.ionq_circuits(args.ions[:1], n)
    gates = sum(c.count_ops().get("gpi2", 0) for c in circuits)
    print(f"ions {args.ions}, one at a time")
    print(f"eps prior {arch['eps_typical']:.2e} ({'published' if p['eps_prior'] is not None else 'engine default'}), "
          f"readout P(0|1) {arch['p0_given_1']:.4f} P(1|0) {arch['p1_given_0']:.4f}")
    print(f"per ion: N = {n} pairs, {len(circuits)} circuits, {gates:,} native gates, {args.shots} shots each")
    if os.environ.get("IONQ_API_KEY") and not args.simulator:
        show_estimates(ion_estimates(backend_for(args), args.ions, n, args.shots))


def cmd_run(args, d):
    path = d / f"run_{args.tag}.json"
    partial = d / f"run_{args.tag}.partial.json"
    if path.exists() and not args.force:
        raise SystemExit(f"{path.name} exists; pass --force to overwrite")
    state = json.loads(partial.read_text()) if partial.exists() else None
    if state and state["ions"] != args.ions:
        raise SystemExit(f"{partial.name} is for ions {state['ions']}; finish it or delete it")
    if state is None:
        p = priors(args)
        if not args.simulator and (p["eps_prior"] is None or p["spam"] is None):
            raise SystemExit("no published characterization: pass --eps-prior and --spam")
        state = {"tag": args.tag, "backend": args.backend, "simulator": args.simulator,
                 "ions": args.ions, "shots": args.shots, "t_submit": now(),
                 "characterization": p["characterization"], "eps_prior": p["eps_prior"],
                 "spam": p["spam"], "per_ion": {}}
    if not args.simulator and args.max_cost is None:
        raise SystemExit("hardware runs need --max-cost (in IonQ's cost unit)")
    todo = [q for q in args.ions if str(q) not in state["per_ion"]]
    arch = ionq._arch(state["eps_prior"], state["spam"])
    n = ionq._profile(arch).gate_rep_n
    backend = backend_for(args)
    print(f"{len(todo)} ions to go ({len(state['per_ion'])} done)")
    budget = None
    if not args.simulator:
        estimated = show_estimates(ion_estimates(backend, todo, n, state["shots"]))
        done = sum(float(e["cost"]) for r in state["per_ion"].values()
                   for e in r.get("ionq_estimates") or [] if e and e.get("cost") is not None)
        if estimated is None:
            raise SystemExit("IonQ returned no estimate, so --max-cost cannot be checked")
        if done + estimated > args.max_cost:
            raise SystemExit(f"IonQ estimates {done + estimated:,.2f} for the whole run, above --max-cost {args.max_cost:,.2f}")
        budget = args.max_cost - done
    for q in todo:
        report = ionq.run_ionq(backend, [q], shots=state["shots"], eps_prior=state["eps_prior"],
                               spam=state["spam"], max_cost=budget)
        if budget is not None:
            budget -= sum(float(e["cost"]) for e in report.ionq_estimates)
        state["per_ion"][str(q)] = report.to_dict()
        partial.write_text(json.dumps(state, indent=1, default=str))
        r = report.results[0]
        print(f"ion {q}: eps {r.epsilon_sx:.3e} +- {r.epsilon_sx_sigma:.1e}, chi2/dof {r.chi2_dof:.2f}, "
              f"job {report.job_ids[0]}, {now()[:19]}")
    reports = [state["per_ion"][str(q)] for q in args.ions]
    payload = {k: state[k] for k in ("tag", "backend", "simulator", "t_submit", "characterization")}
    payload["t_done"] = now()
    payload["report"] = {"ions": args.ions, "shots": state["shots"], "gate_rep_n": reports[0]["gate_rep_n"],
                         "eps_prior": reports[0]["eps_prior"],
                         "p0_given_1": reports[0]["p0_given_1"], "p1_given_0": reports[0]["p1_given_0"],
                         "ionq_estimates": [e for r in reports for e in r.get("ionq_estimates") or []],
                         "job_ids": [j for r in reports for j in r["job_ids"]],
                         "results": [r["results"][0] for r in reports],
                         "per_ion": reports}
    path.write_text(json.dumps(payload, indent=1, default=str))
    partial.unlink()
    print(f"saved {path}")


def cmd_analyze(args, d):
    runs = {t: json.loads((d / f"run_{t}.json").read_text()) for t in TAGS
            if (d / f"run_{t}.json").exists()}
    if not runs:
        raise SystemExit("no runs yet")
    res = {t: {r["ion"]: r for r in run["report"]["results"]} for t, run in runs.items()}
    out: Dict[str, Any] = {"created": now(), "runs": list(runs)}
    for t, rs in res.items():
        eps = np.array([r["epsilon_sx"] for r in rs.values()])
        print(f"run {t}: median eps {np.median(eps):.3e}, range {eps.min():.3e} - {eps.max():.3e}, "
              f"poor fits {sum(r['poor_fit'] for r in rs.values())}/{len(rs)}")
    if len(runs) == 2:
        ions = sorted(set(res["A"]) & set(res["B"]))
        zs = [(res["A"][q]["epsilon_sx"] - res["B"][q]["epsilon_sx"])
              / np.hypot(res["A"][q]["epsilon_sx_sigma"], res["B"][q]["epsilon_sx_sigma"]) for q in ions]
        k, n = sum(abs(z) <= 2 for z in zs), len(zs)
        p = float(binom.cdf(k, n, SPLIT_TARGET))
        print(f"\nA vs B: {k}/{n} ions within 2 sigma, binomial p = {p:.3g} "
              f"-> {'pass' if p >= ALPHA else 'fail'}")
        for q, z in zip(ions, zs):
            a, b = res["A"][q], res["B"][q]
            print(f"   ion {q:>2}: A {a['epsilon_sx']:.3e} +- {a['epsilon_sx_sigma']:.1e}   "
                  f"B {b['epsilon_sx']:.3e} +- {b['epsilon_sx_sigma']:.1e}   z {z:+.2f}")
        worse = []
        for q in ions:
            above = []
            for t in TAGS:
                med = np.median([r["epsilon_sx"] for r in res[t].values()])
                above.append((res[t][q]["epsilon_sx"] - med) / res[t][q]["epsilon_sx_sigma"] > 2)
            if all(above):
                worse.append(q)
        print(f"ions more than 2 sigma above the median in both runs: {worse or 'none'}")
        out.update({"a_vs_b_z": dict(zip(ions, zs)), "within": k, "n": n, "binomial_p": p,
                    "worse_than_device": worse})
    for t, run in runs.items():
        char = run.get("characterization") or {}
        if char.get("eps_1q") is not None:
            med = np.median([r["epsilon_sx"] for r in res[t].values()])
            se = char.get("eps_1q_stderr")
            print(f"run {t}: Canary median {med:.3e} vs IonQ published {char['stat']} "
                  f"{char['eps_1q']:.3e}" + (f" +- {se:.1e}" if se else "")
                  + f" (fetched {char['fetched'][:16]})")
    (d / "analysis.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"saved {d / 'analysis.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("estimate", "run", "analyze"))
    ap.add_argument("--backend", default="forte-1")
    ap.add_argument("--simulator", default=None, help="IonQ cloud simulator noise model, e.g. forte-1")
    ap.add_argument("--ions", type=lambda s: [int(x) for x in s.split(",")], default=IONS)
    ap.add_argument("--tag", choices=TAGS)
    ap.add_argument("--shots", type=int, default=ionq.DEFAULT_SHOTS)
    ap.add_argument("--max-cost", type=float, default=None, help="limit in IonQ's cost unit, checked against IonQ's own estimates")
    ap.add_argument("--eps-prior", type=float, default=None)
    ap.add_argument("--spam", type=float, nargs=2, default=None, metavar=("P0_GIVEN_1", "P1_GIVEN_0"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.command == "run" and not args.tag:
        ap.error("run needs --tag")
    d = out_dir(args)
    {"estimate": cmd_estimate, "run": cmd_run, "analyze": cmd_analyze}[args.command](args, d)


if __name__ == "__main__":
    main()
