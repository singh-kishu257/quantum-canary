import argparse
import importlib.util
import json
import pathlib
import sys
import warnings
from datetime import datetime, timezone

import truststore
truststore.inject_into_ssl()

import numpy as np
from qiskit import transpile

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "inversion", SCRIPT_DIR / "1_inversion.py")
inv = importlib.util.module_from_spec(_spec)
sys.modules["inversion"] = inv
_spec.loader.exec_module(inv)

DATA_DIR = SCRIPT_DIR / "data" / "hardware"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ARCHITECTURE  = "superconducting"
DEFAULT_SHOTS = dict(t1=300, ramsey=1000, gate=500, echo=500)
FIT_FRACTION  = 0.70


def get_backend(backend_name: str, instance: str = "open-instance"):
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(channel="ibm_quantum_platform", instance=instance)
    return service.backend(backend_name)


def get_dry_run_backend():
    try:
        from qiskit_aer import AerSimulator
        return AerSimulator(method="statevector")
    except ImportError:
        return None


def select_qubits(backend, n: int) -> list:
    if backend is None:
        return list(range(n))
    try:
        props  = backend.properties()
        scores = []
        for q in range(backend.num_qubits):
            try:
                t1 = float(props.t1(q)) or 0.0
                re = float(props.readout_error(q)) or 1.0
                scores.append((q, t1 * (1.0 - re)))
            except Exception:
                scores.append((q, 0.0))
        scores.sort(key=lambda x: -x[1])
        chosen = [q for q, _ in scores[:n]]
        print(f"  Auto-selected qubits: {chosen}")
        return chosen
    except Exception as e:
        print(f"  WARNING: auto-selection failed ({e}); using 0..{n-1}")
        return list(range(n))


def load_profiles(backend, qubits: list, props, dry_run: bool) -> list:
    profiles = []
    for q in qubits:
        if dry_run:
            p = inv.BackendProfile.from_architecture(ARCHITECTURE)
        else:
            p = inv.get_live_profile(backend, qubit_id=q)
            if props is not None:
                try:
                    t1     = float(props.t1(q))
                    t2_cal = float(props.t2(q))
                    t2     = min(t2_cal, t1 / 4.0)
                    p = inv.BackendProfile(
                        architecture=p.architecture,
                        T1_prior_s=t1, T2_prior_s=t2,
                        dt_ns=p.dt_ns, backend_name=p.backend_name,
                        custom_arch=p.custom_arch,
                        calibration_source=p.calibration_source,
                        prior_confidence="live")
                    print(f"[Canary] Q{q:>3d}: T1_prior={t1*1e6:.1f}µs  "
                          f"T2_prior={t2*1e6:.1f}µs (from {t2_cal*1e6:.1f}µs cal)")
                except Exception as exc:
                    print(f"[Canary] Q{q} profile patch failed: {exc}")
        profiles.append(p)
    return profiles


def _shots_for_meta(meta: dict) -> list:
    DS = DEFAULT_SHOTS
    return (
        [DS["t1"]]     * meta["n_t1"]    +
        [DS["ramsey"]] * meta["n_ramsey"] +
        [DS["gate"]]   * meta["n_gate"]   +
        [DS["echo"]]   * meta["n_echo"]
    )


def _strip_delays(qc):
    c = qc.copy()
    c.data = [inst for inst in c.data if inst.operation.name != "delay"]
    return c


def build_and_transpile(profiles: list, qubits: list,
                        backend, dry_run: bool) -> tuple:
    all_circuits, all_shots, all_qids = [], [], []
    per_qubit_metas = []

    for q_id, profile in zip(qubits, profiles):
        circuits, meta = inv.build_probe_circuits(profile)
        per_qubit_metas.append(meta)
        shots_seq = _shots_for_meta(meta)
        assert len(circuits) == len(shots_seq)
        all_circuits.extend(circuits)
        all_shots.extend(shots_seq)
        all_qids.extend([q_id] * len(circuits))

    circuits_per_qubit = len(all_circuits) // len(qubits)

    if dry_run:
        stripped = [_strip_delays(qc) for qc in all_circuits]
        pubs = [(qc, None, sh) for qc, sh in zip(stripped, all_shots)]
    else:
        print(f"  Transpiling {len(all_circuits)} circuits "
              f"({circuits_per_qubit} per qubit, per-qubit layout, opt=0)... ",
              end="", flush=True)
        t_circuits = [
            transpile(qc, backend=backend,
                      initial_layout=[q_id], optimization_level=0)
            for qc, q_id in zip(all_circuits, all_qids)
        ]
        print("done")
        pubs = [(tc, None, sh) for tc, sh in zip(t_circuits, all_shots)]

    return pubs, all_shots, per_qubit_metas, circuits_per_qubit


def submit(backend, pubs: list, dry_run: bool):
    if dry_run:
        from qiskit.primitives import StatevectorSampler
        sampler = StatevectorSampler()
        print(f"  [dry-run] {len(pubs)} PUBs → StatevectorSampler... ",
              end="", flush=True)
        result = sampler.run(pubs).result()
        print("done")
        return result

    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(backend)
    print(f"  Submitting {len(pubs)} PUBs as one job... ", end="", flush=True)
    job = sampler.run(pubs)
    print(f"job_id={job.job_id()}")
    print("  Waiting for results... ", end="", flush=True)
    result = job.result()
    print("done")
    return result


def _single_qubit_counts(pub_result, shots: int) -> dict:
    for reg in ["c", "meas", "measure"]:
        try:
            raw = getattr(pub_result.data, reg).get_counts()
            c   = {"0": int(raw.get("0", 0)), "1": int(raw.get("1", 0))}
            total = c["0"] + c["1"]
            if total < shots:
                c["0"] += shots - total
            return c
        except AttributeError:
            continue
    return {"0": shots // 2, "1": shots - shots // 2}


def reassemble(job_result, all_shots: list,
               per_qubit_metas: list, qubits: list,
               circuits_per_qubit: int) -> list:
    per_qubit_counts = []
    for q_idx in range(len(qubits)):
        start  = q_idx * circuits_per_qubit
        meta   = per_qubit_metas[q_idx]
        n_circ = meta["n_t1"] + meta["n_ramsey"] + meta["n_gate"] + meta["n_echo"]
        counts = [
            _single_qubit_counts(job_result[start + i], all_shots[start + i])
            for i in range(n_circ)
        ]
        per_qubit_counts.append(counts)
    return per_qubit_counts


def predict_and_residualize(counts_list: list, meta: dict,
                             result, profile) -> dict:
    arch = profile.constants
    p0g1 = arch.get("p0_given_1", 0.0)
    p1g0 = arch.get("p1_given_0", 0.0)
    n_t1, n_ram   = meta["n_t1"], meta["n_ramsey"]
    n_gate, n_echo = meta["n_gate"], meta["n_echo"]

    t1_c     = counts_list[:n_t1]
    ramsey_c = counts_list[n_t1:n_t1 + n_ram]
    gate_c   = counts_list[n_t1 + n_ram:n_t1 + n_ram + n_gate]
    echo_c   = counts_list[n_t1 + n_ram + n_gate:]
    records  = []

    def _rec(probe, cond, p_meas, p_model, sh):
        sigma = max(np.sqrt(p_model * (1 - p_model) / sh), 1e-4) if sh > 0 else np.nan
        z     = (p_meas - p_model) / sigma if sh > 0 else np.nan
        return dict(probe=probe, condition=float(cond),
                    p_meas=float(p_meas), p_model=float(p_model),
                    sigma=float(sigma), z=float(z), shots=int(sh))

    for d, c in zip(meta["t1_delays_s"], t1_c):
        sh = c["0"] + c["1"]
        pm = c["1"] / sh if sh else np.nan
        records.append(_rec("T1", d, pm,
                            float(inv.forward_t1(d, result.T1_s, p0g1, p1g0)), sh))

    T2r = result.T2_ramsey_s or result.T2_s
    for i, t in enumerate(meta["ramsey_delays_s"]):
        px, py = inv.forward_ramsey_xy(t, T2r, result.delta_omega)
        for label, c, pm_model in [("Ramsey-X", ramsey_c[2 * i],     px),
                                    ("Ramsey-Y", ramsey_c[2 * i + 1], py)]:
            sh = c["0"] + c["1"]
            pm = c["1"] / sh if sh else np.nan
            records.append(_rec(label, t, pm, float(pm_model), sh))

    for N, c in zip(meta["gate_rep_N"], gate_c):
        sh      = c["0"] + c["1"]
        pm      = c["0"] / sh if sh else np.nan
        p_model = float(inv.forward_gate(
            np.array([N]), result.epsilon_sx, p0g1, p1g0)[0])
        records.append(_rec("Gate", int(N), pm, p_model, sh))

    T2e = result.T2_echo_s or min(result.T2_s, 2.0 * result.T1_s)
    for d, c in zip(meta["echo_delays_s"], echo_c):
        sh = c["0"] + c["1"]
        pm = c["1"] / sh if sh else np.nan
        records.append(_rec("Echo", d, pm,
                            float(inv.forward_echo(d, T2e, p0g1, p1g0)), sh))

    z = np.array([r["z"] for r in records if np.isfinite(r["z"])])
    return dict(records=records,
                chi2_dof_joint=float(np.mean(z ** 2)) if len(z) else np.nan,
                n_conditions=len(records))


def check_admissibility(result) -> dict:
    T1, T2 = result.T1_s, result.T2_s
    ok = T1 > 0 and T2 > 0 and T2 <= 2.0 * T1 * 1.001
    return dict(T1_positive=T1 > 0, T2_positive=T2 > 0,
                T2_le_2T1=T2 <= 2.0 * T1 * 1.001,
                T2_over_2T1_ratio=float(T2 / (2.0 * T1)) if T1 > 0 else np.nan,
                all_admissible=bool(ok))


def _hypergeometric_split(counts: dict, fit_frac: float, rng) -> tuple:
    n0, n1 = counts["0"], counts["1"]
    n_fit  = int(round((n0 + n1) * fit_frac))
    fit_n1 = int(rng.hypergeometric(ngood=n1, nbad=n0, nsample=n_fit))
    fit_n0 = n_fit - fit_n1
    return ({"0": fit_n0, "1": fit_n1}, {"0": n0 - fit_n0, "1": n1 - fit_n1})


def split_counts_list(counts_list: list, fit_frac: float, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    fit, tst = [], []
    for c in counts_list:
        f, t = _hypergeometric_split(c, fit_frac, rng)
        fit.append(f); tst.append(t)
    return fit, tst


def _avg_shots(counts_list: list) -> int:
    if not counts_list:
        return 1
    return max(1, int(round(np.mean([c["0"] + c["1"] for c in counts_list]))))


def evaluate_holdout(test_counts: list, meta: dict,
                     fit_result, profile) -> dict:
    pred = predict_and_residualize(test_counts, meta, fit_result, profile)
    z    = np.array([r["z"] for r in pred["records"] if np.isfinite(r["z"])])
    cov  = float(np.mean(np.abs(z) <= 1.96)) if len(z) else np.nan
    return dict(chi2_dof_predictive=pred["chi2_dof_joint"],
                coverage_95=cov, n_conditions=pred["n_conditions"])


def invert_and_analyze(physical_qubits: list, per_qubit_metas: list,
                       per_qubit_counts: list, per_qubit_profiles: list,
                       seed: int) -> list:
    DS      = DEFAULT_SHOTS
    results = []

    for local_q, (phys_q, meta, counts_list, profile) in enumerate(
            zip(physical_qubits, per_qubit_metas,
                per_qubit_counts, per_qubit_profiles)):

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            result_full = inv.lindblad_inversion(
                counts_list, meta, profile,
                shots_t1=DS["t1"], shots_ramsey=DS["ramsey"],
                shots_gate=DS["gate"], shots_echo=DS["echo"],
                qubit_id=phys_q,
                timestamp=datetime.now(timezone.utc).isoformat())

        model_fit     = predict_and_residualize(
            counts_list, meta, result_full, profile)
        admissibility = check_admissibility(result_full)

        fit_c, test_c = split_counts_list(
            counts_list, FIT_FRACTION, seed=seed + local_q)
        n_t1  = meta["n_t1"]
        n_ram = meta["n_ramsey"]
        n_g   = meta["n_gate"]
        fit_sh = dict(
            t1     = _avg_shots(fit_c[:n_t1]),
            ramsey = _avg_shots(fit_c[n_t1:n_t1 + n_ram]),
            gate   = _avg_shots(fit_c[n_t1 + n_ram:n_t1 + n_ram + n_g]),
            echo   = _avg_shots(fit_c[n_t1 + n_ram + n_g:]))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            result_fit = inv.lindblad_inversion(
                fit_c, meta, profile,
                shots_t1=fit_sh["t1"], shots_ramsey=fit_sh["ramsey"],
                shots_gate=fit_sh["gate"], shots_echo=fit_sh["echo"],
                qubit_id=phys_q,
                timestamp=datetime.now(timezone.utc).isoformat())

        holdout = evaluate_holdout(test_c, meta, result_fit, profile)

        print(f"  Q{phys_q:>3d}: "
              f"T1={result_full.T1_s:.4g}s  T2={result_full.T2_s:.4g}s  "
              f"Δω={result_full.delta_omega:.4g} rad/s  "
              f"ε_sx={result_full.epsilon_sx:.3e}  "
              f"χ²={model_fit['chi2_dof_joint']:.2f}  "
              f"cov95={holdout['coverage_95']:.2f}  "
              f"T2≤2T1={'OK' if admissibility['all_admissible'] else 'VIOLATED'}")

        cal_source = profile.calibration_source
        cal        = profile.constants
        results.append(dict(
            qubit_id=phys_q,
            meta=meta,
            calibration=dict(
                T1_prior_s=profile.T1_prior_s,
                T2_prior_s=profile.T2_prior_s,
                p0_given_1=cal.get("p0_given_1", 0.0),
                p1_given_0=cal.get("p1_given_0", 0.0),
                eps_typical=cal.get("eps_typical"),
                prior_confidence=profile.prior_confidence,
                T1_source=cal_source.T1_source if cal_source else "unknown",
                T2_source=cal_source.T2_source if cal_source else "unknown",
                spam_source=cal_source.spam_source if cal_source else "unknown",
            ),
            counts_list=counts_list,
            result_full=dict(
                T1_s=result_full.T1_s,
                T1_sigma_s=result_full.T1_sigma_s,
                T2_s=result_full.T2_s,
                T2_sigma_s=result_full.T2_sigma_s,
                T2_ramsey_s=result_full.T2_ramsey_s,
                T2_ramsey_sigma_s=result_full.T2_ramsey_sigma_s,
                T2_echo_s=result_full.T2_echo_s,
                T2_echo_sigma_s=result_full.T2_echo_sigma_s,
                delta_omega=result_full.delta_omega,
                delta_omega_sigma=result_full.delta_omega_sigma,
                epsilon_sx=result_full.epsilon_sx,
                epsilon_sx_sigma=result_full.epsilon_sx_sigma,
                t1_chi2_dof=result_full.t1_chi2_dof,
                ramsey_chi2_dof=result_full.ramsey_chi2_dof,
                gate_chi2_dof=result_full.gate_chi2_dof,
                echo_chi2_dof=result_full.echo_chi2_dof,
            ),
            model_fit=model_fit,
            admissibility=admissibility,
            holdout=holdout,
        ))
    return results


def compare_runs(run_a_path: pathlib.Path, run_b_data: dict) -> None:
    if not run_a_path.exists():
        return
    with open(run_a_path) as f:
        run_a = json.load(f)
    a_by_q = {q["qubit_id"]: q for q in run_a["qubits"]}
    params = [("T1_s", "T1_sigma_s", "T1 (s)"),
              ("T2_s", "T2_sigma_s", "T2 (s)"),
              ("delta_omega", "delta_omega_sigma", "Δω (rad/s)"),
              ("epsilon_sx", "epsilon_sx_sigma", "ε_sx")]
    print("\n" + "=" * 72)
    print("  RUN A vs RUN B — z_j = (A−B)/√(σ_A²+σ_B²)")
    print("=" * 72)
    for q in run_b_data["qubits"]:
        qid = q["qubit_id"]
        if qid not in a_by_q:
            continue
        a = a_by_q[qid]["result_full"]
        b = q["result_full"]
        print(f"\n  Q{qid}:")
        for vk, sk, lbl in params:
            va, sa = a[vk], a[sk]
            vb, sb = b[vk], b[sk]
            denom  = np.sqrt(sa ** 2 + sb ** 2)
            z      = (va - vb) / denom if denom > 0 and np.isfinite(denom) else np.nan
            flag   = "OK" if np.isfinite(z) and abs(z) <= 2.0 else "DRIFT"
            print(f"    {lbl:14s}  A={va:.4g}±{sa:.2g}  "
                  f"B={vb:.4g}±{sb:.2g}  z={z:+.2f}  {flag}")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_fez",
                    choices=["ibm_fez", "ibm_kingston", "ibm_marrakesh"])
    ap.add_argument("--instance", default="open-instance")

    qubit_group = ap.add_mutually_exclusive_group(required=True)
    qubit_group.add_argument("--qubits")
    qubit_group.add_argument("--n-qubits", type=int)

    ap.add_argument("--run-label", required=True, choices=["A", "B"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}] "
          f"5_ibm_hardware.py  Run {args.run_label}")

    if args.dry_run:
        backend       = get_dry_run_backend()
        backend_label = "dry-run"
        print(f"  Mode: {backend_label}")
        props = None
    else:
        print(f"  Backend : {args.backend}  ({args.instance})")
        print("  Connecting... ", end="", flush=True)
        backend = get_backend(args.backend, args.instance)
        print(f"done  [{backend.num_qubits}-qubit Heron r2]")
        backend_label = args.backend
        props = backend.properties()

    if args.qubits:
        qubits = [int(q.strip()) for q in args.qubits.split(",")]
        print(f"  Qubits  : {qubits}  (explicit)")
    else:
        qubits = select_qubits(backend, args.n_qubits)
    nq = len(qubits)

    print(f"\n  Loading per-qubit live profiles...")
    per_qubit_profiles = load_profiles(backend, qubits, props, args.dry_run)

    pubs, all_shots, per_qubit_metas, cpq = build_and_transpile(
        per_qubit_profiles, qubits, backend, args.dry_run)

    DS = DEFAULT_SHOTS
    shots_per_q = (per_qubit_metas[0]["n_t1"]     * DS["t1"]     +
                   per_qubit_metas[0]["n_ramsey"]  * DS["ramsey"] +
                   per_qubit_metas[0]["n_gate"]    * DS["gate"]   +
                   per_qubit_metas[0]["n_echo"]    * DS["echo"])

    print(f"\n  Strategy     : {nq} qubits × {cpq} circuits = {len(pubs)} PUBs, 1 job")
    print(f"  Shots/qubit  : {shots_per_q:,}  "
          f"(T1={DS['t1']} Ramsey={DS['ramsey']} Gate={DS['gate']} Echo={DS['echo']})")
    print(f"  Delays       : per-qubit, matched to each qubit's live T1/T2 prior")
    print()

    job_result = submit(backend, pubs, args.dry_run)

    print("  Reassembling per-qubit counts... ", end="", flush=True)
    per_qubit_counts = reassemble(
        job_result, all_shots, per_qubit_metas, qubits, cpq)
    print("done")

    print("\nRunning Lindblad inversion per qubit...")
    qubit_results = invert_and_analyze(
        qubits, per_qubit_metas, per_qubit_counts,
        per_qubit_profiles, args.seed)

    label_str = "-".join(map(str, qubits))
    out_path  = DATA_DIR / f"run_{args.run_label}_{args.backend}_q{label_str}.json"
    payload   = dict(
        run_label=args.run_label,
        backend=backend_label,
        architecture=ARCHITECTURE,
        timestamp=datetime.now(timezone.utc).isoformat(),
        qubits=qubit_results,
    )
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved: {out_path}")

    if args.run_label == "B":
        run_a = DATA_DIR / f"run_A_{args.backend}_q{label_str}.json"
        compare_runs(run_a, payload)

    print(f"\n  Done. {len(qubit_results)}/{nq} qubits completed.")


if __name__ == "__main__":
    main()