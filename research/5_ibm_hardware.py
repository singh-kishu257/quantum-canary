import argparse
import importlib.util
import json
import os
import pathlib
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import curve_fit

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "inversion", SCRIPT_DIR / "1_inversion.py")
inv = importlib.util.module_from_spec(_spec)
sys.modules["inversion"] = inv
_spec.loader.exec_module(inv)

DATA_DIR = SCRIPT_DIR / "data" / "hardware"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ARCH           = inv.ARCH_DEFAULTS["trapped_ion"]
GATE_REP_SHOTS = 500
RB_SHOTS       = 300
RB_DEPTHS      = [1, 2, 4, 8, 16, 32, 64]
RB_SEQUENCES   = 5
SEED           = 46
COST_PER_SHOT  = 0.08


def get_backend(backend_arg: str, api_key: str):
    from qiskit_ionq import IonQProvider
    provider = IonQProvider(api_key)
    if backend_arg.startswith("ionq_simulator:"):
        b = provider.get_backend("ionq_simulator")
        b.set_options(noise_model=backend_arg.split(":", 1)[1])
        return b
    return provider.get_backend(backend_arg)


def is_real_qpu(backend_arg: str) -> bool:
    return backend_arg.startswith("qpu.")


def ionq_backend_name(backend_arg: str) -> str:
    if backend_arg.startswith("ionq_simulator:"):
        return backend_arg.split(":", 1)[1]
    if backend_arg.startswith("qpu."):
        return backend_arg[4:]
    return "forte-1"


def load_per_qubit_profiles(qubits: list, api_key: str, backend_arg: str) -> dict:
    bname = ionq_backend_name(backend_arg)
    profiles = {}
    for q in qubits:
        p = inv.get_live_profile(
            f"ionq:{bname}", qubit_id=q, ionq_token=api_key)
        profiles[q] = p
    return profiles


def gate_rep_circuit_nq(N: int, n_qubits: int):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n_qubits, n_qubits, name=f"gate_rep_N{N}")
    for _ in range(N):
        for q in range(n_qubits):
            qc.sx(q)
            qc.sxdg(q)
    qc.measure(list(range(n_qubits)), list(range(n_qubits)))
    return qc


def rb_circuit_nq(depth: int, seed: int, n_qubits: int):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import random_clifford, Clifford
    rng = np.random.default_rng(seed)
    qc    = QuantumCircuit(n_qubits, n_qubits, name=f"rb_d{depth}_s{seed}")
    total = Clifford(QuantumCircuit(1))
    for _ in range(depth):
        c = random_clifford(1, seed=int(rng.integers(0, 2**31)))
        for q in range(n_qubits):
            qc.compose(c.to_circuit(), qubits=[q], inplace=True)
        total = total.compose(c)
    inv_c = total.adjoint()
    for q in range(n_qubits):
        qc.compose(inv_c.to_circuit(), qubits=[q], inplace=True)
    qc.measure(list(range(n_qubits)), list(range(n_qubits)))
    return qc


def submit_and_wait(backend, circuit, shots: int):
    from qiskit import transpile
    tqc = transpile(circuit, backend=backend, optimization_level=0)
    try:
        job = backend.run(tqc, shots=shots,
                          error_mitigation={"debias": False, "sharpen": False})
    except TypeError:
        job = backend.run(tqc, shots=shots)
    return job.result()


def counts_to_per_qubit(result, n_qubits: int, shots: int) -> list:
    counts = result.get_counts()
    per_q  = [{"0": 0, "1": 0} for _ in range(n_qubits)]
    for bitstring, count in counts.items():
        bs = bitstring.replace(" ", "").zfill(n_qubits)
        for q in range(n_qubits):
            bit = bs[-(q + 1)]
            if bit in ("0", "1"):
                per_q[q][bit] += count
    for q in range(n_qubits):
        total = per_q[q]["0"] + per_q[q]["1"]
        if total < shots:
            per_q[q]["0"] += shots - total
    return per_q


def save_checkpoint(path: pathlib.Path, data: dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def load_checkpoint(path: pathlib.Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def run_gate_rep(backend, qubits: list, profiles: dict,
                 chk: dict, chk_path: pathlib.Path) -> dict:
    n_q   = len(qubits)
    first = profiles[qubits[0]]
    Nv    = first.gate_rep_n

    print(f"  Gate rep (SX·SX† pairs): N={Nv}")
    print(f"  {GATE_REP_SHOTS} shots × {n_q} qubits parallel")

    p0_by_N = chk.get("gate_rep_p0_by_N", {})
    done    = set(str(k) for k in p0_by_N.keys())

    for N in Nv:
        if str(N) in done:
            print(f"    N={N}: loaded from checkpoint")
            continue
        try:
            qc     = gate_rep_circuit_nq(N, n_q)
            result = submit_and_wait(backend, qc, GATE_REP_SHOTS)
            per_q  = counts_to_per_qubit(result, n_q, GATE_REP_SHOTS)
            row    = {}
            for qi, q_id in enumerate(qubits):
                sh = per_q[qi]["0"] + per_q[qi]["1"]
                p0 = per_q[qi]["0"] / sh if sh > 0 else 0.5
                row[q_id] = p0
            p0_by_N[str(N)] = row
            chk["gate_rep_p0_by_N"] = p0_by_N
            save_checkpoint(chk_path, chk)
            print(f"    N={N}: p0={[f'{row[q]:.3f}' for q in qubits]}")
        except Exception as e:
            print(f"    N={N} FAILED — {e}  (checkpoint saved, aborting gate rep)")
            chk["gate_rep_p0_by_N"] = p0_by_N
            save_checkpoint(chk_path, chk)
            break

    results = {}
    for q_id in qubits:
        profile = profiles[q_id]
        cal     = profile.constants
        p0g1    = cal["p0_given_1"]
        p1g0    = cal["p1_given_0"]
        eps0    = cal["eps_typical"]

        Nv_done  = [N for N in Nv if str(N) in p0_by_N]
        p0_meas  = [p0_by_N[str(N)][q_id] for N in Nv_done]

        if len(Nv_done) < 2:
            print(f"    Q{q_id}: insufficient data")
            results[q_id] = dict(Nv=Nv_done, p0_measured=p0_meas,
                                 eps_sx=float("nan"), eps_sx_sigma=float("nan"),
                                 p0g1=p0g1, p1g0=p1g0)
            continue

        sigma = [max(np.sqrt(p * (1 - p) / GATE_REP_SHOTS), 1e-4)
                 for p in p0_meas]
        try:
            def model(Nv, eps):
                return np.array([
                    float(inv.forward_gate(np.array([n]), eps, p0g1, p1g0)[0])
                    for n in Nv
                ])
            popt, pcov = curve_fit(
                model, Nv_done, p0_meas,
                p0=[eps0], bounds=(0, ARCH["eps_max"]),
                sigma=sigma, absolute_sigma=True)
            eps     = float(popt[0])
            eps_sig = float(np.sqrt(pcov[0, 0]))
        except Exception:
            eps, eps_sig = float("nan"), float("nan")

        print(f"    Q{q_id}: eps_sx={eps:.4e}±{eps_sig:.2e}  "
              f"p0={[f'{v:.3f}' for v in p0_meas]}  "
              f"p0|1={p0g1:.4f}  p1|0={p1g0:.4f}  eps0={eps0:.2e}")

        results[q_id] = dict(Nv=Nv_done, p0_measured=p0_meas,
                             eps_sx=eps, eps_sx_sigma=eps_sig,
                             p0g1=p0g1, p1g0=p1g0, eps_prior=eps0)
    return results, Nv


def fit_rb(depths: list, survival: list) -> tuple:
    def model(m, A, p, B):
        return A * np.array(p) ** np.array(m) + B
    try:
        popt, pcov = curve_fit(model, depths, survival,
                               p0=[0.5, 0.99, 0.5],
                               bounds=([0, 0, 0], [1, 1, 1]))
        A, p, B     = popt
        epc         = (1.0 - p) / 2.0
        epc_sigma   = float(np.sqrt(pcov[1, 1])) / 2.0
        eps_rb      = epc / 1.875
        eps_rb_sig  = epc_sigma / 1.875
        return float(p), float(epc), float(eps_rb), float(eps_rb_sig)
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan")


def run_rb(backend, qubits: list, chk: dict, chk_path: pathlib.Path) -> dict:
    n_q      = len(qubits)
    survival = chk.get("rb_survival", {str(q): {str(d): [] for d in RB_DEPTHS}
                                       for q in qubits})
    done_seqs = chk.get("rb_done_seqs", 0)

    print(f"  RB: depths={RB_DEPTHS}  {RB_SEQUENCES} seqs  {RB_SHOTS} shots  {n_q}q")
    if done_seqs > 0:
        print(f"  Resuming from sequence {done_seqs+1}/{RB_SEQUENCES}")

    for seq_idx in range(done_seqs, RB_SEQUENCES):
        seq_failed = False
        for depth in RB_DEPTHS:
            seq_seed = SEED + seq_idx * 10000 + depth
            try:
                qc     = rb_circuit_nq(depth, seq_seed, n_q)
                result = submit_and_wait(backend, qc, RB_SHOTS)
                per_q  = counts_to_per_qubit(result, n_q, RB_SHOTS)
                for qi, q_id in enumerate(qubits):
                    sh = per_q[qi]["0"] + per_q[qi]["1"]
                    p0 = per_q[qi]["0"] / sh if sh > 0 else 0.5
                    survival[str(q_id)][str(depth)].append(p0)
            except Exception as e:
                print(f"    Seq {seq_idx+1} depth {depth} FAILED — {e}")
                seq_failed = True
                break

        chk["rb_survival"]  = survival
        chk["rb_done_seqs"] = seq_idx + (0 if seq_failed else 1)
        save_checkpoint(chk_path, chk)

        if seq_failed:
            print(f"    Checkpoint saved after partial sequence {seq_idx+1}.")
            break
        print(f"    Sequence {seq_idx+1}/{RB_SEQUENCES} done")

    results = {}
    for q_id in qubits:
        depths_with_data = [d for d in RB_DEPTHS
                            if len(survival[str(q_id)][str(d)]) > 0]
        if len(depths_with_data) < 3:
            print(f"    Q{q_id}: insufficient RB data ({len(depths_with_data)} depths)")
            results[q_id] = dict(depths=depths_with_data, mean_survival=[],
                                 p_decay=float("nan"), epc=float("nan"),
                                 eps_rb=float("nan"), eps_rb_sigma=float("nan"))
            continue
        mean_surv = [float(np.mean(survival[str(q_id)][str(d)]))
                     for d in depths_with_data]
        p_decay, epc, eps_rb, eps_rb_sig = fit_rb(depths_with_data, mean_surv)
        print(f"    Q{q_id}: EPC={epc:.4e}  eps_rb={eps_rb:.4e}±{eps_rb_sig:.2e}")
        results[q_id] = dict(depths=depths_with_data, mean_survival=mean_surv,
                             p_decay=p_decay, epc=epc,
                             eps_rb=eps_rb, eps_rb_sigma=eps_rb_sig)
    return results


def compare_and_print(gate_rep: dict, rb: dict, qubits: list):
    print("\n" + "=" * 72)
    print("  CANARY ε_sx  (SX·SX†)  vs  RB ε_sx  —  PER QUBIT")
    print("=" * 72)
    print(f"  {'Q':>4}  {'Canary':>14}  {'RB':>14}  {'C/RB':>8}  {'z':>6}  {'OK?':>6}")
    print("-" * 72)
    for q_id in qubits:
        c_eps = gate_rep[q_id]["eps_sx"]
        c_sig = gate_rep[q_id]["eps_sx_sigma"]
        r_eps = rb[q_id]["eps_rb"]
        r_sig = rb[q_id]["eps_rb_sigma"]
        ratio = c_eps / r_eps if r_eps > 0 else float("nan")
        denom = np.sqrt(c_sig ** 2 + r_sig ** 2)
        z     = ((c_eps - r_eps) / denom
                 if denom > 0 and np.isfinite(denom) else float("nan"))
        flag  = "OK" if np.isfinite(z) and abs(z) <= 2.0 else "CHECK"
        print(f"  {q_id:>4}  {c_eps:>14.4e}  {r_eps:>14.4e}  "
              f"{ratio:>8.3f}  {z:>+6.2f}  {flag:>6}")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits",          required=True,
                    help="Comma-separated qubit indices, e.g. 0,1,2,3,4,5,6,7")
    ap.add_argument("--backend",         default="ionq_simulator:forte-1",
                    help="ionq_simulator:forte-1 | ionq_simulator | qpu.forte-enterprise-1")
    ap.add_argument("--live",            action="store_true",
                    help="Required when targeting a real QPU (qpu.*)")
    ap.add_argument("--gate-rep-shots",  type=int, default=GATE_REP_SHOTS)
    ap.add_argument("--rb-shots",        type=int, default=RB_SHOTS)
    ap.add_argument("--rb-seqs",         type=int, default=RB_SEQUENCES)
    ap.add_argument("--resume",          default=None,
                    help="Path to an existing checkpoint JSON to resume from")
    args = ap.parse_args()

    qubits = [int(q.strip()) for q in args.qubits.split(",")]

    if is_real_qpu(args.backend) and not args.live:
        sys.exit(f"ERROR: '{args.backend}' is real QPU. Add --live to confirm.")

    api_key = os.environ.get("IONQ_API_KEY")
    if not api_key:
        sys.exit("ERROR: IONQ_API_KEY not set.\n"
                 "PowerShell: $env:IONQ_API_KEY = 'your_key'")

    ts           = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    chk_path     = (pathlib.Path(args.resume) if args.resume
                    else DATA_DIR / f"checkpoint_ionq_{ts}.json")
    chk          = load_checkpoint(chk_path)

    n_gate_circ  = 3
    n_rb_circ    = args.rb_seqs * len(RB_DEPTHS)
    total_shots  = (n_gate_circ * args.gate_rep_shots +
                    n_rb_circ   * args.rb_shots)

    print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}] "
          f"6_ionq_eps_rb.py")
    print(f"  Backend        : {args.backend}")
    print(f"  Qubits         : {qubits}  (n={len(qubits)}, parallel per circuit)")
    print(f"  Gate rep shots : {args.gate_rep_shots}/circuit  (3 circuits, SX·SX† pairs)")
    print(f"  RB             : depths={RB_DEPTHS}")
    print(f"  RB shots       : {args.rb_shots}/circuit  "
          f"({args.rb_seqs} seqs × {len(RB_DEPTHS)} depths)")
    print(f"  Total shots    : {total_shots:,}")
    print(f"  Error mitig.   : OFF  (debias=False, sharpen=False)")
    print(f"  Opt level      : 0")
    print(f"  Checkpoint     : {chk_path}")

    if is_real_qpu(args.backend):
        est = total_shots * COST_PER_SHOT
        print(f"  Est. cost      : ${est:,.2f}  (at ${COST_PER_SHOT}/shot)")
        if input("  Type RUN to confirm: ").strip() != "RUN":
            sys.exit("Aborted.")
    else:
        nm = (args.backend.split(":", 1)[1] if ":" in args.backend
              else "noiseless")
        print(f"  Est. cost      : $0.00  (simulator, {nm})")
    print()

    backend = get_backend(args.backend, api_key)

    print("Loading per-qubit live profiles from IonQ API...")
    profiles = load_per_qubit_profiles(qubits, api_key,
                                       args.backend if is_real_qpu(args.backend)
                                       else args.backend)

    print("\nRunning Canary gate repetition...")
    gate_rep_results, Nv = run_gate_rep(
        backend, qubits, profiles, chk, chk_path)

    print("\nRunning Clifford RB...")
    rb_results = run_rb(backend, qubits, chk, chk_path)

    compare_and_print(gate_rep_results, rb_results, qubits)

    cal_summary = {
        q: dict(
            eps_prior   = profiles[q].constants["eps_typical"],
            p0g1        = profiles[q].constants["p0_given_1"],
            p1g0        = profiles[q].constants["p1_given_0"],
            confidence  = profiles[q].prior_confidence,
        )
        for q in qubits
    }

    out = DATA_DIR / f"ionq_eps_rb_{'-'.join(map(str, qubits))}_{ts}.json"
    with open(out, "w") as f:
        json.dump(dict(
            backend   = args.backend,
            timestamp = datetime.now(timezone.utc).isoformat(),
            qubits    = qubits,
            gate_rep_N = Nv,
            calibration = cal_summary,
            gate_rep  = {str(k): v for k, v in gate_rep_results.items()},
            rb        = {str(k): v for k, v in rb_results.items()},
        ), f, indent=2)
    print(f"\n  Saved: {out}")

    if chk_path.exists():
        chk_path.unlink()
        print(f"  Checkpoint removed: {chk_path}")


if __name__ == "__main__":
    main()