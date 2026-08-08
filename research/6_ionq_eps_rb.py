import argparse
import importlib.util
import json
import os
import pathlib
import sys
import time
import warnings
from datetime import datetime, timezone

import certifi
import urllib3
urllib3.disable_warnings()
os.environ["SSL_CERT_FILE"]      = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import numpy as np
from scipy.optimize import curve_fit
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford, Clifford
from qiskit_ionq import IonQProvider

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "inversion", SCRIPT_DIR / "1_inversion.py")
inv = importlib.util.module_from_spec(_spec)
sys.modules["inversion"] = inv
_spec.loader.exec_module(inv)

DATA_DIR = SCRIPT_DIR / "data" / "hardware"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ARCH          = inv.ARCH_DEFAULTS["trapped_ion"]
P0G1          = ARCH["p0_given_1"]
P1G0          = ARCH["p1_given_0"]
COST_PER_SHOT = 0.08

GATE_REP_SHOTS = 1000
RB_SHOTS       = 500
RB_DEPTHS      = [1, 2, 4, 8, 16, 32, 64, 128]
RB_SEQUENCES   = 10
SEED           = 46


def get_backend(backend_arg: str, api_key: str):
    provider = IonQProvider(api_key)
    if backend_arg.startswith("ionq_simulator:"):
        b = provider.get_backend("ionq_simulator")
        b.set_options(noise_model=backend_arg.split(":", 1)[1])
        return b
    return provider.get_backend(backend_arg)


def is_real_qpu(backend_arg: str) -> bool:
    return backend_arg.startswith("qpu.")


def submit_and_wait(backend, circuit, shots: int):
    from qiskit import transpile as qk_transpile
    tqc = qk_transpile(circuit, backend=backend, optimization_level=0)
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


def gate_rep_circuit(N: int, n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    for _ in range(2 * N):
        for q in range(n_qubits):
            qc.x(q)
        qc.barrier(*range(n_qubits))
    qc.measure(list(range(n_qubits)), list(range(n_qubits)))
    return qc


def fit_eps_sx(Nv: list, p0_measured: list) -> tuple:
    def model(N, eps):
        return 0.5 * (1.0 + (1.0 - 2.0 * eps) ** (2.0 * np.array(N, dtype=float)))
    try:
        popt, pcov = curve_fit(model, Nv, p0_measured,
                               p0=[1e-4], bounds=(0, 0.5),
                               sigma=[max(np.sqrt(p * (1 - p) / GATE_REP_SHOTS), 1e-4)
                                      for p in p0_measured],
                               absolute_sigma=True)
        return float(popt[0]), float(np.sqrt(pcov[0, 0]))
    except Exception:
        return float("nan"), float("nan")


def rb_circuit(depth: int, rng, n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
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


def fit_rb(depths: list, survival: list) -> tuple:
    def model(m, A, p, B):
        return A * np.array(p) ** np.array(m) + B
    try:
        popt, pcov = curve_fit(model, depths, survival,
                               p0=[0.5, 0.99, 0.5],
                               bounds=([0, 0, 0], [1, 1, 1]))
        A, p, B = popt
        epc = (1.0 - p) / 2.0
        epc_sigma = float(np.sqrt(pcov[1, 1])) / 2.0
        avg_gates_per_clifford = 1.875
        eps_rb     = epc / avg_gates_per_clifford
        eps_rb_sig = epc_sigma / avg_gates_per_clifford
        return float(p), float(epc), float(eps_rb), float(eps_rb_sig)
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan")


def run_gate_rep(backend, qubits: list) -> dict:
    profile = inv.BackendProfile.from_architecture("trapped_ion")
    Nv      = profile.gate_rep_n
    n_q     = len(qubits)
    print(f"  Gate rep: N={Nv}, {GATE_REP_SHOTS} shots × {n_q} qubits parallel")

    per_qubit_p0 = [[] for _ in range(n_q)]

    for N in Nv:
        qc   = gate_rep_circuit(N, n_q)
        res  = submit_and_wait(backend, qc, GATE_REP_SHOTS)
        pq   = counts_to_per_qubit(res, n_q, GATE_REP_SHOTS)
        for qi in range(n_q):
            sh   = pq[qi]["0"] + pq[qi]["1"]
            p0   = pq[qi]["0"] / sh if sh > 0 else 0.5
            per_qubit_p0[qi].append(p0)
        print(f"    N={N} done")

    results = {}
    for qi, q_id in enumerate(qubits):
        eps, eps_sig = fit_eps_sx(Nv, per_qubit_p0[qi])
        print(f"    Qubit {q_id}: eps_sx={eps:.4e} ± {eps_sig:.2e}  "
              f"p0 measured: {[f'{v:.3f}' for v in per_qubit_p0[qi]]}")
        results[q_id] = dict(Nv=Nv, p0_measured=per_qubit_p0[qi],
                             eps_sx=eps, eps_sx_sigma=eps_sig)
    return results


def run_rb(backend, qubits: list) -> dict:
    rng = np.random.default_rng(SEED)
    n_q = len(qubits)
    print(f"  RB: depths={RB_DEPTHS}, {RB_SEQUENCES} sequences, "
          f"{RB_SHOTS} shots, {n_q} qubits parallel")

    per_qubit_survival = {q: {d: [] for d in RB_DEPTHS} for q in qubits}

    for seq_idx in range(RB_SEQUENCES):
        for depth in RB_DEPTHS:
            qc  = rb_circuit(depth, rng, n_q)
            res = submit_and_wait(backend, qc, RB_SHOTS)
            pq  = counts_to_per_qubit(res, n_q, RB_SHOTS)
            for qi, q_id in enumerate(qubits):
                sh  = pq[qi]["0"] + pq[qi]["1"]
                p0  = pq[qi]["0"] / sh if sh > 0 else 0.5
                per_qubit_survival[q_id][depth].append(p0)
        print(f"    Sequence {seq_idx+1}/{RB_SEQUENCES} done")

    results = {}
    for q_id in qubits:
        mean_survival = [float(np.mean(per_qubit_survival[q_id][d]))
                         for d in RB_DEPTHS]
        p_decay, epc, eps_rb, eps_rb_sig = fit_rb(RB_DEPTHS, mean_survival)
        print(f"    Qubit {q_id}: EPC={epc:.4e}  eps_rb={eps_rb:.4e} ± {eps_rb_sig:.2e}  "
              f"p_decay={p_decay:.6f}")
        results[q_id] = dict(depths=RB_DEPTHS,
                             mean_survival=mean_survival,
                             p_decay=p_decay, epc=epc,
                             eps_rb=eps_rb, eps_rb_sigma=eps_rb_sig)
    return results


def compare_and_print(gate_rep_results: dict, rb_results: dict, qubits: list):
    print("\n" + "=" * 72)
    print("  CANARY eps_sx  vs  RB eps_sx  —  PER QUBIT")
    print("=" * 72)
    print(f"  {'Qubit':>6}  {'Canary eps_sx':>16}  {'RB eps_sx':>16}  "
          f"{'Ratio C/RB':>12}  {'Agreement':>10}")
    print("-" * 72)
    for q_id in qubits:
        c_eps = gate_rep_results[q_id]["eps_sx"]
        c_sig = gate_rep_results[q_id]["eps_sx_sigma"]
        r_eps = rb_results[q_id]["eps_rb"]
        r_sig = rb_results[q_id]["eps_rb_sigma"]
        ratio = c_eps / r_eps if r_eps > 0 else float("nan")
        denom = np.sqrt(c_sig**2 + r_sig**2)
        z     = (c_eps - r_eps) / denom if denom > 0 else float("nan")
        flag  = "OK" if abs(z) <= 2.0 else "CHECK"
        print(f"  {q_id:>6}  {c_eps:>14.4e}    {r_eps:>14.4e}  "
              f"  {ratio:>10.3f}  {flag:>10}  (z={z:+.2f})")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", required=True)
    ap.add_argument("--backend", default="ionq_simulator:forte-1")
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()

    qubits = [int(q.strip()) for q in args.qubits.split(",")]

    if is_real_qpu(args.backend) and not args.live:
        sys.exit(f"ERROR: '{args.backend}' is real hardware. Add --live.")

    api_key = os.environ.get("IONQ_API_KEY")
    if not api_key:
        sys.exit("ERROR: IONQ_API_KEY not set. "
                 "PowerShell: $env:IONQ_API_KEY = 'your_key'")

    profile  = inv.BackendProfile.from_architecture("trapped_ion")
    Nv       = profile.gate_rep_n
    n_q      = len(qubits)
    gate_jobs = len(Nv)
    rb_jobs   = RB_SEQUENCES * len(RB_DEPTHS)
    total_shots = (gate_jobs * GATE_REP_SHOTS +
                   rb_jobs   * RB_SHOTS)

    print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}] "
          f"6_ionq_eps_rb.py")
    print(f"  Backend      : {args.backend}")
    print(f"  Qubits       : {qubits}  (n={n_q}, parallel)")
    print(f"  Gate rep N   : {Nv}  ({GATE_REP_SHOTS} shots/circuit)")
    print(f"  RB depths    : {RB_DEPTHS}  ({RB_SEQUENCES} sequences × {RB_SHOTS} shots)")
    print(f"  Total shots  : {total_shots:,}")

    if is_real_qpu(args.backend):
        est = total_shots * COST_PER_SHOT
        print(f"  Est. cost    : ${est:,.2f}")
        if input("  Type RUN to confirm: ").strip() != "RUN":
            sys.exit("Aborted.")
    else:
        nm = args.backend.split(":", 1)[1] if ":" in args.backend else "noiseless"
        print(f"  Est. cost    : $0.00  (simulator, {nm})")
    print()

    backend = get_backend(args.backend, api_key)

    print("Running Canary gate repetition...")
    gate_rep_results = run_gate_rep(backend, qubits)

    print("\nRunning Clifford RB...")
    rb_results = run_rb(backend, qubits)

    compare_and_print(gate_rep_results, rb_results, qubits)

    out = DATA_DIR / f"ionq_eps_rb_{'- '.join(map(str, qubits))}.json"
    with open(out, "w") as f:
        json.dump(dict(backend=args.backend,
                       timestamp=datetime.now(timezone.utc).isoformat(),
                       qubits=qubits,
                       gate_rep=gate_rep_results,
                       rb=rb_results), f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()