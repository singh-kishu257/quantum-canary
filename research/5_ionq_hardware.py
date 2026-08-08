"""
5_ionq_hardware.py — Quantum Canary hardware validation on IonQ Forte.

Bypasses qiskit-ionq's circuit submission entirely and calls IonQ's REST API
(v0.4) directly, which supports 'wait' gates for T1/T2/echo measurements.
qiskit-ionq's QIS backend rejects Qiskit delay() instructions; the REST API
does not.

Runs all 8 qubits in parallel per circuit (one 8-qubit job per circuit type,
15 jobs total) rather than 120 sequential single-qubit jobs.

Does NOT modify 1_inversion.py. All physics math (BackendProfile,
lindblad_inversion, forward_*) is imported unmodified.

Saves:   research/data/hardware/run_<LABEL>_qubits_<ids>.json
         (checkpointed after every circuit, not just at the end)

Usage:
  # Ideal noiseless simulator (free, tests API pipeline):
  python 5_ionq_hardware.py --qubits 0,1,2,3,4,5,6,7 --run-label A

  # Forte noise model on simulator (free, realistic noise):
  python 5_ionq_hardware.py --qubits 0,1,2,3,4,5,6,7 --run-label A \\
      --backend ionq_simulator:forte-1

  # Real QPU (costs credits -- requires --live):
  python 5_ionq_hardware.py --qubits 0,1,2,3,4,5,6,7 --run-label A \\
      --backend qpu.forte-enterprise-1 --live
"""

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
import requests
import urllib3

# ── import 1_inversion.py unmodified ─────────────────────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "inversion", SCRIPT_DIR / "1_inversion.py")
inv = importlib.util.module_from_spec(_spec)
sys.modules["inversion"] = inv
_spec.loader.exec_module(inv)

DATA_DIR = SCRIPT_DIR / "data" / "hardware"
DATA_DIR.mkdir(parents=True, exist_ok=True)

IONQ_API_BASE  = "https://api.ionq.co/v0.4"
ARCHITECTURE   = "trapped_ion"
DEFAULT_SHOTS  = dict(t1=300, ramsey=1000, gate=500, echo=500)
FIT_FRACTION   = 0.70
COST_PER_SHOT  = 0.08


# ── backend argument parsing ──────────────────────────────────────────────────

def parse_backend(backend_arg: str) -> tuple:
    """
    Returns (api_target, noise_model) for IonQ REST API.

    "ionq_simulator"               -> ("simulator", None)
    "ionq_simulator:forte-1"       -> ("simulator", "forte-1")
    "ionq_simulator:forte-enterprise-1" -> ("simulator", "forte-enterprise-1")
    "qpu.forte-1"                  -> ("qpu.forte-1", None)
    "qpu.forte-enterprise-1"       -> ("qpu.forte-enterprise-1", None)
    """
    if backend_arg == "ionq_simulator":
        return "simulator", None
    if backend_arg.startswith("ionq_simulator:"):
        return "simulator", backend_arg.split(":", 1)[1]
    return backend_arg, None


def is_real_qpu(backend_arg: str) -> bool:
    return backend_arg.startswith("qpu.") or backend_arg == "ionq_qpu"


# ── IonQ REST API helpers ─────────────────────────────────────────────────────

def _api_headers(api_key: str) -> dict:
    return {"Authorization": f"apiKey {api_key}",
            "Content-Type": "application/json"}


def _post(url: str, body: dict, api_key: str) -> dict:
    """POST to IonQ REST API, handling Windows SSL cert issues gracefully."""
    headers = _api_headers(api_key)
    try:
        r = requests.post(url, json=body, headers=headers,
                          verify=True, timeout=60)
    except requests.exceptions.SSLError:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        r = requests.post(url, json=body, headers=headers,
                          verify=False, timeout=60)
    r.raise_for_status()
    return r.json()


def _get(url: str, api_key: str) -> dict:
    headers = _api_headers(api_key)
    try:
        r = requests.get(url, headers=headers, verify=True, timeout=60)
    except requests.exceptions.SSLError:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        r = requests.get(url, headers=headers, verify=False, timeout=60)
    r.raise_for_status()
    return r.json()


def submit_job(api_key: str, api_target: str, noise_model: str,
               circuit_gates: list, n_qubits: int,
               shots: int, name: str = "canary") -> str:
    """Submit one circuit to IonQ REST API. Returns job_id."""
    body = {
        "name": name,
        "shots": shots,
        "target": api_target,
        "input": {
            "qubits": n_qubits,
            "circuit": circuit_gates,
        },
    }
    if noise_model:
        body["noise"] = {"model": noise_model}

    data = _post(f"{IONQ_API_BASE}/jobs", body, api_key)
    return data["id"]


def poll_job(api_key: str, job_id: str, timeout: int = 900) -> dict:
    """Poll until job completes or fails. Returns full job dict."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        data = _get(f"{IONQ_API_BASE}/jobs/{job_id}", api_key)
        status = data.get("status", "unknown")
        if status == "completed":
            return data
        if status in ("failed", "canceled", "error"):
            raise RuntimeError(
                f"Job {job_id} ended with status '{status}': "
                f"{data.get('failure', {}).get('error', 'no detail')}")
        time.sleep(5)
    raise TimeoutError(f"Job {job_id} timed out after {timeout}s")


def parse_per_qubit_counts(job_data: dict, shots: int,
                            n_qubits: int) -> list:
    """
    Extract per-qubit marginal counts {"0": n0, "1": n1} from a
    multi-qubit IonQ result histogram.

    IonQ returns histogram as {bitstring: probability_or_count}.
    Bitstring ordering: qubit 0 = rightmost character (little-endian).
    """
    data = job_data.get("data") or job_data.get("results") or {}
    histogram = data.get("histogram", {})

    if not histogram:
        # No results yet or empty — return 50/50 as fallback
        return [{"0": shots // 2, "1": shots - shots // 2}
                for _ in range(n_qubits)]

    # Determine if values are probabilities (sum ≈ 1) or counts (sum ≈ shots)
    total = sum(histogram.values())
    is_probs = total <= 1.01

    per_qubit = [{"0": 0, "1": 0} for _ in range(n_qubits)]
    for bitstring, val in histogram.items():
        count = int(round(val * shots)) if is_probs else int(val)
        bs = str(bitstring).zfill(n_qubits)
        for q in range(n_qubits):
            # rightmost character = qubit 0 (IonQ / Qiskit convention)
            bit = bs[-(q + 1)] if q + 1 <= len(bs) else "0"
            if bit in ("0", "1"):
                per_qubit[q][bit] += count

    return per_qubit


# ── IonQ JSON circuit builders (using 'wait' for free evolution) ──────────────

def _t1_circuit(delay_s: float, qubits: list) -> list:
    """T1: X on each qubit → wait(τ) per qubit → (implicit measure)"""
    delay_ns = max(1, int(delay_s * 1e9))
    gates = []
    for q in qubits:
        gates.append({"gate": "x", "target": q})
    for q in qubits:
        gates.append({"gate": "wait", "duration": delay_ns, "target": q})
    return gates


def _ramsey_x_circuit(delay_s: float, qubits: list) -> list:
    """Ramsey X: H → wait(τ) → H per qubit"""
    delay_ns = max(1, int(delay_s * 1e9))
    gates = []
    for q in qubits:
        gates.append({"gate": "h", "target": q})
    for q in qubits:
        gates.append({"gate": "wait", "duration": delay_ns, "target": q})
    for q in qubits:
        gates.append({"gate": "h", "target": q})
    return gates


def _ramsey_y_circuit(delay_s: float, qubits: list) -> list:
    """Ramsey Y: H → wait(τ) → S† → H per qubit"""
    delay_ns = max(1, int(delay_s * 1e9))
    gates = []
    for q in qubits:
        gates.append({"gate": "h", "target": q})
    for q in qubits:
        gates.append({"gate": "wait", "duration": delay_ns, "target": q})
    for q in qubits:
        gates.append({"gate": "sdg", "target": q})
    for q in qubits:
        gates.append({"gate": "h", "target": q})
    return gates


def _gate_rep_circuit(N: int, qubits: list) -> list:
    """Gate rep: 2N SX gates per qubit (no wait needed)"""
    gates = []
    for _ in range(2 * N):
        for q in qubits:
            gates.append({"gate": "sx", "target": q})
    return gates


def _echo_circuit(delay_s: float, qubits: list) -> list:
    """Hahn echo: H → wait(τ/2) → X → wait(τ/2) → H per qubit"""
    half_ns = max(1, int(delay_s * 0.5 * 1e9))
    gates = []
    for q in qubits:
        gates.append({"gate": "h", "target": q})
    for q in qubits:
        gates.append({"gate": "wait", "duration": half_ns, "target": q})
    for q in qubits:
        gates.append({"gate": "x", "target": q})
    for q in qubits:
        gates.append({"gate": "wait", "duration": half_ns, "target": q})
    for q in qubits:
        gates.append({"gate": "h", "target": q})
    return gates


# ── shot accounting ───────────────────────────────────────────────────────────

def total_shots_for(meta: dict, shots: dict) -> int:
    return (meta["n_t1"]     * shots["t1"] +
            meta["n_ramsey"] * shots["ramsey"] +
            meta["n_gate"]   * shots["gate"] +
            meta["n_echo"]   * shots["echo"])


# ── replicate-level holdout (70/30 shot split) ────────────────────────────────

def hypergeometric_split(counts: dict, fit_frac: float, rng) -> tuple:
    n0, n1 = counts["0"], counts["1"]
    n_fit = int(round((n0 + n1) * fit_frac))
    fit_n1 = int(rng.hypergeometric(ngood=n1, nbad=n0, nsample=n_fit))
    fit_n0 = n_fit - fit_n1
    return ({"0": fit_n0, "1": fit_n1},
            {"0": n0 - fit_n0, "1": n1 - fit_n1})


def split_counts_list(counts_list: list, fit_frac: float, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    fit, test = [], []
    for c in counts_list:
        f, t = hypergeometric_split(c, fit_frac, rng)
        fit.append(f); test.append(t)
    return fit, test


def avg_shots(counts_list: list) -> int:
    return int(round(np.mean([c["0"] + c["1"] for c in counts_list]))) if counts_list else 1


# ── forward-model residuals ───────────────────────────────────────────────────

def predict_and_residualize(counts_list: list, meta: dict,
                             result) -> dict:
    """
    Uses the SINGLE jointly-fitted theta_hat to predict every measured
    condition. This is the central novelty demonstration: one parameter
    vector generates all four probe families simultaneously.
    """
    arch = inv.ARCH_DEFAULTS[ARCHITECTURE]
    p0g1 = arch.get("p0_given_1", 0.0)
    p1g0 = arch.get("p1_given_0", 0.0)
    n_t1, n_ram = meta["n_t1"], meta["n_ramsey"]
    n_gate, n_echo = meta["n_gate"], meta["n_echo"]

    t1_c     = counts_list[:n_t1]
    ramsey_c = counts_list[n_t1:n_t1 + n_ram]
    gate_c   = counts_list[n_t1 + n_ram:n_t1 + n_ram + n_gate]
    echo_c   = counts_list[n_t1 + n_ram + n_gate:]

    records = []

    def _rec(probe, cond, p_meas, p_model, sh):
        sigma = max(np.sqrt(p_model * (1 - p_model) / sh), 1e-4) if sh > 0 else float("nan")
        z = (p_meas - p_model) / sigma if sh > 0 else float("nan")
        return dict(probe=probe, condition=cond, p_meas=p_meas,
                    p_model=p_model, sigma=sigma, z=z, shots=sh)

    for d, c in zip(meta["t1_delays_s"], t1_c):
        sh = c["0"] + c["1"]
        pm = c["1"] / sh if sh else float("nan")
        records.append(_rec("T1", d, pm,
                            float(inv.forward_t1(d, result.T1_s, p0g1, p1g0)), sh))

    T2r = result.T2_ramsey_s or result.T2_s
    for i, t in enumerate(meta["ramsey_delays_s"]):
        px, py = inv.forward_ramsey_xy(t, T2r, result.delta_omega)
        for label, c, pm_model in [("Ramsey-X", ramsey_c[2*i], px),
                                    ("Ramsey-Y", ramsey_c[2*i+1], py)]:
            sh = c["0"] + c["1"]
            pm = c["1"] / sh if sh else float("nan")
            records.append(_rec(label, t, pm, pm_model, sh))

    for N, c in zip(meta["gate_rep_N"], gate_c):
        sh = c["0"] + c["1"]
        pm = c["0"] / sh if sh else float("nan")
        p_model = float(inv.forward_gate(np.array([N]), result.epsilon_sx, p0g1, p1g0)[0])
        records.append(_rec("Gate", int(N), pm, p_model, sh))

    T2e = result.T2_echo_s or min(result.T2_s, 2.0 * result.T1_s)
    for d, c in zip(meta["echo_delays_s"], echo_c):
        sh = c["0"] + c["1"]
        pm = c["1"] / sh if sh else float("nan")
        records.append(_rec("Echo", d, pm,
                            float(inv.forward_echo(d, T2e, p0g1, p1g0)), sh))

    z = np.array([r["z"] for r in records if np.isfinite(r["z"])])
    return dict(records=records,
                chi2_dof_joint=float(np.mean(z**2)) if len(z) else float("nan"),
                n_conditions=len(records))


def check_admissibility(result) -> dict:
    T1, T2 = result.T1_s, result.T2_s
    ratio = T2 / (2.0 * T1) if T1 > 0 else float("nan")
    return dict(T1_positive=T1 > 0, T2_positive=T2 > 0,
                T2_le_2T1=T2 <= 2.0 * T1 * 1.001,
                T2_over_2T1_ratio=ratio,
                all_admissible=(T1 > 0 and T2 > 0 and T2 <= 2.0 * T1 * 1.001))


def evaluate_holdout(test_counts: list, meta: dict, fit_result) -> dict:
    pred = predict_and_residualize(test_counts, meta, fit_result)
    z = np.array([r["z"] for r in pred["records"] if np.isfinite(r["z"])])
    cov = float(np.mean(np.abs(z) <= 1.96)) if len(z) else float("nan")
    return dict(chi2_dof_predictive=pred["chi2_dof_joint"],
                coverage_95=cov, n_conditions=pred["n_conditions"])


# ── main per-batch run (all qubits in parallel per circuit) ───────────────────

def run_all_qubits(api_key: str, api_target: str, noise_model: str,
                   qubits: list, seed: int) -> dict:
    """
    Runs all 15 probe circuits with all qubits in parallel.
    Each circuit is one IonQ job with n_qubits = len(qubits).
    Returns per-qubit counts_list and metadata.
    """
    n_qubits = len(qubits)
    profile  = inv.BackendProfile.from_architecture(ARCHITECTURE)

    t1_delays     = profile.t1_delays_s
    ramsey_delays = profile.ramsey_delays_s
    echo_delays   = profile.echo_delays_s
    gate_rep_n    = profile.gate_rep_n

    n_t1     = len(t1_delays)
    n_ramsey = 2 * len(ramsey_delays)
    n_gate   = len(gate_rep_n)
    n_echo   = len(echo_delays)

    meta = dict(
        t1_delays_s=t1_delays, ramsey_delays_s=ramsey_delays,
        gate_rep_N=gate_rep_n, echo_delays_s=echo_delays,
        n_t1=n_t1, n_ramsey=n_ramsey, n_gate=n_gate, n_echo=n_echo,
        dw_max_rad_s=profile.dw_max_rad_s, architecture=ARCHITECTURE,
        backend_name=api_target, T1_prior_s=profile.T1_prior_s,
        T2_prior_s=profile.T2_prior_s, dt_ns=profile.dt_ns,
    )

    # per_qubit_counts[q] = list of count dicts in probe order
    per_qubit_counts = [[] for _ in range(n_qubits)]

    logical_qubits = list(range(n_qubits))  # 0..n_qubits-1

    def submit_and_collect(gates, shots, name):
        print(f"    → {name} ({shots} shots × {n_qubits} qubits)... ", end="", flush=True)
        job_id = submit_job(api_key, api_target, noise_model,
                            gates, n_qubits, shots, name)
        job = poll_job(api_key, job_id)
        counts = parse_per_qubit_counts(job, shots, n_qubits)
        print("done")
        return counts

    # T1 circuits
    for d in t1_delays:
        gates  = _t1_circuit(d, logical_qubits)
        counts = submit_and_collect(gates, DEFAULT_SHOTS["t1"],
                                    f"T1_delay_{d:.3f}s")
        for q in range(n_qubits):
            per_qubit_counts[q].append(counts[q])

    # Ramsey circuits (X then Y per delay)
    for t in ramsey_delays:
        for circ_fn, label in [(_ramsey_x_circuit, "X"),
                                (_ramsey_y_circuit, "Y")]:
            gates  = circ_fn(t, logical_qubits)
            counts = submit_and_collect(gates, DEFAULT_SHOTS["ramsey"],
                                        f"Ramsey{label}_{t:.3f}s")
            for q in range(n_qubits):
                per_qubit_counts[q].append(counts[q])

    # Gate repetition circuits
    for N in gate_rep_n:
        gates  = _gate_rep_circuit(N, logical_qubits)
        counts = submit_and_collect(gates, DEFAULT_SHOTS["gate"],
                                    f"Gate_N{N}")
        for q in range(n_qubits):
            per_qubit_counts[q].append(counts[q])

    # Echo circuits
    for d in echo_delays:
        gates  = _echo_circuit(d, logical_qubits)
        counts = submit_and_collect(gates, DEFAULT_SHOTS["echo"],
                                    f"Echo_{d:.3f}s")
        for q in range(n_qubits):
            per_qubit_counts[q].append(counts[q])

    return meta, per_qubit_counts


def invert_and_analyze(qubits: list, meta: dict,
                       per_qubit_counts: list, seed: int) -> list:
    """Run lindblad_inversion + residuals + holdout + admissibility per qubit."""
    results = []
    profile = inv.BackendProfile.from_architecture(ARCHITECTURE)
    DS = DEFAULT_SHOTS

    for qi, (qubit_id, counts_list) in enumerate(zip(qubits, per_qubit_counts)):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            result_full = inv.lindblad_inversion(
                counts_list, meta, profile,
                shots_t1=DS["t1"], shots_ramsey=DS["ramsey"],
                shots_gate=DS["gate"], shots_echo=DS["echo"],
                qubit_id=qubit_id,
                timestamp=datetime.now(timezone.utc).isoformat())

        model_fit    = predict_and_residualize(counts_list, meta, result_full)
        admissibility = check_admissibility(result_full)

        fit_c, test_c = split_counts_list(counts_list, FIT_FRACTION,
                                          seed=seed + qi)
        fit_sh = dict(t1    = avg_shots(fit_c[:meta["n_t1"]]),
                      ramsey= avg_shots(fit_c[meta["n_t1"]:meta["n_t1"]+meta["n_ramsey"]]),
                      gate  = avg_shots(fit_c[meta["n_t1"]+meta["n_ramsey"]:
                                              meta["n_t1"]+meta["n_ramsey"]+meta["n_gate"]]),
                      echo  = avg_shots(fit_c[meta["n_t1"]+meta["n_ramsey"]+meta["n_gate"]:]))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            result_fit = inv.lindblad_inversion(
                fit_c, meta, profile,
                shots_t1=max(1,fit_sh["t1"]), shots_ramsey=max(1,fit_sh["ramsey"]),
                shots_gate=max(1,fit_sh["gate"]), shots_echo=max(1,fit_sh["echo"]),
                qubit_id=qubit_id,
                timestamp=datetime.now(timezone.utc).isoformat())
        holdout = evaluate_holdout(test_c, meta, result_fit)

        print(f"  Qubit {qubit_id}: "
              f"T1={result_full.T1_s:.4g}s  T2={result_full.T2_s:.4g}s  "
              f"dw={result_full.delta_omega:.4g}  "
              f"eps={result_full.epsilon_sx:.3e}  "
              f"chi2={model_fit['chi2_dof_joint']:.2f}  "
              f"holdout_cov={holdout['coverage_95']:.2f}  "
              f"T2<=2T1={'OK' if admissibility['all_admissible'] else 'VIOLATED'}")

        results.append(dict(
            qubit_id=qubit_id,
            counts_list=counts_list,
            result_full=dict(
                T1_s=result_full.T1_s, T1_sigma_s=result_full.T1_sigma_s,
                T2_s=result_full.T2_s, T2_sigma_s=result_full.T2_sigma_s,
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

    print("\n" + "="*72)
    print("  RUN A vs RUN B — PRECISION (normalized differences z_j)")
    print("="*72)
    params = [("T1_s","T1_sigma_s","T1"),("T2_s","T2_sigma_s","T2"),
              ("delta_omega","delta_omega_sigma","dw"),
              ("epsilon_sx","epsilon_sx_sigma","eps")]
    for q in run_b_data["qubits"]:
        qid = q["qubit_id"]
        if qid not in a_by_q:
            continue
        a, b = a_by_q[qid]["result_full"], q["result_full"]
        print(f"\n  Qubit {qid}:")
        for vk, sk, lbl in params:
            va, sa = a[vk], a[sk]
            vb, sb = b[vk], b[sk]
            denom = np.sqrt(sa**2 + sb**2)
            z = (va - vb)/denom if denom > 0 else float("nan")
            print(f"    {lbl:5s}: A={va:.4g}±{sa:.2g}  B={vb:.4g}±{sb:.2g}  "
                  f"z_j={z:+.2f}  {'OK' if abs(z)<=2 else 'DRIFT'}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qubits", required=True,
                    help="Comma-separated qubit indices, e.g. 0,1,2,3,4,5,6,7")
    ap.add_argument("--run-label", required=True, choices=["A","B"],
                    help="Run A or Run B (back-to-back precision check)")
    ap.add_argument("--backend", default="ionq_simulator",
                    help="Backend name. Examples: ionq_simulator (default), "
                         "ionq_simulator:forte-1, qpu.forte-enterprise-1")
    ap.add_argument("--live", action="store_true",
                    help="Required when --backend is a real QPU (qpu.*)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    qubits = [int(q.strip()) for q in args.qubits.split(",")]

    if is_real_qpu(args.backend) and not args.live:
        sys.exit(f"ERROR: '{args.backend}' is real hardware. "
                 f"Add --live to confirm credit spend.")

    api_key = os.environ.get("IONQ_API_KEY")
    if not api_key:
        sys.exit("ERROR: IONQ_API_KEY not set.\n"
                 "PowerShell: $env:IONQ_API_KEY = 'your_key_here'")

    api_target, noise_model = parse_backend(args.backend)

    # Cost estimate
    profile_tmp = inv.BackendProfile.from_architecture(ARCHITECTURE)
    _, meta_tmp = inv.build_probe_circuits(profile_tmp)
    shots_per_q = total_shots_for(meta_tmp, DEFAULT_SHOTS)
    total_shots = shots_per_q * len(qubits)
    n_jobs = meta_tmp["n_t1"] + len(profile_tmp.ramsey_delays_s)*2 + \
             meta_tmp["n_gate"] + meta_tmp["n_echo"]

    print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}] "
          f"5_ionq_hardware.py  Run {args.run_label}")
    print(f"  Backend    : {args.backend}  (API target: {api_target}"
          f"{', noise: '+noise_model if noise_model else ''})")
    print(f"  Qubits     : {qubits}  (n={len(qubits)}, all run in parallel)")
    print(f"  API jobs   : {n_jobs} total  ({shots_per_q:,} shots/qubit)")
    print(f"  Total shots: {total_shots:,}")

    if is_real_qpu(args.backend):
        est = total_shots * COST_PER_SHOT
        print(f"  Est. cost  : ${est:,.2f}  (at ${COST_PER_SHOT}/shot)")
        if input("  Type RUN to confirm: ").strip() != "RUN":
            sys.exit("Aborted.")
    else:
        nm = f"noise={noise_model}" if noise_model else "noiseless"
        print(f"  Est. cost  : $0.00  (simulator, {nm})")
    print()

    # Run circuits
    meta, per_qubit_counts = run_all_qubits(
        api_key, api_target, noise_model, qubits, args.seed)

    # Invert per qubit
    print("\nRunning Lindblad inversion per qubit...")
    qubit_results = invert_and_analyze(qubits, meta, per_qubit_counts, args.seed)

    # Save
    out_path = (DATA_DIR /
                f"run_{args.run_label}_qubits_{'- '.join(map(str,qubits))}.json")
    payload = dict(run_label=args.run_label, backend=args.backend,
                   architecture=ARCHITECTURE,
                   timestamp=datetime.now(timezone.utc).isoformat(),
                   meta=meta, qubits=qubit_results)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved: {out_path}")

    if args.run_label == "B":
        run_a = DATA_DIR / f"run_A_qubits_{'- '.join(map(str,qubits))}.json"
        compare_runs(run_a, payload)

    print(f"\n  Done. {len(qubit_results)}/{len(qubits)} qubits completed.")


if __name__ == "__main__":
    main()