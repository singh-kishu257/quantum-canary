import argparse
import csv
import importlib.util
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

import certifi
import urllib3
urllib3.disable_warnings()
os.environ["SSL_CERT_FILE"]      = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import requests as _req
_orig = _req.get
def _g(url, **kw):
    if "ionq.co" in str(url):
        kw["verify"] = False
    return _orig(url, **kw)
_req.get = _g

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("inversion", SCRIPT_DIR / "1_inversion.py")
inv = importlib.util.module_from_spec(_spec)
sys.modules["inversion"] = inv
_spec.loader.exec_module(inv)

DATA_DIR = SCRIPT_DIR / "data" / "hardware"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ARCH          = inv.ARCH_DEFAULTS["trapped_ion"]
SEED          = 47
CANARY_SHOTS  = 500
COST_PER_QGS  = 2.56e-4


def get_backend(backend_arg, api_key):
    from qiskit_ionq import IonQProvider
    p = IonQProvider(api_key)
    if backend_arg.startswith("ionq_simulator:"):
        b = p.get_backend("ionq_simulator", gateset="native")
        b.set_options(noise_model=backend_arg.split(":", 1)[1])
        return b
    return p.get_backend(backend_arg, gateset="native")


def is_qpu(backend_arg):
    return backend_arg.startswith("qpu.")


def backend_name(backend_arg):
    if backend_arg.startswith("ionq_simulator:"):
        return backend_arg.split(":", 1)[1]
    if backend_arg.startswith("qpu."):
        return backend_arg[4:]
    return "forte-1"


def backend_max_qubits(backend):
    try:
        return int(backend.num_qubits)
    except Exception:
        pass
    try:
        return int(backend.configuration().n_qubits)
    except Exception:
        pass
    try:
        return int(backend._num_qubits)
    except Exception:
        pass
    try:
        return backend.target.num_qubits
    except Exception:
        pass
    return None


def resolve_qubits(qubits_arg, backend):
    if qubits_arg.strip().lower() == "all":
        max_q = backend_max_qubits(backend)
        if max_q is None:
            sys.exit("Could not determine qubit count for 'all'. Pass --qubits explicitly.")
        return list(range(max_q))
    return [int(q.strip()) for q in qubits_arg.split(",")]


def fetch_published_device_eps(api_key, bname):
    try:
        url = f"https://api.ionq.co/v0.3/characterizations/backends/qpu.{bname}"
        r = _req.get(url, headers={"Authorization": f"apiKey {api_key}"},
                     params={"limit": 1}, timeout=15, verify=False)
        r.raise_for_status()
        data = r.json()
        chars = data.get("characterizations", [])
        if not chars:
            return float("nan"), float("nan"), None
        c = chars[0]
        f1q = c.get("fidelity", {}).get("1q", {})
        mean = f1q.get("mean")
        stderr = f1q.get("stderr")
        if mean is None:
            return float("nan"), float("nan"), c.get("date")
        eps = 1.0 - float(mean)
        eps_sigma = float(stderr) if stderr is not None else float("nan")
        return eps, eps_sigma, c.get("date")
    except Exception as e:
        print(f"  WARNING: could not fetch published device characterization ({e})")
        return float("nan"), float("nan"), None


def load_profiles(qubits, api_key, backend_arg):
    bname = backend_name(backend_arg)
    p0g1_live, p1g0_live, spam_src, _ = inv.fetch_live_spam(
        "trapped_ion", ionq_api_token=api_key, ionq_backend_name=bname)
    print(f"  SPAM source : {spam_src}  "
          f"(p0|1={p0g1_live:.5f}  p1|0={p1g0_live:.5f})")
    profiles = {}
    for q in qubits:
        p = inv.get_live_profile(f"ionq:{bname}", qubit_id=q, ionq_token=api_key)
        if p.custom_arch is None:
            p.custom_arch = dict(inv.ARCH_DEFAULTS["trapped_ion"])
        p.custom_arch["p0_given_1"] = p0g1_live
        p.custom_arch["p1_given_0"] = p1g0_live
        profiles[q] = p
    return profiles


def _gpi2_phi_gate(qc, q, phi):
    try:
        from qiskit_ionq.ionq_gates import GPI2Gate
        qc.append(GPI2Gate(float(phi)), [q])
    except ImportError:
        qc.r(np.pi / 2, 2 * np.pi * phi, q)


def canary_circuit(N, target_qubits):
    from qiskit import QuantumCircuit
    assert N % 2 == 0, f"N must be even, got N={N}"
    full_width = max(target_qubits) + 1
    qc = QuantumCircuit(full_width, len(target_qubits), name=f"canary_N{N}")
    for _ in range(N):
        for q in target_qubits:
            inv._sqrtx_native_inverse_pair(qc, q, "trapped_ion")
        qc.barrier(*target_qubits)
    for i, q in enumerate(target_qubits):
        qc.measure(q, i)
    return qc


def submit_one(backend, qc, shots):
    from qiskit import transpile
    tqc = transpile(qc, backend=backend, optimization_level=0)
    try:
        job = backend.run(tqc, shots=shots, optimization_level=0,
                          error_mitigation={"debias": False, "sharpen": False})
    except TypeError:
        job = backend.run(tqc, shots=shots, optimization_level=0)
    return job.result().get_counts()


def counts_to_per_qubit(counts, target_qubits, shots):
    per_q = {q: {"0": 0, "1": 0} for q in target_qubits}
    n     = len(target_qubits)
    for bs, cnt in counts.items():
        bs = bs.replace(" ", "").zfill(n)
        for i, q in enumerate(target_qubits):
            b = bs[-(i + 1)]
            if b in ("0", "1"):
                per_q[q][b] += cnt
    for q in target_qubits:
        t = per_q[q]["0"] + per_q[q]["1"]
        if t < shots:
            per_q[q]["0"] += shots - t
    return per_q


def p0_from_counts(per_q, q_id, shots):
    t = per_q[q_id]["0"] + per_q[q_id]["1"]
    return per_q[q_id]["0"] / t if t > 0 else 0.5


def save_chk(path, data):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def load_chk(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def fit_canary(p0_meas, Nv, arch, shots):
    return inv._invert_gate(
        np.array(p0_meas, dtype=float),
        np.array(Nv, dtype=float),
        arch, shots)


def run_canary(backend, qubits, profiles, chk, chk_path, shots):
    Nv_raw = profiles[qubits[0]].gate_rep_n
    Nv     = [N if N % 2 == 0 else N + 1 for N in Nv_raw]
    print(f"  Canary: N={Nv}  ({[2*N for N in Nv]} native gate pairs/qubit)  "
          f"{shots} shots  single-qubit circuits, one qubit at a time")
    print(f"  Loop: for each qubit → run all {len(Nv)} N values → fit ε_sx → next qubit")

    by_qubit = {int(k): v for k, v in
                chk.get("canary_results_by_qubit", {}).items()}

    results = {}

    for q in qubits:
        q_data = by_qubit.get(q, {})

        # Already fully done — load from checkpoint
        if "eps_sx" in q_data:
            eps = q_data["eps_sx"]
            sig = q_data.get("eps_sx_sigma", float("nan"))
            cal = profiles[q].constants
            results[q] = dict(
                Nv=Nv,
                p0_measured=[q_data.get(str(N), float("nan")) for N in Nv],
                eps_sx=eps, eps_sx_sigma=sig,
                p0g1=cal["p0_given_1"], p1g0=cal["p1_given_0"])
            print(f"  Q{q:2d}: checkpoint  eps_sx={eps:.4e} ± {sig:.2e}")
            continue

        print(f"  Q{q:2d}: submitting N={Nv} ...")
        stopped = False

        for N in Nv:
            if str(N) in q_data:
                print(f"    N={N:5d}: checkpoint  p0={q_data[str(N)]:.4f}")
                continue
            try:
                qc     = canary_circuit(N, [q])
                counts = submit_one(backend, qc, shots)
                per_q  = counts_to_per_qubit(counts, [q], shots)
                p0     = p0_from_counts(per_q, q, shots)
                q_data[str(N)] = p0
                by_qubit[q] = q_data
                chk["canary_results_by_qubit"] = {str(k): v
                                                   for k, v in by_qubit.items()}
                save_chk(chk_path, chk)
                print(f"    N={N:5d}: p0={p0:.4f}")
            except Exception as e:
                chk["canary_results_by_qubit"] = {str(k): v
                                                   for k, v in by_qubit.items()}
                save_chk(chk_path, chk)
                print(f"    N={N:5d}: FAILED ({e}) — checkpoint saved, stopping")
                stopped = True
                break

        if stopped:
            break

        # All N values done for this qubit — fit ε_sx now
        p0_meas = [q_data.get(str(N), float("nan")) for N in Nv]
        Nv_done = [N for N, p in zip(Nv, p0_meas) if np.isfinite(p)]
        p0_done = [p for p in p0_meas if np.isfinite(p)]

        cal  = profiles[q].constants
        arch = dict(ARCH)
        arch["p0_given_1"]  = cal["p0_given_1"]
        arch["p1_given_0"]  = cal["p1_given_0"]
        arch["eps_typical"] = cal["eps_typical"]

        if len(Nv_done) >= 2:
            eps, sig, _ = fit_canary(p0_done, Nv_done, arch, shots)
        else:
            eps, sig = float("nan"), float("nan")

        q_data["eps_sx"]       = eps
        q_data["eps_sx_sigma"] = sig
        by_qubit[q] = q_data
        chk["canary_results_by_qubit"] = {str(k): v for k, v in by_qubit.items()}
        save_chk(chk_path, chk)

        results[q] = dict(
            Nv=Nv_done, p0_measured=p0_done,
            eps_sx=eps, eps_sx_sigma=sig,
            p0g1=cal["p0_given_1"], p1g0=cal["p1_given_0"])
        print(f"  Q{q:2d}: eps_sx={eps:.4e} ± {sig:.2e}  "
              f"p0={[f'{p:.3f}' for p in p0_done]}")

    return results, Nv


def print_comparison(canary, qubits, published_eps, published_eps_sigma):
    vals = np.array([canary[q]["eps_sx"] for q in qubits], dtype=float)
    vals = vals[np.isfinite(vals)]
    print(f"\n{'='*72}")
    print(f"  Canary per-qubit ε_sx  vs  IonQ published device-average 1Q infidelity")
    print(f"  IonQ's calibration API reports one device-wide value (GST-based RB),")
    print(f"  not per-qubit values. Canary provides per-qubit resolution.")
    print(f"{'='*72}")
    print(f"  {'Q':>4}  {'Canary eps_sx':>16}  {'sigma':>12}")
    print(f"  {'-'*38}")
    for q in qubits:
        e = canary[q]["eps_sx"]
        s = canary[q]["eps_sx_sigma"]
        if np.isfinite(e):
            print(f"  {q:>4}  {e:>16.4e}  {s:>12.2e}")
        else:
            print(f"  {q:>4}  {'insufficient data':>16}")
    if len(vals) > 0:
        print(f"\n  N valid qubits       : {len(vals)}/{len(qubits)}")
        print(f"  Canary mean eps_sx   : {np.mean(vals):.4e}")
        print(f"  Canary std eps_sx    : {np.std(vals):.4e}")
        print(f"  Canary min / max     : {np.min(vals):.4e} / {np.max(vals):.4e}")
    if np.isfinite(published_eps):
        print(f"  IonQ published eps   : {published_eps:.4e} +/- {published_eps_sigma:.2e}")
        if len(vals) > 0:
            ratio = np.mean(vals) / published_eps
            print(f"  Ratio (Canary/IonQ)  : {ratio:.3f}")
    else:
        print(f"  IonQ published eps   : unavailable")
    print(f"{'='*72}")


def estimate_cost(qubits, Nv, shots_c):
    # Single-qubit circuits: cost = same total gate-shots as parallel
    # IonQ bills by gate-qubit-shot regardless of circuit structure
    gates_c = sum(2 * N for N in Nv) * len(qubits) * shots_c
    return gates_c * COST_PER_QGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits",        default="all",
                    help="'all' (default, sweeps every qubit on the backend), "
                         "or comma list e.g. 0,5,9,14")
    ap.add_argument("--backend",       default="ionq_simulator:forte-1")
    ap.add_argument("--live",          action="store_true")
    ap.add_argument("--canary-shots",  type=int, default=CANARY_SHOTS)
    ap.add_argument("--resume",        default=None)
    args = ap.parse_args()

    if is_qpu(args.backend) and not args.live:
        sys.exit(f"'{args.backend}' is a real QPU. Add --live to confirm charges.")

    api_key = os.environ.get("IONQ_API_KEY")
    if not api_key:
        sys.exit("IONQ_API_KEY not set.\n  PowerShell: $env:IONQ_API_KEY='key'")

    ts       = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    chk_path = (pathlib.Path(args.resume) if args.resume
                else DATA_DIR / f"chk_ionq_{ts}.json")
    chk      = load_chk(chk_path)

    print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}]  6_ionq_eps_rb.py")
    print(f"  Backend       : {args.backend}")
    print(f"  Canary shots  : {args.canary_shots}")
    print(f"  Method        : Canary gate repetition, one qubit at a time (single-qubit circuits)")
    print(f"  No error mitigation, no optimization, native gateset")

    backend = get_backend(args.backend, api_key)
    qubits  = resolve_qubits(args.qubits, backend)
    print(f"  Qubits        : {len(qubits)}  ({qubits[0]}-{qubits[-1]})")

    print("\nLoading per-qubit live profiles...")
    all_profiles = load_profiles(qubits, api_key, args.backend)

    bname = backend_name(args.backend)
    published_eps, published_eps_sigma, cal_date = fetch_published_device_eps(api_key, bname)
    if np.isfinite(published_eps):
        print(f"  Published 1Q device eps : {published_eps:.4e} +/- {published_eps_sigma:.2e} "
              f"(characterization date epoch={cal_date})")
    else:
        print(f"  Published 1Q device eps : unavailable for backend={bname}")

    if is_qpu(args.backend):
        Nv_est = all_profiles[qubits[0]].gate_rep_n
        Nv_est = [N if N % 2 == 0 else N + 1 for N in Nv_est]
        cost   = estimate_cost(qubits, Nv_est, args.canary_shots)
        print(f"\n  Est. total cost : ${cost:,.0f}")
        print(f"  Budget impact   : ${cost:,.0f} of $30,000")
        if input("  Type RUN to confirm: ").strip() != "RUN":
            sys.exit("Aborted.")

    print(f"\n--- Canary gate repetition (all qubits) ---")
    canary_results, Nv = run_canary(backend, qubits, all_profiles,
                                    chk, chk_path, args.canary_shots)

    print_comparison(canary_results, qubits, published_eps, published_eps_sigma)

    cal_summary = {q: dict(eps_prior=all_profiles[q].constants["eps_typical"],
                           p0g1=all_profiles[q].constants["p0_given_1"],
                           p1g0=all_profiles[q].constants["p1_given_0"])
                   for q in qubits}

    stem = f"ionq_canary_allq_{qubits[0]}-{qubits[-1]}_{ts}"

    try:
        out_json = DATA_DIR / f"{stem}.json"
        with open(out_json, "w") as f:
            json.dump(dict(backend=args.backend,
                           timestamp=datetime.now(timezone.utc).isoformat(),
                           qubits=qubits,
                           gate_rep_N=Nv,
                           published_device_eps_1q=published_eps,
                           published_device_eps_1q_sigma=published_eps_sigma,
                           published_characterization_date=cal_date,
                           calibration=cal_summary,
                           canary={str(k): v for k, v in canary_results.items()}),
                      f, indent=2)
        print(f"\n  Saved JSON : {out_json}")
    except Exception as e:
        print(f"\n  JSON save FAILED: {e}")

    try:
        out_csv = DATA_DIR / f"{stem}.csv"
        rows    = []
        for q in qubits:
            c = canary_results.get(q, {})
            rows.append(dict(
                qubit                  = q,
                backend                = args.backend,
                timestamp              = datetime.now(timezone.utc).isoformat(),
                canary_eps_sx          = c.get("eps_sx",       float("nan")),
                canary_sigma           = c.get("eps_sx_sigma", float("nan")),
                published_device_eps_1q       = published_eps,
                published_device_eps_1q_sigma = published_eps_sigma,
                p0g1                   = cal_summary[q]["p0g1"],
                p1g0                   = cal_summary[q]["p1g0"],
            ))
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"  Saved CSV  : {out_csv}")
    except Exception as e:
        print(f"  CSV save FAILED: {e}")

    if chk_path.exists():
        chk_path.unlink()
        print(f"  Checkpoint removed: {chk_path}")


if __name__ == "__main__":
    main()