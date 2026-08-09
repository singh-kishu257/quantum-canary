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
from scipy.optimize import curve_fit

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
XEB_SHOTS     = 500
XEB_CIRCUITS  = 3
XEB_DEPTHS    = [1, 500, 2000]
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


def _gpi2_gate(qc, q):
    try:
        from qiskit_ionq.ionq_gates import GPI2Gate
        qc.append(GPI2Gate(0), [q])
    except ImportError:
        qc.sx(q)


def _gpi2_phi_gate(qc, q, phi):
    try:
        from qiskit_ionq.ionq_gates import GPI2Gate
        qc.append(GPI2Gate(float(phi)), [q])
    except ImportError:
        qc.r(np.pi / 2, 2 * np.pi * phi, q)


def canary_circuit(N, target_qubits):
    from qiskit import QuantumCircuit
    assert N % 2 == 0, f"N must be even for GPI2 gate rep (GPI2^4=I), got N={N}"
    full_width = max(target_qubits) + 1
    qc         = QuantumCircuit(full_width, len(target_qubits), name=f"canary_N{N}")
    for _ in range(2 * N):
        for q in target_qubits:
            _gpi2_gate(qc, q)
        qc.barrier(*target_qubits)
    for i, q in enumerate(target_qubits):
        qc.measure(q, i)
    return qc


def xeb_circuit(depth, circ_seed, target_qubits, full_width):
    from qiskit import QuantumCircuit
    rng = np.random.default_rng([SEED, circ_seed])
    phases = rng.uniform(0, 1, size=(depth, len(target_qubits)))
    qc = QuantumCircuit(full_width, len(target_qubits),
                        name=f"xeb_d{depth}_c{circ_seed}")
    for step in range(depth):
        for idx, q in enumerate(target_qubits):
            _gpi2_phi_gate(qc, q, phases[step, idx])
        qc.barrier(*target_qubits)
    for i, q in enumerate(target_qubits):
        qc.measure(q, i)
    return qc, phases


def ideal_p0_xeb(phases, target_qubit_idx):
    n_steps = phases.shape[0]
    state = np.array([1.0, 0.0], dtype=complex)
    for step in range(n_steps):
        phi = phases[step, target_qubit_idx]
        angle = np.pi / 2
        ax   = np.cos(2 * np.pi * phi)
        ay   = np.sin(2 * np.pi * phi)
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        H  = (angle / 2) * (ax * sx + ay * sy)
        U  = np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * (ax * sx + ay * sy)
        state = U @ state
    return float(abs(state[0]) ** 2)


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


def fit_xeb_curve(depths, fidelities):
    def model(m, eps):
        return (1.0 - float(eps)) ** np.asarray(m, float)
    try:
        popt, pcov = curve_fit(
            model, depths, fidelities,
            p0=[1e-4],
            bounds=([0], [0.5]),
            maxfev=20000)
        return float(popt[0]), float(np.sqrt(pcov[0, 0]))
    except Exception:
        return float("nan"), float("nan")


def run_canary(backend, qubits, profiles, chk, chk_path, shots):
    n_q        = len(qubits)
    full_width = max(qubits) + 1
    Nv_raw     = profiles[qubits[0]].gate_rep_n
    Nv         = [N if N % 2 == 0 else N + 1 for N in Nv_raw]
    print(f"  Canary: N={Nv}  ({[2*N for N in Nv]} GPI2 gates/qubit)  "
          f"{shots} shots  {n_q}q  circuit_width={full_width}")

    p0_by_N = chk.get("canary_p0_by_N", {})

    for N in Nv:
        if str(N) in p0_by_N:
            print(f"    N={N:4d}: loaded from checkpoint")
            continue
        try:
            qc     = canary_circuit(N, qubits)
            counts = submit_one(backend, qc, shots)
            per_q  = counts_to_per_qubit(counts, qubits, shots)
            p0_by_N[str(N)] = {q_id: p0_from_counts(per_q, q_id, shots)
                               for q_id in qubits}
            chk["canary_p0_by_N"] = p0_by_N
            save_chk(chk_path, chk)
            print(f"    N={N:4d}: p0={[f'{p0_by_N[str(N)][q]:.3f}' for q in qubits]}")
        except Exception as e:
            chk["canary_p0_by_N"] = p0_by_N
            save_chk(chk_path, chk)
            print(f"    N={N:4d}: FAILED ({e}) — checkpoint saved, stopping Canary")
            break

    results = {}
    for q_id in qubits:
        cal  = profiles[q_id].constants
        arch = dict(ARCH)
        arch["p0_given_1"]  = cal["p0_given_1"]
        arch["p1_given_0"]  = cal["p1_given_0"]
        arch["eps_typical"] = cal["eps_typical"]
        Nv_done = [N for N in Nv if str(N) in p0_by_N]
        p0_meas = [p0_by_N[str(N)][q_id] for N in Nv_done]
        if len(Nv_done) < 2:
            results[q_id] = dict(Nv=Nv_done, p0_measured=p0_meas,
                                 eps_sx=float("nan"), eps_sx_sigma=float("nan"),
                                 p0g1=cal["p0_given_1"], p1g0=cal["p1_given_0"])
            continue
        eps, sig, _ = fit_canary(p0_meas, Nv_done, arch, shots)
        print(f"    Q{q_id}: ε_sx = {eps:.4e} ± {sig:.2e}")
        results[q_id] = dict(Nv=Nv_done, p0_measured=p0_meas,
                             eps_sx=eps, eps_sx_sigma=sig,
                             p0g1=cal["p0_given_1"], p1g0=cal["p1_given_0"])
    return results, Nv


def run_xeb(backend, qubits, chk, chk_path, shots, n_circuits, depths):
    n_q        = len(qubits)
    full_width = max(qubits) + 1

    xeb_data  = chk.get("xeb_data",
                        {str(q): {str(d): {"p0_meas": [], "p0_ideal": []}
                                  for d in depths} for q in qubits})
    done_circs = chk.get("xeb_done_circs", 0)

    print(f"  XEB: depths={depths}  {n_circuits} circuits  {shots} shots  "
          f"{n_q}q  circuit_width={full_width}  (same ions as Canary)")
    print(f"  Gate: GPI2(random_phi) — same gate as Canary, different phases")
    print(f"  F_XEB(m) = <P_ideal>_meas / <P_ideal^2>_ideal ≈ (1-ε)^m")
    if done_circs > 0:
        print(f"  Resuming from circuit {done_circs + 1}/{n_circuits}")

    for ci in range(done_circs, n_circuits):
        failed = False
        for depth in depths:
            qc, phases = xeb_circuit(depth, ci, qubits, full_width)
            try:
                counts = submit_one(backend, qc, shots)
                per_q  = counts_to_per_qubit(counts, qubits, shots)
                for idx, q_id in enumerate(qubits):
                    p0_m   = p0_from_counts(per_q, q_id, shots)
                    p0_i   = ideal_p0_xeb(phases, idx)
                    xeb_data[str(q_id)][str(depth)]["p0_meas"].append(p0_m)
                    xeb_data[str(q_id)][str(depth)]["p0_ideal"].append(p0_i)
            except Exception as e:
                print(f"    circuit {ci+1} depth {depth} FAILED: {e}")
                failed = True
                break
        chk["xeb_data"]       = xeb_data
        chk["xeb_done_circs"] = ci + (0 if failed else 1)
        save_chk(chk_path, chk)
        if failed:
            break
        print(f"    circuit {ci+1}/{n_circuits} done")

    results = {}
    for q_id in qubits:
        depths_ok = [d for d in depths
                     if len(xeb_data[str(q_id)][str(d)]["p0_meas"]) > 0]
        if len(depths_ok) < 2:
            print(f"    Q{q_id}: insufficient XEB data")
            results[q_id] = dict(depths=depths_ok, fidelities=[],
                                 eps_sx=float("nan"), eps_sx_sigma=float("nan"))
            continue

        fidelities = []
        for d in depths_ok:
            p0_meas  = np.array(xeb_data[str(q_id)][str(d)]["p0_meas"])
            p0_ideal = np.array(xeb_data[str(q_id)][str(d)]["p0_ideal"])
            p0_ideal_sq_mean = float(np.mean(p0_ideal ** 2))
            p0_ideal_mean    = float(np.mean(p0_ideal))
            denom = p0_ideal_sq_mean - 0.5
            if abs(denom) < 1e-8:
                fidelities.append(float("nan"))
                continue
            f_xeb = (float(np.mean(p0_meas)) - 0.5) / denom
            fidelities.append(float(np.clip(f_xeb, 0, 1)))

        valid = [(d, f) for d, f in zip(depths_ok, fidelities) if np.isfinite(f)]
        if len(valid) < 2:
            print(f"    Q{q_id}: insufficient valid XEB fidelities")
            results[q_id] = dict(depths=depths_ok, fidelities=fidelities,
                                 eps_sx=float("nan"), eps_sx_sigma=float("nan"))
            continue

        d_valid = [v[0] for v in valid]
        f_valid = [v[1] for v in valid]
        eps, eps_sig = fit_xeb_curve(d_valid, f_valid)
        print(f"    Q{q_id}: F_XEB={[f'{f:.4f}' for f in f_valid]}  "
              f"ε_sx={eps:.4e} ± {eps_sig:.2e}")
        results[q_id] = dict(depths=depths_ok, fidelities=fidelities,
                             eps_sx=eps, eps_sx_sigma=eps_sig)
    return results


def print_comparison(canary, xeb, qubits):
    common = [q for q in qubits if q in xeb
              and np.isfinite(canary[q]["eps_sx"])
              and np.isfinite(xeb[q]["eps_sx"])
              and xeb[q]["eps_sx"] > 0]
    print(f"\n{'='*72}")
    print(f"  Canary ε_sx  vs  XEB ε_sx  —  per qubit")
    print(f"  Both measure error per GPI2(φ) gate application.")
    print(f"  Canary: deterministic φ=0, depolarizing model fit.")
    print(f"  XEB:    random φ, cross-entropy fidelity fit. (Arute et al. 2019)")
    print(f"{'='*72}")
    print(f"  {'Q':>3}  {'Canary':>14}  {'XEB':>14}  {'ratio':>8}  {'z':>7}  {'flag':>6}")
    print(f"  {'-'*68}")
    for q in qubits:
        c_e = canary[q]["eps_sx"]
        c_s = canary[q]["eps_sx_sigma"]
        if q not in xeb or not np.isfinite(xeb[q]["eps_sx"]):
            print(f"  {q:>3}  {c_e:>14.4e}  {'insufficient data':>14}")
            continue
        x_e = xeb[q]["eps_sx"]
        x_s = xeb[q]["eps_sx_sigma"]
        if not (np.isfinite(c_e) and np.isfinite(x_e) and x_e > 0):
            print(f"  {q:>3}  {c_e:>14.4e}  {x_e:>14.4e}  {'NaN':>8}")
            continue
        ratio = c_e / x_e
        denom = np.sqrt(c_s**2 + x_s**2)
        z     = (c_e - x_e) / denom if denom > 0 else float("nan")
        flag  = "OK" if np.isfinite(z) and abs(z) <= 2 else "CHECK"
        print(f"  {q:>3}  {c_e:>14.4e}  {x_e:>14.4e}  {ratio:>8.3f}  {z:>+7.2f}  {flag:>6}")
    if common:
        ratios = [canary[q]["eps_sx"] / xeb[q]["eps_sx"] for q in common]
        print(f"\n  Geometric mean ratio (Canary/XEB): "
              f"{np.exp(np.mean(np.log(ratios))):.3f}  "
              f"({len(common)}/8 qubits with valid XEB)")
    print(f"{'='*72}")


def estimate_cost(qubits, Nv, shots_c, depths, n_circs, shots_x):
    n_q     = len(qubits)
    gates_c = sum(2 * N for N in Nv) * n_q * shots_c
    gates_x = sum(depths) * n_q * shots_x * n_circs
    return (gates_c + gates_x) * COST_PER_QGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits",        required=True,
                    help="e.g. 0,5,9,14,19,24,29,34  (QPU) or 0,3,6,9,12,15,18,21 (sim)")
    ap.add_argument("--backend",       default="ionq_simulator:forte-1")
    ap.add_argument("--live",          action="store_true")
    ap.add_argument("--xeb-depths",    default=None,
                    help="Comma-separated XEB depths. Default: 1,500,2000")
    ap.add_argument("--xeb-circuits",  type=int, default=XEB_CIRCUITS)
    ap.add_argument("--xeb-shots",     type=int, default=XEB_SHOTS)
    ap.add_argument("--canary-shots",  type=int, default=CANARY_SHOTS)
    ap.add_argument("--skip-xeb",      action="store_true")
    ap.add_argument("--resume",        default=None)
    args = ap.parse_args()

    qubits    = [int(q.strip()) for q in args.qubits.split(",")]
    xeb_depths = ([int(d) for d in args.xeb_depths.split(",")]
                  if args.xeb_depths else XEB_DEPTHS)

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
    print(f"  Qubits        : {qubits}")
    print(f"  XEB           : depths={xeb_depths}  circuits={args.xeb_circuits}  "
          f"shots={args.xeb_shots}")
    print(f"  Canary shots  : {args.canary_shots}")
    print(f"  Method        : Canary gate-rep then single-qubit XEB — same ions, sequential")
    print(f"  No error mitigation, no optimization, native gateset")

    backend = get_backend(args.backend, api_key)

    try:
        max_q = backend.configuration().n_qubits
        if max(qubits) >= max_q:
            sys.exit(f"Qubit index {max(qubits)} exceeds backend max ({max_q-1}). "
                     f"For simulator use indices 0-{max_q-1}.")
    except AttributeError:
        pass

    print("\nLoading per-qubit live profiles...")
    all_profiles = load_profiles(qubits, api_key, args.backend)

    if is_qpu(args.backend):
        Nv_est = all_profiles[qubits[0]].gate_rep_n
        Nv_est = [N if N % 2 == 0 else N + 1 for N in Nv_est]
        cost   = estimate_cost(qubits, Nv_est, args.canary_shots,
                               xeb_depths, args.xeb_circuits, args.xeb_shots)
        print(f"\n  Est. total cost : ${cost:,.0f}")
        print(f"  Budget impact   : ${cost:,.0f} of $30,000")
        if input("  Type RUN to confirm: ").strip() != "RUN":
            sys.exit("Aborted.")

    print("\n--- Canary gate repetition ---")
    canary_results, Nv = run_canary(backend, qubits, all_profiles,
                                    chk, chk_path, args.canary_shots)

    xeb_results = {}
    if not args.skip_xeb:
        print("\n--- Single-qubit XEB (Arute et al. Nature 2019) ---")
        xeb_results = run_xeb(backend, qubits, chk, chk_path,
                              args.xeb_shots, args.xeb_circuits, xeb_depths)

    print_comparison(canary_results, xeb_results, qubits)

    cal_summary = {q: dict(eps_prior=all_profiles[q].constants["eps_typical"],
                           p0g1=all_profiles[q].constants["p0_given_1"],
                           p1g0=all_profiles[q].constants["p1_given_0"])
                   for q in qubits}

    stem = f"ionq_canary_xeb_{'-'.join(map(str, qubits))}_{ts}"

    try:
        out_json = DATA_DIR / f"{stem}.json"
        with open(out_json, "w") as f:
            json.dump(dict(backend=args.backend,
                           timestamp=datetime.now(timezone.utc).isoformat(),
                           qubits=qubits,
                           gate_rep_N=Nv,
                           xeb_depths=xeb_depths,
                           calibration=cal_summary,
                           canary={str(k): v for k, v in canary_results.items()},
                           xeb={str(k): v for k, v in xeb_results.items()}),
                      f, indent=2)
        print(f"\n  Saved JSON : {out_json}")
    except Exception as e:
        print(f"\n  JSON save FAILED: {e}")

    try:
        out_csv = DATA_DIR / f"{stem}.csv"
        rows    = []
        for q in qubits:
            c = canary_results.get(q, {})
            x = xeb_results.get(q, {})
            rows.append(dict(
                qubit         = q,
                backend       = args.backend,
                timestamp     = datetime.now(timezone.utc).isoformat(),
                canary_eps_sx = c.get("eps_sx",      float("nan")),
                canary_sigma  = c.get("eps_sx_sigma", float("nan")),
                xeb_eps_sx    = x.get("eps_sx",      float("nan")),
                xeb_sigma     = x.get("eps_sx_sigma", float("nan")),
                p0g1          = cal_summary[q]["p0g1"],
                p1g0          = cal_summary[q]["p1g0"],
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