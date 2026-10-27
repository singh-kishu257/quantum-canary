"""Numerical checks behind the derivations of the manuscript's Appendix A (printed; nothing is written).
A1 Canary's RB difference D(m) = A p^m with no floor under gate-independent non-unital noise and readout error
   (exact density matrices, random sequences from the engine's Clifford table).
A2 QE's free-floor RB fit at three lengths is an exact interpolation; its noise amplification, its bias under noise
   alone, the data's consistency with a 1/2-floor exponential, and the effect of the readout-shifted floor (Kingston QE1).
A3 Fisher information per shot of Canary's delays and RB lengths.
A4 Ramsey delay ratios, the change in predicted probabilities for aliased offsets, and the out-of-range reports.
A5 Cramer-Rao bound of Canary's own design on the ideal-regime qubits (research/data/ideal_parity.csv).
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit, minimize_scalar
import hw_common as h
from quantum_canary import inversion as inv, simulation as sim

warnings.filterwarnings("ignore")
R = inv.RB_PULSES_PER_CLIFFORD
f = lambda x: x ** 2 * np.exp(-x) / (1 - np.exp(-x))
g = lambda x: x ** 2 * np.exp(-2 * x) / (1 - np.exp(-2 * x))


def a1(n=3000, lengths=(1, 20, 100, 300)):
    gates, mult, finish = inv._rb_clifford_table()
    rz = lambda t: np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])
    pulse = {None: np.eye(2), "sx": np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2, "x": np.array([[0, 1], [1, 0]])}
    U = [rz(c) @ pulse[p] @ rz(a) for a, p, c in gates]
    gam, th, e01, e10 = 0.004, 0.02, 0.03, 0.01
    K = [np.array([[1, 0], [0, np.sqrt(1 - gam)]]), np.array([[0, np.sqrt(gam)], [0, 0]])]
    Rx = np.cos(th / 2) * np.eye(2) - 1j * np.sin(th / 2) * np.array([[0, 1], [1, 0]])
    noise = lambda r: Rx @ sum(k @ r @ k.conj().T for k in K) @ Rx.conj().T
    M0, rho0 = np.diag([1 - e10, e01]), np.diag([0.99, 0.01]).astype(complex)
    P = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
    ptm = lambda ch: np.real(np.array([[np.trace(a @ ch(b)) / 2 for b in P] for a in P]))
    tw = np.mean([ptm(lambda r, u=u: u.conj().T @ noise(u @ r @ u.conj().T) @ u) for u in U], axis=0)
    p = np.sort(np.abs(np.linalg.eigvals(tw[1:, 1:])))[-1]
    X = np.array([[0, 1], [1, 0]])
    A = np.real(np.trace(M0 @ noise(rho0 - X @ rho0 @ X)))
    floor = np.real(np.trace(M0 @ noise(np.eye(2) / 2)))
    rng = np.random.default_rng(1)

    def mean_p0(m, end):
        out = []
        for _ in range(n):
            idx, net = list(rng.integers(24, size=m)), 0
            for i in idx:
                net = mult[net][i]
            r = rho0.copy()
            for i in idx + [finish[net][end]]:
                r = noise(U[i] @ r @ U[i].conj().T)
            out.append(np.real(np.trace(M0 @ r)))
        return np.mean(out)
    print(f"A1 twirl p {p:.6f}; predicted A = Tr[M0 L(psi0 - psi1)] = {A:.4f}; single-ended floor Tr[M0 L(I/2)] = {floor:.4f}")
    for m in lengths:
        print(f"   m {m:4d}: D/p^m = {(mean_p0(m, 0) - mean_p0(m, 1)) / p ** m:.4f}")


def interp(F, L):
    rr = (F[1] - F[2]) / (F[0] - F[1])
    try:
        a = brentq(lambda a: (a ** L[1] - a ** L[2]) / (a ** L[0] - a ** L[1]) - rr, 1e-9, 1 - 1e-12)
    except ValueError:
        return None
    A = (F[0] - F[1]) / (a ** L[0] - a ** L[1])
    return a, A, F[0] - A * a ** L[0]


def fixed(F, s, L, B=0.5):
    model = lambda m, A, a: A * a ** m + B
    (A, a), _ = curve_fit(model, L, F, p0=[0.49, 0.9995], sigma=s, absolute_sigma=True, bounds=([0, 0], [1, 1]))
    return a, A, float(np.sum(((F - model(L, A, a)) / s) ** 2))


def a2():
    L = np.array([10.0, 100.0, 500.0])
    epc = lambda a: (1 - a) / 2
    ref = h.qe_refits()
    ref = ref[(ref.parameter == "epsilon_sx") & (ref.step == "QE1")].set_index("qubit")
    cal = json.load(open(h.RUN / "calibration_C1.json"))
    pr = {i: {p["name"]: p["value"] for p in q} for i, q in enumerate(cal["qubits"])}
    rows = []
    for q, entries in h._qe_raw("QE1", "rb").items():
        per = {m: [e["counts"]["0"] / e["shots"] for e in entries if e["metadata"]["xval"] == m] for m in L}
        F = np.array([np.mean(per[m]) for m in L])
        s = np.array([np.std(per[m], ddof=1) / np.sqrt(len(per[m])) for m in L])
        rows.append((q, F, s, interp(F, L), fixed(F, s, L)))
    ok = [r for r in rows if r[3] is not None and r[0] in ref.index]
    mine = np.array([epc(r[3][0]) / epc(r[4][0]) for r in ok])
    qe = np.array([ref.loc[r[0], "default"] / ref.loc[r[0], "floor_fixed"] for r in ok])
    print(f"A2 {len(rows)} qubits, interior exact solution for {len(ok)}; free/fixed EPC ratio median: exact interpolation "
          f"{np.median(mine):.3f}, QE saved {np.median(qe):.3f}; |mine/QE - 1| median {np.median(np.abs(mine / qe - 1)):.1e}")
    B = np.array([r[3][2] for r in ok])
    print(f"   free floor median {np.median(B):.3f}, share > 1/2 {np.mean(B > 0.5):.2f}")
    chis = np.array([r[4][2] for r in rows])
    print(f"   floor-fixed fit chi2 (1 dof): median {np.median(chis):.2f}, share > 3.84 {np.mean(chis > 3.84):.2f}, "
          f"share > 10.83 {np.mean(chis > 10.83):.2f}")

    def sd(fun, F, s):
        base, gr = fun(F), []
        for k in range(3):
            dF = np.zeros(3); dF[k] = 1e-6
            gr.append((fun(F + dF) - base) / 1e-6)
        return float(np.sqrt(np.sum((np.array(gr) * s) ** 2)))
    amp = np.array([(sd(lambda X: epc(interp(X, L)[0]), F, s) / epc(ex[0]), sd(lambda X: epc(fixed(X, s, L)[0]), F, s) / epc(fx[0]))
                    for q, F, s, ex, fx in ok])
    print(f"   relative SD of EPC from sampling noise: free {np.median(amp[:, 0]):.2f}, fixed {np.median(amp[:, 1]):.3f}, "
          f"ratio {np.median(amp[:, 0] / amp[:, 1]):.1f}")
    rng = np.random.default_rng(3)
    sim_ratio = []
    for q, F, s, ex, fx in rows:
        v = [interp(fx[1] * fx[0] ** L + 0.5 + rng.normal(0, s), L) for _ in range(200)]
        sim_ratio.append(np.nanmedian([epc(x[0]) / epc(fx[0]) if x is not None else np.nan for x in v]))
    print(f"   noise alone (floor-fixed model + each qubit's sampling noise): median free/fixed {np.nanmedian(sim_ratio):.2f}")
    shift = np.array([0.5 * (pr[q]["prob_meas0_prep1"] - pr[q]["prob_meas1_prep0"]) for q, *_ in rows])
    rr = np.array([epc(fixed(F, s, L, 0.5 + dB)[0]) / epc(fx[0]) for (q, F, s, ex, fx), dB in zip(rows, shift)])
    print(f"   readout floor shift (e01 - e10)/2 median {np.median(shift):+.4f}; EPC with floor 1/2(1 + e01 - e10) / floor 1/2: "
          f"median {np.median(rr):.3f} (IQR {np.percentile(rr, 25):.3f}-{np.percentile(rr, 75):.3f})")
    med = np.median(np.array([r[1] for r in rows]), axis=0)
    print(f"   chip-median survival {np.round(med, 4)}; its exact free fit: floor {interp(med, L)[2]:.3f}")


def a3():
    xs = brentq(lambda x: x - 2 * (1 - np.exp(-x)), 0.5, 3)
    print(f"A3 x = 2(1 - e^-x): {xs:.4f} (numeric argmax {minimize_scalar(lambda x: -f(x), bounds=(.01, 10), method='bounded').x:.4f}); "
          f"g(x) = f(2x)/4: {np.allclose(g(np.linspace(.1, 3, 30)), f(2 * np.linspace(.1, 3, 30)) / 4)}; argmax g {xs / 2:.4f}")
    print(f"   relaxation 0.5/1/1.5 T: {[round(f(x) / f(xs), 3) for x in (0.5, 1, 1.5)]}; echo: "
          f"{[round(g(x) / g(xs / 2), 3) for x in (0.5, 1, 1.5)]}; RB m* eps = {xs / 2 / (2 * R):.3f}")
    for ratio in (0.25, 0.5, 1, 2, 4):
        x = 2 * R * 0.3 * ratio
        print(f"   RB longest length, true eps = {ratio} x prior: k m = {x:.3f}, share {g(x) / g(xs / 2):.3f}")
    share = lambda x: g(x) / g(xs / 2)
    x0 = 2 * R * 0.3
    print(f"   share at least the prior's ({share(x0):.3f}) up to eps/prior = "
          f"{brentq(lambda x: share(x) - share(x0), xs / 2, 3) / x0:.2f}; at least 0.80 for eps/prior in "
          f"{brentq(lambda x: share(x) - 0.8, 0.05, xs / 2) / x0:.2f}-{brentq(lambda x: share(x) - 0.8, xs / 2, 3) / x0:.2f}")


def a4():
    d = h.load()
    v, _ = h.wide(d, "delta_omega")
    rng = h.canary_dw_range_hz()
    dev, change = [], []
    for q in json.load(open(h.RUN / "C1.json"))["report"]["qubits"]:
        prof = inv.BackendProfile(architecture="superconducting", T1_prior_s=q["T1_prior_s"], T2_prior_s=q["T2_prior_s"],
                                  dt_ns=q["dt_ns"] or 0.2222, prior_confidence="prior_informed",
                                  custom_arch=dict(inv.ARCH_DEFAULTS["superconducting"], eps_typical=q["eps_prior"]))
        dl = np.array(prof.ramsey_delays_s)
        dev.append(np.max(np.abs(dl / dl[0] - [1, 5, 10, 15])))
        w, step = 2 * np.pi * 1e3, 2 * np.pi / dl[0]
        change.append(max(np.max(np.abs(np.cos(w * dl) - np.cos((w + k * step) * dl))) for k in (1, 2)) / 2)
    print(f"A4 delay ratio deviation from (1, 5, 10, 15): median {np.median(dev):.1e}, max {np.max(dev):.1e}; change of predicted "
          f"P1 for aliases k = 1, 2: median {np.median(change):.1e}, max {np.max(change):.1e}")
    hl = d[(d.tool == "canary") & (d.parameter == "delta_omega")].pivot(index="qubit", columns="step", values="health")
    near = {"C1": "QE1", "C2": "QE2", "C3": "QE2"}
    q1 = v.QE1.dropna()
    beyond = q1[q1.abs() >= rng.reindex(q1.index)].index
    kinds, green = {}, {}
    for q in beyond:
        inv_t = rng[q] / 0.45
        for s in h.CANARY_STEPS:
            c, e = v[s][q], v[near[s]][q]
            if not np.isfinite(e):
                kind = "no QE value"
            elif np.isclose(abs(c), rng[q], rtol=1e-3):
                kind = "at the range limit"
            elif abs(e - c) <= 3000:
                kind = "within 3 kHz of QE"
            elif abs((e - c) / inv_t - round((e - c) / inv_t)) * inv_t <= 3000:
                kind = f"alias k = {int(round((e - c) / inv_t)):+d}"
            else:
                kind = "other"
            kinds[kind] = kinds.get(kind, 0) + 1
            if hl[s][q] == "GREEN":
                green[kind] = green.get(kind, 0) + 1
    print(f"   {3 * len(beyond)} reports on {len(beyond)} out-of-range qubits: {kinds}; GREEN: {green}")
    print(f"   qubit 134: QE1 {v.QE1[134] / 1e3:.2f} kHz, 1/(2 tau_min) = {rng[134] / 0.9 / 1e3:.2f} kHz")


def a5():
    d = pd.read_csv(h.RUN.parents[1] / "ideal_parity.csv")
    shots = dict(t1=300, ram=600, rb=24, echo=500)

    def fisher(G, p, n):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        G = np.atleast_2d(G)
        return (G * (n / (p * (1 - p)))) @ G.T
    rows = []
    for (arch, _), gq in d.groupby(["architecture", "instance_id"]):
        gq = gq.set_index("parameter")
        truth = sim.generate_device(arch, n_qubits=1, seed=int(gq.seed.iloc[0]), regime="ideal", spam_spread_frac=0.0,
                                    perturbation_frac=None, dw_population="estimator_window")
        prior = sim.generate_calibration_prior(truth, qubit=0, t0_s=0.0, perturbation_frac=None)
        assert np.isclose(prior.T1_prior_s, gq.prior_value["T1_s"], rtol=1e-10)
        assert np.isclose(prior.epsilon_prior, gq.prior_value["epsilon_sx"], rtol=1e-10)
        e01, e10 = float(prior.spam_p0_given_1_prior), float(prior.spam_p1_given_0_prior)
        const = dict(inv.ARCH_DEFAULTS[arch])
        const.update(T1_s=prior.T1_prior_s, T2_s=prior.T2_prior_s, eps_typical=prior.epsilon_prior, p0_given_1=e01, p1_given_0=e10)
        prof = inv.BackendProfile(architecture=arch, T1_prior_s=prior.T1_prior_s, T2_prior_s=prior.T2_prior_s,
                                  dt_ns=inv.ARCH_DEFAULTS[arch]["dt_ns"], custom_arch=const, prior_confidence="prior_informed")
        vis = 1 - e01 - e10
        T1, T2, dw, eps = (gq.true_value[k] for k in ("T1_s", "T2_s", "delta_omega", "epsilon_sx"))
        t = np.array(prof.t1_delays_s); e = np.exp(-t / T1)
        i_t1 = fisher(vis * e * t / T1 ** 2, e10 + vis * e, shots["t1"])[0, 0]
        t = np.array(prof.echo_delays_s); e = np.exp(-t / T2)
        i_echo = fisher(-vis * 0.5 * e * t / T2 ** 2, e10 + vis * 0.5 * (1 - e), shots["echo"])[0, 0]
        t = np.array(prof.ramsey_delays_s); e = np.exp(-t / T2); cs, sn = np.cos(dw * t), np.sin(dw * t)
        i_ram = (fisher(np.vstack([-vis * 0.5 * cs * e * t / T2 ** 2, vis * 0.5 * e * t * sn]), e10 + vis * 0.5 * (1 - e * cs), shots["ram"])
                 + fisher(np.vstack([-vis * 0.5 * sn * e * t / T2 ** 2, -vis * 0.5 * e * t * cs]), e10 + vis * 0.5 * (1 - e * sn), shots["ram"]))
        p, c = 1 - 2 * R * eps, 0.5 * (1 - e10 + e01)
        G, pr = [], []
        for m, k in zip(prof.rb_lengths, (10, 20, 30)):
            for j in range(k):
                sg = 1 if j % 2 == 0 else -1
                pr.append(c + sg * 0.5 * vis * p ** m)
                G.append([sg * 0.5 * p ** m, -sg * vis * R * m * p ** (m - 1)])
        i_rb = fisher(np.array(G).T, np.array(pr), shots["rb"])
        crb = {"T1_s": i_t1 ** -0.5, "T2_s": i_echo ** -0.5, "delta_omega": np.linalg.inv(i_ram)[1, 1] ** 0.5,
               "epsilon_sx": np.linalg.inv(i_rb)[1, 1] ** 0.5}
        rows += [dict(arch=arch, param=k, ratio=gq.sigma_canary[k] / crb[k], z=(gq.canary_value[k] - gq.true_value[k]) / crb[k])
                 for k in crb]
    x = pd.DataFrame(rows)
    print(f"A5 {len(x) // 4} qubits; 2-sigma sampling uncertainty of an efficiency from 500 qubits: +-{2 * np.sqrt(2 / 499):.3f}")
    for (a, p), gq in x.groupby(["arch", "param"], sort=False):
        print(f"   {a[:2]} {p:12s} sigma/CRB median {gq.ratio.median():.3f} | efficiency 1/var(err/CRB) {1 / gq.z.var(ddof=1):.2f}")


if __name__ == "__main__":
    a1(); a2(); a3(); a4(); a5()
