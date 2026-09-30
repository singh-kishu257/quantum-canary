# Fix report: information acquisition in Quantum Canary

Branch `fix/information-acquisition`. Workload, device model, seeds and budgets are those of
`research/hsim_before_after.py`: 20 sessions (seeds `1000 + s`), 12 ions per session from IonQ's 30 benchmarked
qubits, per-pair ZZ error drawn from each pair's own direct-RB history, HSIMN08V01 (63 ZZ, 7 chain pairs × 9),
5 batches × 2,000 shots, default `CanaryConfig` except `seed` and `anneal_steps=4000`. Fidelity is the Hellinger
fidelity to the ideal distribution; ΔF is paired by session, with 95% bootstrap CIs over sessions (4,000 resamples).
Files: `hsim_results_before.json` (code before these changes), `hsim_results_fixed.json` (after),
`hsim_results_control.json`, `ablations/variant_*.json`. Tables come from `hsim_summarize.py`.

## Summary

| Target | Result | Met? |
|---|---|---|
| Canary delivered: ΔF vs identity > 0 with CI lower bound > 0 (original) | +0.003 [−0.014, +0.023] | **No** |
| Canary delivered: CI lower bound ≥ −0.005 (revised) | lower bound −0.014 | **No** |
| Canary final map: ΔF vs identity ≥ +0.03 (original) | +0.034 [+0.017, +0.055] | Yes (mean; the CI includes values below 0.03) |
| Canary final map captures ≥ 50% of the perfect-screening (>2%) gain over the random-map mean (revised) | 157% [134%, 192%] | Yes, but not discriminating: the old code scored 88% |
| No session where Canary delivered is worse than identity by > 0.03 | 4 of 20 (worst −0.076) | **No** (before: 9 of 20) |
| Calibration shot fraction ≤ 0.35 | 0.333 in every session (sweep reported separately: 0.574 including the sweep) | Yes |
| Verdicts unchanged in kind: e-BH on Tier-1 evidence only | `decision.py` and `estimator.py` unchanged; isolation tests pass; 0 false degraded verdicts | Yes |
| No-heterogeneity control: Canary not worse than identity by > 0.01 on average | delivered +0.002 [−0.001, +0.004], final map +0.002 | Yes |

What changed in outcome, paired against the same sessions before the changes:

- **Final map:** +0.031 [+0.009, +0.060] better (17/20 sessions). The true weighted pair error of the final map
  fell from 0.320 to 0.250. For reference: random-map mean 0.424, perfect screening 0.299 (>2%) and 0.275 (>1%),
  oracle 0.168. It now captures 61% [53%, 69%] of the oracle gain over the random-map mean (before: 34%).
- **Delivered data:** +0.028 [+0.015, +0.042] better (17/20). This moves it from significantly worse than identity
  (−0.025) to indistinguishable from identity (+0.003). The first batches are still the problem (cold start, below).
- **Distinct pairs with Tier-1 evidence:** 19.9 of 66 per session, up from 5.
- **With the optional warm-start sweep** (9,000 extra shots, not given to the baselines): delivered +0.048
  [+0.032, +0.068] vs identity, 20/20 sessions, worst session +0.005.

## Results: before vs after (20 sessions)

Before (code at `adb4d9c`; reproduces the reported table):

| Method | Mean F | Obs. RMSE | ΔF vs identity [95% CI] | median / min / max ΔF | wins |
|---|---|---|---|---|---|
| identity (raw native) | 0.815 | 0.1291 | — | — | — |
| random map (raw native) | 0.818 | 0.1305 | +0.003 [−0.029, +0.035] | +0.009 / −0.176 / +0.150 | 14/20 |
| debias-style avg (25 maps) | 0.786 | 0.1448 | −0.029 [−0.045, −0.014] | −0.020 / −0.109 / +0.027 | 4/20 |
| hardened only (twirl, identity map) | 0.816 | 0.1284 | +0.001 [−0.002, +0.003] | +0.001 / −0.013 / +0.014 | 11/20 |
| Canary delivered (all batches) | 0.790 | 0.1442 | −0.025 [−0.043, −0.006] | −0.029 / −0.094 / +0.053 | 6/20 |
| Canary last 2 batches | 0.805 | 0.1360 | −0.010 [−0.031, +0.011] | −0.016 / −0.112 / +0.097 | 5/20 |
| Canary final map (deployed) | 0.818 | 0.1270 | +0.003 [−0.031, +0.034] | +0.010 / −0.232 / +0.148 | 12/20 |

After (default configuration; rows below the line were added, run after the original rows on an independent seed
stream):

| Method | Mean F | Obs. RMSE | ΔF vs identity [95% CI] | median / min / max ΔF | wins |
|---|---|---|---|---|---|
| identity (raw native) | 0.815 | 0.1291 | — | — | — |
| random map (raw native) | 0.818 | 0.1305 | +0.003 [−0.029, +0.035] | +0.009 / −0.176 / +0.150 | 14/20 |
| debias-style avg (25 maps) | 0.786 | 0.1448 | −0.029 [−0.045, −0.014] | −0.020 / −0.109 / +0.027 | 4/20 |
| hardened only (twirl, identity map) | 0.815 | 0.1287 | +0.000 [−0.002, +0.003] | −0.001 / −0.009 / +0.012 | 10/20 |
| **Canary delivered (all batches)** | **0.818** | 0.1268 | **+0.003 [−0.014, +0.023]** | +0.005 / −0.076 / +0.107 | 11/20 |
| Canary last 2 batches | 0.839 | 0.1127 | +0.024 [+0.008, +0.045] | +0.014 / −0.015 / +0.143 | 16/20 |
| **Canary final map (deployed)** | **0.849** | 0.1051 | **+0.034 [+0.017, +0.055]** | +0.026 / −0.025 / +0.151 | 17/20 |
| Canary + sweep delivered (all batches) | 0.863 | 0.0972 | +0.048 [+0.032, +0.068] | +0.042 / +0.005 / +0.160 | 20/20 |
| Canary + sweep final map (deployed) | 0.859 | 0.0998 | +0.045 [+0.028, +0.066] | +0.031 / −0.003 / +0.163 | 19/20 |
| Canary final map (raw native) | 0.849 | 0.1061 | +0.034 [+0.017, +0.055] | +0.027 / −0.025 / +0.146 | 16/20 |
| random map (mean of 100) | 0.779 | 0.1575 | −0.036 [−0.053, −0.018] | −0.032 / −0.106 / +0.042 | 4/20 |
| perfect screening >2% (mean of 100) | 0.824 | 0.1233 | +0.009 [−0.009, +0.031] | −0.001 / −0.042 / +0.134 | 10/20 |
| perfect screening >1% (mean of 100) | 0.836 | 0.1151 | +0.021 [+0.004, +0.042] | +0.010 / −0.021 / +0.141 | 14/20 |
| oracle map (raw native) | 0.893 | 0.0791 | +0.078 [+0.061, +0.100] | +0.062 / +0.038 / +0.203 | 20/20 |
| oracle map (deployed) | 0.892 | 0.0797 | +0.077 [+0.061, +0.098] | +0.063 / +0.039 / +0.203 | 20/20 |

Identity, random and debias rows are identical per session in both runs (to 1e-12). "Hardened only" twirls with the
Canary's random generator, whose state after the batches depends on the Canary code, so it differs slightly (0.816
vs 0.815). "Perfect screening" is the mean over 100 random maps that avoid every used pair whose true infidelity
exceeds the threshold. "Oracle" is the best of 8 annealing runs (50,000 steps) on the true pair and readout tables.

Per-session diagnostics (after): calibration shot fraction 0.333 in all sessions (4 of 12 circuits per batch: one
packed CB pair of circuits and two SPAM circuits); 15,000 shots spent per session, equal to the budget given to the
baselines; distinct Tier-1 pairs 19.85 (19–20); sessions with a batch in which no run was trusted: 4 (one batch each
in sessions 3, 4, 6 and 8).

### Gain accounting (over the mean of 100 random maps, F = 0.779)

| Method | gain | fraction of perfect screening >2% gain | fraction of perfect screening >1% gain | fraction of oracle gain |
|---|---|---|---|---|
| Canary final map (raw native), after | +0.070 | 1.57 [1.34, 1.92] | 1.23 [1.11, 1.37] | 0.61 [0.53, 0.68] |
| Canary final map (deployed), after | +0.070 | 1.57 [1.36, 1.90] | 1.23 [1.12, 1.37] | 0.61 [0.52, 0.69] |
| Canary final map (deployed), before | +0.039 | 0.88 | 0.69 | 0.34 |
| Canary delivered, after | +0.039 | 0.88 [0.63, 1.18] | 0.69 [0.51, 0.87] | 0.34 [0.24, 0.44] |
| Canary delivered, before | +0.011 | 0.24 | 0.19 | 0.09 |
| Canary + sweep final map (deployed) | +0.080 | 1.80 [1.58, 2.17] | 1.41 [1.32, 1.56] | 0.70 [0.65, 0.74] |
| Canary + sweep delivered | +0.084 | 1.88 [1.61, 2.32] | 1.47 [1.35, 1.65] | 0.73 [0.69, 0.77] |
| identity (raw native) | +0.036 | 0.80 [0.39, 1.22] | 0.63 [0.31, 0.92] | 0.31 [0.16, 0.46] |
| perfect screening >2% | +0.045 | — | — | — |
| perfect screening >1% | +0.057 | — | — | — |
| oracle map | +0.114 | — | — | — |

"Before" fractions use the after run's reference rows. That is valid because the devices and reference maps are
identical; there is no CI because the reference rows were not run alongside the old code. In these sessions the
identity map is itself 0.80 of the >2% perfect-screening gain above the random-map mean (weighted pair error 0.318 vs
0.424). So the revised target (≥ 50%) does not separate a good method from a lucky fixed map: the old code passed
it too. Against identity, Canary's final map beats perfect screening at both thresholds (+0.034 vs +0.009 and
+0.021): pair-level CB estimates rank the bulk of pairs, not only the outliers.

## No-heterogeneity control (20 sessions, every pair's stochastic error 0.0042)

Same sessions and seeds; only the depolarizing part of every pair is set to 0.0042 (coherent over-rotations,
readout and pulse errors unchanged). Default configuration, original rows only.

| Method | Mean F | Obs. RMSE | ΔF vs identity [95% CI] | median / min / max ΔF | wins |
|---|---|---|---|---|---|
| identity (raw native) | 0.840 | 0.1120 | — | — | — |
| random map (raw native) | 0.840 | 0.1121 | −0.000 [−0.003, +0.003] | −0.000 / −0.011 / +0.014 | 10/20 |
| debias-style avg (25 maps) | 0.841 | 0.1117 | +0.000 [−0.002, +0.002] | +0.001 / −0.008 / +0.010 | 11/20 |
| hardened only (twirl, identity map) | 0.841 | 0.1109 | +0.001 [−0.001, +0.003] | +0.001 / −0.007 / +0.012 | 12/20 |
| Canary delivered (all batches) | 0.842 | 0.1106 | +0.002 [−0.001, +0.004] | +0.001 / −0.006 / +0.016 | 11/20 |
| Canary last 2 batches | 0.843 | 0.1096 | +0.003 [+0.000, +0.005] | +0.004 / −0.006 / +0.015 | 13/20 |
| Canary final map (deployed) | 0.842 | 0.1103 | +0.002 [−0.001, +0.004] | +0.001 / −0.010 / +0.013 | 13/20 |

Canary is not worse than identity on average (+0.0017; worst session −0.006 delivered, −0.010 final). It made 0
degraded verdicts (630 healthy, 10 undecided). Calibration fraction 0.333; 20 pairs measured per session.

## Ablations (sessions 0–9)

Changes are switched off through their config fields (`research/hsim_ablations.py`). The default variant reproduces
the acceptance run's sessions 0–9 exactly, and "before" reproduces the original code's sessions 0–9 exactly.

| Variant | delivered ΔF vs identity [95% CI] | final ΔF vs identity [95% CI] | worst delivered ΔF | final-map pair error | Tier-1 pairs | cal. fraction |
|---|---|---|---|---|---|---|
| before (all changes off) | −0.010 [−0.039, +0.018] | +0.006 [−0.057, +0.060] | −0.092 | 0.356 | 5.0 | 0.333 |
| A | −0.005 [−0.030, +0.017] | +0.053 [+0.022, +0.088] | −0.085 | 0.254 | 19.8 | 0.333 |
| A+B | −0.005 [−0.030, +0.017] | +0.055 [+0.022, +0.092] | −0.085 | 0.251 | 20.0 | 0.333 |
| A+B+C | +0.014 [−0.017, +0.044] | +0.061 [+0.029, +0.098] | −0.089 | 0.238 | 19.7 | 0.333 |
| A+B+C+D (default) | +0.012 [−0.020, +0.045] | +0.050 [+0.023, +0.084] | −0.076 | 0.256 | 19.8 | 0.333 |
| default, explore_mode="thompson" | +0.022 [−0.010, +0.055] | +0.037 [−0.005, +0.079] | −0.072 | 0.286 | 19.4 | 0.333 |
| default, screen_calibration=False | +0.015 [−0.017, +0.046] | +0.057 [+0.029, +0.090] | −0.087 | 0.243 | 19.8 | 0.333 |
| default, screening=False | +0.015 [−0.025, +0.056] | +0.048 [+0.014, +0.086] | −0.097 | 0.267 | 20.0 | 0.333 |

"A+B+C" means screening, local exploration, dilution calibration and suspect-first probing on, with D off
(`exploit_fraction=0.25`, `explore_typical=False`, `trust_requires_vetting=False`). "A" and "A+B" use Thompson
exploration from the population mixture, as before.

Paired differences between variants (same sessions):

| Comparison | delivered: paired ΔF [95% CI] (sessions better) | final map: paired ΔF [95% CI] (sessions better) |
|---|---|---|
| A − before | +0.004 [−0.004, +0.013] (7/10) | **+0.046 [+0.019, +0.089] (10/10)** |
| A+B − A | +0.000 [−0.000, +0.000] (5/10) | +0.002 [−0.000, +0.007] (6/10) |
| A+B+C − A+B | **+0.019 [+0.007, +0.033] (8/10)** | +0.006 [−0.005, +0.018] (6/10) |
| A+B+C+D − A+B+C | −0.001 [−0.011, +0.007] (5/10) | −0.011 [−0.020, +0.001] (1/10) |
| local − thompson (both default otherwise) | −0.009 [−0.024, +0.004] (3/10) | +0.013 [−0.007, +0.039] (5/10) |
| calibration on − off | −0.003 [−0.009, +0.003] (4/10) | −0.007 [−0.016, +0.003] (1/10) |
| screening on − off | −0.003 [−0.016, +0.008] (5/10) | +0.002 [−0.013, +0.016] (5/10) |

What contributes what, on 10 sessions:

- **A (packing):** almost all of the final-map gain. It measures 4 pairs per CB circuit pair instead of 1, so the
  exploit's 7 pairs are all calibrated within two batches.
- **B (coverage):** no measurable effect. With one pack per batch, the design pairs fill the budget before coverage
  candidates compete; test 3 shows coverage working when the designs are narrow (32 vs 6 pairs).
- **C:** the delivered-data gain (+0.019). Exploratory seatings stay near the exploit instead of being near random.
  Within the full system, turning screening alone off changes nothing measurable, so this gain most likely comes
  from local exploration rather than from screening. That is not separately ablated with D off.
- **D:** did not help. It is slightly negative on the final map (−0.011, CI reaching +0.001), plausibly because
  `exploit_fraction=0.5` leaves 2 explorers instead of 3.
- **Dilution calibration:** did not help (−0.007 final map, CI reaching +0.003).
- **Local vs Thompson exploration:** not distinguishable at this sample size.

## Verdict validity

- `decision.py`, `estimator.py`, `compiler.py`, `resources.py`, `program.py`, `executors.py`, `emulator.py` and
  `assignment.py` are byte-identical to `main`.
- `tests/test_screening.py::test_screening_never_changes_tier1_evidence_or_verdicts` checks three things:
  - with and without screening on the same Tier-1 data, every `HealthModel` evidence list and `canary.decisions`
    are identical, even though the designs differ;
  - `HealthModel.record` and `HealthModel.update` are never called by a screening-enabled batch without probes
    (patched to raise);
  - corrupting the screening model does not change recomputed verdicts.
- Degraded verdicts checked against the true tables: 17 in the acceptance run (14 with true infidelity ≥ 2%, 3 in
  the 0.8–2% indifference zone, 0 at or below the 0.8% healthy threshold); 0 in the control; 0 false in every
  ablation variant. The old code made 1 degraded verdict in 20 sessions; it measured about 5 pairs per session.
- Healthy verdicts are stored only as counts in the result files, so they were not checked per resource.

## Changes (commits)

| Commit | Change |
|---|---|
| `a640d00` | `--forte-dir` for the experiment script |
| `8bb3cc5` | Port stale estimator tests (see deviations) |
| `3e00c8c` | A: packed cycle-benchmarking probes |
| `8ade70a` | B: coverage-aware probe selection, `Canary.sweep` |
| `8068cc2` | C: screening model (guidance only) |
| `f5886d4`, `2e6c8df` | D: exploration and trust hygiene; its tests (committed separately on request) |
| `5a5f465` | C extension: local exploration (`explore_mode`) |
| `ad592b8`, `7f1fbb3` | C extension: per-program dilution calibration; keep the suspicion penalty with vague calibrated readings |
| `39105a7` | Suspect-first probe priority |
| `660fa44`, `639a5b9`, `c57af9f` | Evaluation rows and flags, summariser, ablation driver, README, reproduced baseline |

With every switch at its old setting (`probe_pack=1, coverage_weight=0, screening=False, explore_mode="thompson",
exploit_fraction=0.25, explore_typical=False, trust_requires_vetting=False`), a 4-batch emulator run gives
evidence, verdicts, designs, probes and delivered frequencies identical to the old code to 1e-9 (checked after every
commit). On the HSIM experiment it reproduces the old Canary rows exactly (sessions 0–9).

## Config fields added (all in `CanaryConfig`)

| Field | Default | Change | Meaning | Old behaviour |
|---|---|---|---|---|
| `probe_pack` | `4` | A | Maximum disjoint ion pairs per cycle-benchmarking circuit | `1` |
| `coverage_weight` | `1.0` | B | Probe weight = design stake + `coverage_weight * exp(-discounted Tier-1 observation count)`; every pair among the available ions is a candidate | `0.0` |
| `screening` | `True` | C | Learn relative per-resource damping from algorithm contrasts (guidance only) | `False` |
| `screen_model_sd` | `0.05` | C | Log-units floor added to each screening observation's variance | — |
| `screen_min_signal_z` | `4.0` | C | A group is used only if every \|m\| > max(0.05, z·se) and all signs agree | — |
| `screen_scale` | `0.0018` | C | Placement penalty per unit of positive suspicion for resources without Tier-1 evidence. `pair_damping(0.005) / 3`, so a 3σ suspicion adds about the prior mean damping of a pair | `0.0` |
| `explore_mode` | `"local"` | C ext. | `"local"`: explorers are the exploit with one ion moved or two swapped (≤ 3 used pairs changed), chosen to test uncalibrated or suspect pairs; `"thompson"`: independent posterior draws | `"thompson"` |
| `screen_calibration` | `True` | C ext. | Rescale each program's screening estimates by a dilution factor fitted to pairs that also have Tier-1 estimates (weighted, clipped to [0.05, 1], ≥ 3 pairs). Placement costs of pairs without Tier-1 evidence then use the population prior updated with that reading | `False` |
| `exploit_fraction` | `0.5` | D | Share of algorithm circuits for the exploit; explorers share the rest, two circuits each where possible | `1 / assignments_per_batch` |
| `explore_typical` | `True` | D | Draws for resources without Tier-1 evidence come from the typical population component only (added beyond the brief so D can be switched off) | `False` |
| `trust_requires_vetting` | `True` | D | A run is trusted only if every pair it uses has Tier-1 evidence or suspicion < `trust_suspicion_z` (and no resource is degraded) | `False` |
| `trust_suspicion_z` | `2.0` | D | Suspicion threshold for trust, for skipping local moves onto suspect pairs, and for "suspect" in probe priority | — |

## Files changed

| File | Why |
|---|---|
| `src/quantum_canary/probes.py` | `cb_packed_program` (k disjoint CB sequences in one circuit) and the packing caveat |
| `src/quantum_canary/results.py` | `marginal_frequencies`; exploit-only fallback in `BatchResult.frequencies()/counts()` |
| `src/quantum_canary/canary.py` | Packed CB jobs and split learning; all-pairs coverage selection with suspect-first tiers; `sweep()`; screening updates, per-program guidance, persistence (state version 3, version 2 loads with empty screening); circuit allocation; trust rule |
| `src/quantum_canary/design.py` | Suspicion penalty; calibrated screening costs; local exploration; typical-component draws |
| `src/quantum_canary/screening.py` (new) | `ScreeningModel`: per-program centred contrasts, forgetting, ridge, suspicion, dilution calibration |
| `src/quantum_canary/config.py` | New fields, validation, docstring |
| `src/quantum_canary/__init__.py` | Export `ScreeningModel`, `cb_packed_program` |
| `tests/test_packed_probes.py` (new) | Tests 1–2 |
| `tests/test_coverage.py` (new) | Tests 3–4; probe priority |
| `tests/test_screening.py` (new) | Tests 5–6; calibration; costs; persistence |
| `tests/test_local_exploration.py` (new) | Local exploration contract |
| `tests/test_trust.py` (new) | Tests 7–8 |
| `tests/test_estimator.py` | Ported off removed APIs (see deviations) |
| `research/hsim_before_after.py` | `--forte-dir`; added rows and flags (original rows unchanged) |
| `research/hsim_summarize.py`, `research/hsim_ablations.py` (new) | Statistics and ablation driver |
| `README.md` | Call table (`probe`, `sweep`), design/co-schedule/learn/decide steps, limits |

`pytest -q`: 110 passed.

## Limitations, and what did not work

1. **Delivered data misses its targets (cold start).** In batch 1 there is no evidence, so the exploit seating is
   arbitrary, and in these sessions an arbitrary seating is worse than identity (the random-map mean is 0.036
   below identity). The last two batches are +0.024 [+0.008, +0.045] vs identity, with no session below −0.015.
   All five batches together are +0.003, with sessions 1, 4, 5 and 16 below −0.03 (−0.031, −0.076, −0.045, −0.055).
   The sweep removes the cold start (+0.048, 20/20, worst +0.005) but costs 9,000 shots, 60% of the batches'
   budget, which the baselines were not given.
2. **The revised final-map target is not discriminating**, as shown above (old code: 88%).
3. **Change D did not help**, and the final map got slightly worse (−0.011 [−0.020, +0.001], better in only 1 of 10
   sessions).
4. **Dilution calibration did not help** (−0.007 [−0.016, +0.003] on the final map). A factor was fitted in 20/20
   sessions (mean 0.59), but it is poorly determined: 5 sessions clip at 1.0 and 3 are at or near the 0.05 floor.
   The Tier-1-calibrated pairs are mostly the good pairs Canary chose to use. Their spread (roughly 0.002 in
   damping) is small against screening standard errors (roughly 0.02), so the slope's standard error is of order 2.
   Suspect-first probing did not noticeably change the fitted factors in a 6-session diagnostic. Keeping the
   suspicion penalty on top was necessary: when calibrated readings replaced it, a 10.5% pair with suspicion 4.7
   lost its only penalty.
5. **Screening's effect is not measurable inside the full system** (on − off: −0.003 delivered, +0.002 final), and
   it mis-attributes between co-seated pairs. `LogicalProgram.exposure` counts every lightcone gate as a full
   exposure, but HSIM is a near-identity evolution, so distant observables are almost insensitive to a pair they
   are "exposed" to. Their precise near-±1 values pull the estimate of the true culprit down. In a diagnostic run
   with the C commit and default probing (session 4), a 10.5% pair had suspicion −0.36 while a co-seated 4.4% pair
   had +3.14. Without probes, the planted 8% pair in test 6 ranked first in 10/10 seeds. In 8 algorithm-only HSIM
   sessions (C commit), 5 of the 7 exercised pairs above 4% ranked first or second (the others third and fourth),
   while several 2–3.5% pairs ranked 25–44.
6. **Coverage (B) has no measurable effect on this workload**, because the design pairs fill the one-pack budget.
7. **Packed probes are validated only on the emulator,** which has no idle or crosstalk error. Packed and
   single-pair estimates agreed within |Δ|/σ ≤ 1.5 for 4 pairs, and a planted 5% pair was estimated at 4.76%. On
   hardware the pairs of a packed circuit idle while the others are gated; compare packed and single-pair estimates
   for a few pairs before relying on packing (docstring and README).
8. **Two different fallbacks.** The script's "Canary delivered" row falls back to all runs when none is trusted,
   while the library now falls back to the exploit run. This happened in 4 of 100 batches (sessions 3, 4, 6, 8).
   The script row was kept as written (existing rows unchanged).
9. **The sweep exhausts per-ion probe tokens.** Each ion spends 11 × 2 × 250 = 5,500 against the 5,000-token cap, so
   one of the five batch CB packs was suppressed after the sweep: 18 calibration circuits over 5 batches instead of
   20, an in-batch fraction of 0.310.
10. **Deviations from the brief:**
    - `tests/test_estimator.py` was edited, although only new tests were allowed. It called `HealthModel` methods
      removed before this work (5 failures on `main`), and `estimator.py` may not change. The tests now check the
      same properties through `record`/`update`; the relative-group test moved to `test_screening.py`.
    - `explore_typical` was added so D can be ablated.
    - D's tests went in a separate commit, because implementation was committed first on request.
    - The branch also carries in-progress result snapshot commits, superseded by the final results commit.
11. **The per-program screening key** is a hash of the gates and measurements, so recompiling the same circuit at a
    different optimisation level starts a new screening model.
12. **Small sample.** The ablations use 10 sessions, and most sub-component differences are within their CIs.
