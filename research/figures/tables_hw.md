# IBM Kingston hardware tables (ibm_kingston_20261026T014902Z)

## Table H1

Bland–Altman agreement of Canary (mean of the two checks either side of each QE run) with qiskit-experiments in IBM's standard configuration on all operational qubits of ibm_kingston. Differences are Canary − QE, relative to the pair mean (Δω: in Hz). 'Floor fixed': qiskit-experiments' own analysis re-run on its raw counts with the decay floor fixed (RB asymptote 1/2, its value for symmetric readout and unital noise; T1 offset at IBM's P(1|0)); the default re-run reproduces every saved value. 'Within Canary range': qubits whose QE detuning lies inside Canary's unambiguous Ramsey range (Table H5). Predicted: mean and 2.5–97.5th percentiles of Canary − QE standard in the NISQ simulation.

| Parameter | Comparator | Sandwich | n | Bias [95% CI] | 95% LoA (±CI) | 2.5–97.5th pct. | Predicted (in silico) |
|---|---|---|---|---|---|---|---|
| T1 (%) | QE default | mean(C1,C2) − QE1 | 154 | 16.0 [12.1, 19.8] | -31.8 to 63.7 (±6.7) | -23.2 to 67.0 | -1.47; -24.5 to 18.8 |
| T1 (%) | QE default | mean(C2,C3) − QE2 | 154 | 16.1 [12.1, 20.1] | -32.9 to 65.1 (±6.9) | -24.1 to 64.7 | -1.47; -24.5 to 18.8 |
| T1 (%) | QE, floor fixed | mean(C1,C2) − QE1 | 154 | -0.118 [-2.87, 2.63] | -34.0 to 33.8 (±4.8) | -28.3 to 36.3 | -1.47; -24.5 to 18.8 |
| T1 (%) | QE, floor fixed | mean(C2,C3) − QE2 | 154 | -1.57 [-4.10, 0.949] | -32.6 to 29.5 (±4.4) | -36.7 to 31.8 | -1.47; -24.5 to 18.8 |
| T2 (%) | QE default | mean(C1,C2) − QE1 | 153 | -3.01 [-7.85, 1.83] | -62.4 to 56.4 (±8.4) | -64.6 to 61.8 | -2.04; -23.6 to 18.5 |
| T2 (%) | QE default | mean(C2,C3) − QE2 | 153 | 0.313 [-4.19, 4.82] | -55.0 to 55.6 (±7.8) | -66.0 to 60.4 | -2.04; -23.6 to 18.5 |
| Δω/2π (Hz) | QE default | mean(C1,C2) − QE1 | 149 | 200 [-1,859, 2,260] | -24,736 to 25,136 (±3,567) | -21,575 to 31,745 | 119; -719 to 1,276 |
| Δω/2π (Hz) | QE default | mean(C2,C3) − QE2 | 149 | 955 [-853, 2,762] | -20,933 to 22,842 (±3,131) | -14,029 to 31,773 | 119; -719 to 1,276 |
| Δω/2π (Hz) | QE default, within Canary range | mean(C1,C2) − QE1 | 135 | 210 [-16.5, 437] | -2,399 to 2,820 (±393) | -2,453 to 4,409 | 119; -719 to 1,276 |
| Δω/2π (Hz) | QE default, within Canary range | mean(C2,C3) − QE2 | 136 | 269 [-214, 752] | -5,313 to 5,851 (±836) | -3,659 to 4,416 | 119; -719 to 1,276 |
| ε (%) | QE default | mean(C1,C2) − QE1 | 154 | -92.4 [-105, -80.2] | -242 to 57.3 (±21) | -178 to 76.3 | 2.91; -65.3 to 82.3 |
| ε (%) | QE default | mean(C2,C3) − QE2 | 153 | -91.7 [-105, -78.7] | -252 to 68.2 (±23) | -180 to 80.8 | 2.91; -65.3 to 82.3 |
| ε (%) | QE, floor fixed | mean(C1,C2) − QE1 | 154 | -9.68 [-14.0, -5.40] | -62.5 to 43.1 (±7.4) | -54.6 to 54.3 | 2.91; -65.3 to 82.3 |
| ε (%) | QE, floor fixed | mean(C2,C3) − QE2 | 153 | -14.3 [-18.2, -10.3] | -62.9 to 34.4 (±6.9) | -55.8 to 46.4 | 2.91; -65.3 to 82.3 |

## Table H2

Repeatability of each method between repeats about 20 minutes apart (real drift included) and whether it explains the Canary − QE spread: if the two methods disagree only through their own noise and drift, the SD of mean(C1,C2) − QE1 is about √(SD²(C1→C2)/4 + SD²(QE1→QE2)/2). Robust SD = 1.4826 × MAD.

| Parameter | QE analysis | Robust SD, C1→C2 | Robust SD, QE1→QE2 | Predicted SD of Canary − QE | Observed SD of Canary − QE | Median \|change\| C1→C2 | Median \|change\| QE1→QE2 |
|---|---|---|---|---|---|---|---|
| T1 (%) | QE default | 14.4 | 19.6 | 15.6 | 20.4 | 9.85 | 14.5 |
| T1 (%) | QE, floor fixed | 14.4 | 14.1 | 12.3 | 11.3 | 9.85 | 9.55 |
| T2 (%) | QE default | 14.1 | 21.2 | 16.6 | 23.4 | 10.2 | 15.6 |
| Δω/2π (Hz) | QE default, within Canary range | 730 | 693 | 611 | 527 | 481 | 448 |
| ε (%) | QE default | 19.8 | 38.6 | 29.1 | 60.2 | 14.5 | 26.1 |
| ε (%) | QE, floor fixed | 19.8 | 8.05 | 11.4 | 18.9 | 14.5 | 5.49 |

## Table H3

Drift, error-bar and model-fit tests. Drift: for qubits whose two Canary checks differ by more than three combined standard errors, a drifting qubit puts QE1 (measured in between) between them. |z| between repeats tests Canary's error bars where drift is small (ε) and shows drift where it is not (T1, Δω). χ² limits as in the simulation (nominal 1.25% over limit for a correct model). Last column: agreement with QE (floor fixed for T1 and ε) on qubits whose Canary fit passes vs is over its χ² limit in C1.

| Parameter | Changes C1→C2 > 3σ | QE1 between C1 and C2: large / small change | Mean change C1→C2 [95% CI] | \|z\| ≤ 2, C1→C2 / C2→C3 (nominal 95.4%) | χ² over limit, C1/C2/C3 | χ² over limit, simulated (Table S4) | Median \|Canary − QE\|: fits pass / over limit |
|---|---|---|---|---|---|---|---|
| T1 (%) | 27/154 | 12/27 (44%) / 20/127 (16%) | -2.10 [-5.23, 1.03] | 70.8% / 62.3% | 12% / 11% / 14% | 1.8% | 7.11 / 15.6 |
| T2 (%) | 8/153 | 2/8 (25%) / 27/145 (19%) | 0.507 [-2.88, 3.89] | 90.2% / 79.7% | 6% / 8% / 3% | 1.2% | 15.4 / 28.8 |
| Δω/2π (Hz) | 71/154 | 37/68 (54%) / 15/81 (19%) | 61.3 [-421, 544] | 42.2% / 31.2% | 26% / 22% / 30% | 24.0% | 286 / 715 |
| ε (%) | 4/154 | 0/4 (0%) / 2/150 (1%) | -0.0227 [-4.12, 4.08] | 92.9% / 90.2% | 1% / 2% / 3% | 2.2% | 16.4 / 12.4 |

## Table H4

Gate error against IBM's own published sx gate error (IBM's randomized benchmarking, measured about 12 h before the run; not independent of drift since then).

| Method | Median ε | Median ratio to IBM [IQR] | Bias vs IBM, % [95% CI] | 95% LoA vs IBM (%) |
|---|---|---|---|---|
| Canary C1 | 2.246e-04 | 0.96 [0.79, 1.15] | -4.19 [-10.4, 1.99] | -80.3 to 71.9 |
| Canary C2 | 2.177e-04 | 0.96 [0.80, 1.15] | -4.23 [-10.9, 2.41] | -86.0 to 77.5 |
| QE1 default | 9.828e-04 | 3.90 [1.84, 6.69] | 91.2 [78.7, 104] | -63.8 to 246 |
| QE1 floor fixed | 2.487e-04 | 1.10 [0.89, 1.32] | 5.68 [-0.891, 12.3] | -75.3 to 86.6 |
| QE2 floor fixed | 2.580e-04 | 1.10 [0.92, 1.31] | 7.89 [1.03, 14.7] | -76.5 to 92.3 |
| IBM calibration (sx gate error) | 2.311e-04 | 1 | – | – |

## Table H5

Δω outside Canary's design range. Canary's Ramsey probe resolves |Δω| < 0.9π/τ_min (τ_min: its shortest Ramsey delay, set by the T2 prior); beyond it the estimate aliases by multiples of 1/τ_min or stops at the limit. QE's Ramsey (delays 1–49 μs with a 100 kHz offset) resolves about ±150 kHz.

| Quantity | Value |
|---|---|
| Canary unambiguous range \|Δω/2π\| (median, min–max over qubits) | 28,079 Hz (6,926–295,276 Hz) |
| Qubits whose QE1 detuning lies beyond Canary's range | 14/149 (9%) |
| of these, Canary's value off by more than 3 kHz | 13 |
| Canary values at the range limit (any check) | 17 |
| GREEN reports on these qubits (C1–C3); of them off by more than 3 kHz from the nearest QE run | 6/42; 5 (21.1–47.2 kHz) |
| Median \|Canary − QE1\| within range / beyond range | 312 Hz / 32,805 Hz |

## Table H6

Cost of each run on 154 qubits. QPU: IBM's measured usage. IBM-side: merged created-to-finished time of the run's jobs (queue included); classical = wall − IBM-side (circuit generation, transpilation, upload, download, analysis on one laptop). Queue time depends on IBM's load and is not a property of the method.

| Run | Shots per qubit | Jobs | QPU (s) | Classical (s) | IBM-side (s) | Queue (s) | Wall (s) | CPU (s) |
|---|---|---|---|---|---|---|---|---|
| C1 | 8,640 | 1 | 15.0 | 284 | 248 | 1.8 | 531 | 278 |
| QE1 | 92,000 | 4 | 63.0 | 589 | 146 | 5.7 | 735 | 578 |
| C2 | 8,640 | 1 | 15.0 | 287 | 255 | 1.8 | 542 | 280 |
| QE2 | 92,000 | 4 | 63.0 | 643 | 156 | 6.0 | 799 | 625 |
| C3 | 8,640 | 1 | 15.0 | 268 | 904 | 237 | 1,171 | 276 |

## Table H7

Drift on ibm_kingston over the sandwich (Fig. 7). Changed: Canary's three checks differ beyond their error bars (Cochran's Q, 2 dof, on the error bars canary.compare uses; Holm over every qubit and parameter). Spearman ρ: per qubit, QE's change QE2 − QE1 against Canary's change over the same interval (its trajectory interpolated linearly to the times of QE's jobs); 95% CI from Fieller's standard error, Holm over the four parameters. Disagreement: (Canary change − QE change) / combined standard error; SD 1 and 95% within ±2 if both tools see the same change and their error bars are right. QE: floor-fixed refits for T1 and ε; Δω only where both QE values lie inside Canary's range.

| Parameter | n | Changed over C1–C3 (Canary) | Spearman ρ [95% CI] | p (Holm) | Disagreement SD (robust) | \|z\| ≤ 2 (%) |
|---|---|---|---|---|---|---|
| T1 | 154 | 24/154 | +0.26 [+0.10, +0.40] | 3.4e-03 | 2.22 (2.06) | 72 |
| T2 | 153 | 14/153 | +0.26 [+0.10, +0.41] | 3.4e-03 | 1.06 (1.08) | 93 |
| Δω/2π | 133 | 68/133 | +0.50 [+0.36, +0.62] | 3.0e-09 | 5.11 (2.33) | 61 |
| ε | 153 | 3/153 | +0.14 [-0.02, +0.30] | 7.9e-02 | 1.11 (0.90) | 95 |

## Table H8

Do Canary's alarms hold up? Each tool's own change over the same interval (QE1 to QE2; Canary interpolated to QE's job times, as in Table H7) is flagged when it exceeds its standard error (two-sided z, Holm over every qubit and parameter, per tool). A Canary flag QE could not see (a change outside QE's window) is not counted here. Chance: the share QE flags among all qubits. QE is not ground truth: its error bars are as reported.

| Parameter | n | Flagged: Canary / QE / both | QE also flagged, of Canary's flags [95% CI] | Chance (QE's flag rate) | Same direction | Fisher p (one-sided) |
|---|---|---|---|---|---|---|
| T1 | 154 | 8 / 39 / 7 | 7/8 (88%) [47, 100] | 25% | 7/7 | 2.8e-04 |
| T2 | 153 | 5 / 7 / 4 | 4/5 (80%) [28, 99] | 5% | 4/4 | 7.8e-06 |
| Δω/2π | 133 | 38 / 55 / 25 | 25/38 (66%) [49, 80] | 41% | 23/25 | 3.2e-04 |
| ε | 153 | 2 / 2 / 2 | 2/2 (100%) [16, 100] | 1% | 2/2 | 8.6e-05 |

## Table H9

Canary's health verdicts on the 154 qubits of ibm_kingston, and why qubits were YELLOW: a probe's χ²/dof over its limit, T2* and T2 differing by more than 20% of T2 (slow frequency noise), or both. Last column: YELLOW qubits whose probe was over its limit, per probe.

| Check | GREEN | YELLOW | RED | YELLOW: fit over limit only | YELLOW: T2*/T2 rule only | YELLOW: both | Probes over limit (T1 / Ramsey / echo / RB) |
|---|---|---|---|---|---|---|---|
| C1 | 54 | 99 | 1 | 21 | 41 | 37 | 19 / 39 / 9 / 1 |
| C2 | 65 | 88 | 1 | 7 | 36 | 45 | 17 / 33 / 11 / 3 |
| C3 | 57 | 96 | 1 | 20 | 30 | 46 | 20 / 45 / 4 / 5 |
