# NISQ benchmark tables (research/data/nisq_benchmark.csv)

## Table I

Median absolute error [95% CI] against the simulator's truth, 500 simulated qubits per platform (relative error for T₁, T₂, ε; absolute error of Δω/2π). Circuit time: median delays plus gate time per check, a lower bound (no repetition delay or readout). Failed fits: estimates QE did not return, excluded from the medians.

| Platform | Method | Shots | Circuit time (s) | T₁ (%) | T₂ (%) | Δω/2π (Hz) | ε (%) | Failed fits |
|---|---|---|---|---|---|---|---|---|
| SC | Canary | 8,640 | 0.557 | 3.13 [2.81, 3.37] | 5.33 [4.76, 5.74] | 190 [159, 213] | 7.95 [6.85, 9.05] | 0/2000 |
| SC | QE, equal shots | 8,640 | 0.430 | 9.90 [8.56, 11.8] | 10.3 [9.38, 11.4] | 264 [230, 297] | 53.4 [50.8, 55.6] | 3/2000 |
| SC | QE standard | 92,000 | 2.99 | 3.84 [3.38, 4.41] | 5.02 [4.67, 5.55] | 110 [93.5, 123] | 18.7 [17.0, 20.8] | 0/2000 |
| TI | Canary | 8,640 | 42,700 | 2.94 [2.65, 3.30] | 5.20 [4.56, 5.75] | 0.0313 [0.0268, 0.0340] | 7.40 [6.68, 7.95] | 0/2000 |
| TI | QE, equal shots | 8,640 | 37,200 | 6.61 [6.03, 7.46] | 14.2 [12.9, 15.8] | 0.114 [0.103, 0.125] | 54.6 [53.2, 57.0] | 35/2000 |
| TI | QE standard | 92,000 | 251,000 | 2.82 [2.64, 3.03] | 6.85 [6.05, 7.53] | 0.0806 [0.0692, 0.0934] | 29.6 [24.9, 33.8] | 6/2000 |

## Table S1

Accuracy and error-bar calibration per arm (successful fits only; n of 500). Coverage: % of qubits with |z| ≤ 1 and |z| ≤ 2, z = (estimate − truth)/σ.

| Platform | Method | Parameter | n | Median | 90th pct. | Median signed [95% CI] | Cov. 68.3 | Cov. 95.4 | sd(z) |
|---|---|---|---|---|---|---|---|---|---|
| SC | Canary | T₁ (%) | 500 | 3.13 | 7.88 | -0.639 [-1.24, -0.167] | 68.4 | 95.6 | 1.02 |
| SC | Canary | T₂ (%) | 500 | 5.33 | 12.5 | -1.51 [-2.31, -0.499] | 70.0 | 94.8 | 0.957 |
| SC | Canary | Δω/2π (Hz) | 500 | 190 | 634 | -21.5 [-47.7, 10.2] | 69.6 | 93.6 | 1.01 |
| SC | Canary | ε (%) | 500 | 7.95 | 19.6 | -0.320 [-0.881, 1.01] | 62.8 | 93.6 | 1.09 |
| SC | Canary (repeat) | T₁ (%) | 500 | 3.41 | 7.84 | -0.256 [-0.792, 0.396] | 66.6 | 94.8 | 1.03 |
| SC | Canary (repeat) | T₂ (%) | 500 | 5.27 | 12.3 | -1.10 [-1.89, -0.446] | 70.6 | 96.0 | 0.974 |
| SC | Canary (repeat) | Δω/2π (Hz) | 500 | 209 | 698 | -1.29 [-34.8, 27.4] | 66.0 | 95.0 | 1.08 |
| SC | Canary (repeat) | ε (%) | 500 | 7.09 | 19.4 | -0.0961 [-1.38, 1.28] | 67.0 | 95.2 | 1.03 |
| SC | QE, equal shots | T₁ (%) | 498 | 9.90 | 37.6 | 0.957 [-0.169, 2.23] | 75.5 | 94.4 | 0.987 |
| SC | QE, equal shots | T₂ (%) | 500 | 10.3 | 26.2 | -0.0960 [-1.66, 0.943] | 68.6 | 96.8 | 0.967 |
| SC | QE, equal shots | Δω/2π (Hz) | 500 | 264 | 787 | -70.6 [-121, -42.4] | 67.0 | 91.8 | 25.2 |
| SC | QE, equal shots | ε (%) | 499 | 53.4 | 153 | 1.88 [-8.41, 10.8] | 80.0 | 97.6 | 0.802 |
| SC | QE standard | T₁ (%) | 500 | 3.84 | 15.0 | 0.775 [0.150, 1.40] | 72.2 | 96.8 | 0.934 |
| SC | QE standard | T₂ (%) | 500 | 5.02 | 13.7 | 0.772 [-0.240, 1.99] | 68.8 | 95.4 | 0.995 |
| SC | QE standard | Δω/2π (Hz) | 500 | 110 | 381 | -78.1 [-97.0, -60.1] | 54.6 | 85.0 | 1.22 |
| SC | QE standard | ε (%) | 500 | 18.7 | 56.9 | -0.873 [-3.27, 1.98] | 65.2 | 94.4 | 1.05 |
| TI | Canary | T₁ (%) | 500 | 2.94 | 7.45 | -0.226 [-0.869, 0.208] | 67.4 | 94.8 | 1.02 |
| TI | Canary | T₂ (%) | 500 | 5.20 | 10.7 | -0.335 [-1.27, 0.434] | 73.2 | 97.8 | 0.879 |
| TI | Canary | Δω/2π (Hz) | 500 | 0.0313 | 0.0906 | -0.00212 [-0.00708, 0.00298] | 57.4 | 81.8 | 2.08 |
| TI | Canary | ε (%) | 500 | 7.40 | 18.8 | 0.902 [-0.260, 2.37] | 65.6 | 94.6 | 1.11 |
| TI | Canary (repeat) | T₁ (%) | 500 | 2.96 | 7.36 | -0.189 [-0.652, 0.351] | 69.0 | 94.6 | 0.986 |
| TI | Canary (repeat) | T₂ (%) | 500 | 4.84 | 12.3 | -0.414 [-1.10, 0.234] | 71.4 | 96.0 | 0.945 |
| TI | Canary (repeat) | Δω/2π (Hz) | 500 | 0.0328 | 0.104 | -0.000664 [-0.00630, 0.00511] | 49.4 | 82.6 | 2.48 |
| TI | Canary (repeat) | ε (%) | 500 | 7.40 | 19.5 | -1.16 [-2.20, -0.253] | 62.4 | 91.6 | 1.15 |
| TI | QE, equal shots | T₁ (%) | 500 | 6.61 | 17.9 | 0.262 [-0.597, 1.28] | 73.8 | 97.0 | 0.903 |
| TI | QE, equal shots | T₂ (%) | 500 | 14.2 | 43.9 | 2.31 [0.179, 5.27] | 77.2 | 96.6 | 0.938 |
| TI | QE, equal shots | Δω/2π (Hz) | 485 | 0.114 | 0.272 | 0.00877 [-0.0115, 0.0280] | 33.8 | 53.4 | 70.7 |
| TI | QE, equal shots | ε (%) | 480 | 54.6 | 299 | -4.23 [-16.5, 9.40] | 77.5 | 95.2 | 0.869 |
| TI | QE standard | T₁ (%) | 500 | 2.82 | 6.64 | -0.276 [-0.743, 0.352] | 73.2 | 95.8 | 0.959 |
| TI | QE standard | T₂ (%) | 500 | 6.85 | 18.2 | 0.132 [-1.07, 1.73] | 71.0 | 96.2 | 0.953 |
| TI | QE standard | Δω/2π (Hz) | 494 | 0.0806 | 0.232 | 0.00230 [-0.0127, 0.0158] | 15.4 | 28.3 | 586 |
| TI | QE standard | ε (%) | 500 | 29.6 | 89.2 | -2.67 [-6.58, 0.991] | 75.2 | 96.8 | 0.909 |

## Table S2

False alarms of `compare()` between two Canary checks of the same unchanged qubit (independent RB sequences and shot noise): % of qubits flagged [95% CI], with χ²-inflated σ and Holm correction over the four parameters at family level 4.55%.

| Platform | n | T₁ | T₂ | Δω | ε | Any parameter |
|---|---|---|---|---|---|---|
| SC | 500 | 0.8 [0.2, 2.0] | 0.8 [0.2, 2.0] | 0.2 [0.0, 1.1] | 1.2 [0.4, 2.6] | 2.8 [1.5, 4.7] |
| TI | 500 | 1.2 [0.4, 2.6] | 0.4 [0.0, 1.4] | 1.2 [0.4, 2.6] | 1.6 [0.7, 3.1] | 4.2 [2.6, 6.3] |

## Table S3

Robustness: median absolute error of Canary / QE standard in thirds of the coherent share of the gate error and of the calibration prior's relative error (about 167 qubits each).

| Platform | Factor | Third | T₁ (%) | T₂ (%) | Δω/2π (Hz) | ε (%) |
|---|---|---|---|---|---|---|
| SC | Coherent fraction | 0.00125–0.110 | 2.77 / 3.46 | 5.55 / 4.67 | 169 / 111 | 8.79 / 15.6 |
| SC | Coherent fraction | 0.110–0.198 | 2.88 / 4.11 | 5.16 / 5.55 | 177 / 98.9 | 7.47 / 19.8 |
| SC | Coherent fraction | 0.198–0.299 | 3.83 / 4.00 | 5.32 / 4.84 | 206 / 115 | 7.30 / 21.6 |
| SC | Prior error | 0.00206–0.171 | 3.22 / 4.14 | 4.90 / 4.56 | 201 / 98.0 | 7.42 / 18.0 |
| SC | Prior error | 0.171–0.330 | 3.06 / 3.53 | 5.01 / 5.27 | 207 / 106 | 8.08 / 20.6 |
| SC | Prior error | 0.330–0.499 | 2.97 / 3.73 | 6.12 / 5.55 | 152 / 117 | 8.36 / 18.4 |
| TI | Coherent fraction | 0.000133–0.105 | 2.59 / 3.09 | 5.09 / 7.79 | 0.0293 / 0.0808 | 7.45 / 19.5 |
| TI | Coherent fraction | 0.105–0.201 | 3.12 / 2.67 | 5.77 / 6.34 | 0.0312 / 0.0809 | 7.58 / 30.5 |
| TI | Coherent fraction | 0.201–0.300 | 3.01 / 2.85 | 5.11 / 6.46 | 0.0344 / 0.0804 | 6.99 / 35.1 |
| TI | Prior error | 0.00248–0.156 | 3.25 / 2.81 | 5.11 / 5.75 | 0.0315 / 0.0797 | 7.18 / 27.7 |
| TI | Prior error | 0.156–0.327 | 2.95 / 2.93 | 5.17 / 6.38 | 0.0298 / 0.0898 | 7.46 / 27.4 |
| TI | Prior error | 0.327–0.500 | 2.76 / 2.52 | 5.41 / 7.83 | 0.0313 / 0.0794 | 7.51 / 32.9 |

## Table S4

Canary's model checks in the NISQ regime: % of qubits over each probe's χ²/dof limit (nominal 1.25% under a correct model), % per health status, and % of qubits whose simulated T₂*/T₂ is below 0.8, the YELLOW decoupling rule (not stored for trapped ions).

| Platform | Arm | T₁ χ² flag | T₂ χ² flag | Δω χ² flag | ε χ² flag | GREEN | YELLOW | RED | True T₂*/T₂ < 0.8 |
|---|---|---|---|---|---|---|---|---|---|
| SC | Canary | 1.8 | 1.2 | 24.0 | 2.2 | 15.8 | 84.2 | 0.0 | 93.4 |
| SC | Canary (repeat) | 0.6 | 1.4 | 25.4 | 2.4 | 17.0 | 83.0 | 0.0 | 93.4 |
| TI | Canary | 1.0 | 1.4 | 25.0 | 1.8 | 52.0 | 48.0 | 0.0 | – |
| TI | Canary (repeat) | 1.2 | 1.6 | 25.4 | 2.8 | 52.0 | 48.0 | 0.0 | – |

## Table S5

Failed QE fits along the shot sweep (Canary: none at any budget).

| Platform | Shots | T₁ | T₂ | Δω | ε |
|---|---|---|---|---|---|
| SC | 1,079 | 5/200 | 1/200 | 11/200 | 34/200 |
| SC | 2,141 | 3/200 | 0/200 | 4/200 | 8/200 |
| SC | 4,324 | 3/200 | 0/200 | 0/200 | 2/200 |
| SC | 8,648 | 2/200 | 0/200 | 0/200 | 0/200 |
| SC | 17,296 | 0/200 | 0/200 | 0/200 | 0/200 |
| SC | 34,567 | 0/200 | 0/200 | 0/200 | 0/200 |
| SC | 69,117 | 0/200 | 0/200 | 0/200 | 0/200 |
| SC | 138,251 | 0/200 | 0/200 | 0/200 | 0/200 |
| TI | 1,079 | 37/200 | 0/200 | 21/200 | 71/200 |
| TI | 2,141 | 15/200 | 0/200 | 13/200 | 24/200 |
| TI | 4,324 | 1/200 | 0/200 | 7/200 | 5/200 |
| TI | 8,648 | 2/200 | 0/200 | 2/200 | 3/200 |
| TI | 17,296 | 0/200 | 0/200 | 3/200 | 0/200 |
| TI | 34,567 | 0/200 | 0/200 | 1/200 | 0/200 |
| TI | 69,117 | 0/200 | 0/200 | 4/200 | 0/200 |
| TI | 138,251 | 0/200 | 0/200 | 1/200 | 0/200 |
