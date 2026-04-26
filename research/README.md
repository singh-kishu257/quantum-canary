# Research Pipeline

This folder contains the numbered scripts that reproduce all results from the Quantum Canary paper. Each script writes its outputs to a known location that the next script consumes — running them in numerical order reproduces every figure, baseline, and metric reported in the paper.

---

## Pipeline overview

| Script                     | Purpose                                                                               | Inputs                           | Outputs                                                        |
| -------------------------- | ------------------------------------------------------------------------------------- | -------------------------------- | -------------------------------------------------------------- |
| `1_pull_data.py`           | Pull real IBM calibration data from `ibm_fez`, `ibm_kingston`, `ibm_marrakesh`        | IBM Quantum credentials          | `data/ibm_*_calibration.csv`, `data/all_backends_raw.csv`      |
| `1b_pull_data.py`          | Generate physics-grounded synthetic data via Qiskit Aer                               | None                             | `data/sim_data.csv` (15k rows, 3 regimes)                      |
| `2_label_data.py`          | Merge synthetic + IBM anchors, apply 15% T1-drop labels, create train/val/test splits | All `1_*` outputs                | `data/features_data.csv` (21k rows), `data/split_indices.json` |
| `3_validate_data.py`       | Compute correlation matrix, plot feature distributions and class overlap              | `data/features_data.csv`         | Diagnostic figures                                             |
| `4_standard_baseline.py`   | Per-feature majority-vote threshold (naive baseline)                                  | `data/features_data.csv`, splits | `results/baseline_standard_results.json`                       |
| `4b_hotelling_baseline.py` | Hotelling's T² multivariate statistical baseline (Hotelling 1947)                     | same                             | `models/hotelling_params.json`                                 |
| `5_train_mlp.py`           | Train 10-seed MLP ensemble                                                            | same                             | `models/mlp_seed_{0..9}.keras`, scaler                         |
| `6_evaluate.py`            | Three-way comparison (threshold, T², MLP) and final figures                           | all of above                     | `results/evaluation_results.json`, paper figures               |
| `7_realtime_canary.py`     | Live deployment on IBM hardware                                                       | trained models + IBM credentials | `logs/realtime_log.csv`, live figure                           |

---

## Key design decisions

### Hybrid synthetic + real training set

Real IBM calibration data alone is insufficient for training a drift detector because IBM publishes data only at the moment of recalibration — when hardware is at peak health. A model trained exclusively on this data would never encounter drifted samples.

Quantum Canary's hybrid approach pairs **15,000 physics-grounded Qiskit Aer simulations** (parameterised by published Heron r2 fleet statistics and TLS-fluctuation literature) with **6,000 real IBM calibration anchors** that ground the simulation in observed hardware behaviour.

### Three-regime synthetic generator

`1b_pull_data.py` produces samples from three distinct physical regimes:

- **Stable** (T1 ≈ 175 µs, gate errors near published Heron r2 medians) — fresh post-calibration hardware
- **Borderline** (T1 ≈ 140 µs, 20% drop) — crosses the 15% labelling threshold (so labelled drifted=1) but overlaps the stable distribution. Forces the model to learn a non-trivial decision boundary.
- **Drifted** (T1 ≈ 90 µs, 49% drop, gate errors elevated 4-5×) — observed TLS-driven degradation per Carroll et al. 2022

Each batch resamples its noise model, mimicking real qubit-to-qubit and time-to-time variation across the IBM fleet.

### Three independent baselines

The paper compares the MLP against three increasingly sophisticated baselines:

1. **Standard threshold** — per-feature majority vote against training-set means
2. **Hotelling's T²** — multivariate statistical process control (Hotelling 1947)
3. **MLP ensemble** — this work

Both statistical baselines plateau near AUC 0.76, while the MLP achieves 0.92. **The plateau is the central finding**: linear methods cannot resolve the borderline regime regardless of how rigorously they handle multivariate structure. The MLP wins through learned nonlinearity.

---

## Running the pipeline

Requires Python 3.10+, the dependencies in the root `requirements.txt`, and (for the live monitor) an IBM Quantum account.

```bash
# 1. Set up environment from the repo root
pip install -r requirements.txt

# 2. Move into research/ — all scripts use paths relative to this folder
cd research

# 3. Pull IBM calibration data (requires credentials)
python 1_pull_data.py

# 4. Generate synthetic data (~10 minutes on Aer)
python 1b_pull_data.py

# 5. Merge and label
python 2_label_data.py

# 6. (Optional) Inspect data quality and feature correlations
python 3_validate_data.py

# 7. Run baselines
python 4_standard_baseline.py
python 4b_hotelling_baseline.py

# 8. Train MLP ensemble (~2 minutes on CPU)
python 5_train_mlp.py

# 9. Evaluate all three methods together
python 6_evaluate.py

# 10. Live deployment (requires IBM credentials)
echo "YOUR_TOKEN" > api_token.txt
echo "YOUR_CRN"   > crn_instance.txt
python 7_realtime_canary.py
```

---

## Reproducibility notes

- All scripts seed their random generators (numpy seed 42 in data generation; per-seed control in MLP training).
- The 21,000-row training CSV is regenerable from `1_pull_data.py`, `1b_pull_data.py`, and `2_label_data.py` and is therefore not committed to the repository — only the scripts that produce it.
- Live deployment results (`results/ibm_kingston_log.csv`) cannot be reproduced exactly because they reflect a specific 24-hour window on IBM hardware. The committed log is the original deployment capture and should be treated as primary data.

---

## Citations

Source literature referenced by these scripts:

- Carroll, A. et al. (2022). _npj Quantum Information_ 8, 132. — TLS-driven T1 fluctuations on IBM hardware.
- Hotelling, H. (1947). _Multivariate Quality Control_. — T² baseline.
- Krantz, P. et al. (2019). _Applied Physics Reviews_ 6, 021318. — superconducting qubit noise model.
- Preskill, J. (2018). _Quantum_ 2, 79. — NISQ era and the calibration gap.

Full bibliography in the paper's references section.
