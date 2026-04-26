# Quantum Canary: Neural-Network-Based Autonomous Noise Drift Detection in NISQ-Era Quantum Processors

**Kanishka Singh**
_Urbana High School_
*kanishka.singh.ums@gmail.com*

---

## Abstract

Quantum computers exploit superposition, entanglement, and coherence to solve problems intractable for classical hardware. In the Noisy Intermediate-Scale Quantum (NISQ) era, qubit performance degrades continuously due to environmental interference — thermal fluctuations, electromagnetic coupling, and two-level-system (TLS) defects — which disrupts fidelity between calibration cycles. Cloud-based quantum providers refresh published qubit health data on a daily cycle, leaving a multi-hour window in which drift goes undetected and the resulting computations may be silently corrupted. This work introduces **Quantum Canary**, an autonomous monitoring system combining three lightweight quantum probe circuits with a 10-seed multilayer perceptron (MLP) ensemble to classify hardware drift in real time. The MLP is trained on a 21,000-sample physics-grounded hybrid dataset combining Qiskit Aer noise simulations with real IBM calibration anchors. On held-out test data Quantum Canary achieves an AUC of $0.9239$ — a $21.6\%$ improvement over Hotelling's $T^2$ ($\text{AUC}=0.7674$), the strongest non-ML statistical baseline. In a 24-hour live deployment on `ibm_kingston`, the system correctly anticipated **3 of 3 IBM recalibration events** with an average lead time of **3.34 hours**.

---

## 1. Introduction

### 1.1 The NISQ regime and the calibration-data gap

Preskill (2018) coined **NISQ** — Noisy Intermediate-Scale Quantum — to describe the current era of 50-1000-qubit systems without full error correction. NISQ qubits are extraordinarily fragile: superconducting transmons must be cooled below 20 mK, isolated from electromagnetic noise, and shielded from cosmic radiation. Even under these conditions, qubit parameters drift continuously.

Cloud quantum providers — IBM Quantum, Google Quantum AI, AWS Braket — publish calibration data on a roughly daily cycle. Between snapshots, **the user has no visibility into the actual state of the hardware they are paying for**. A computation that begins on a freshly calibrated qubit and finishes hours later may have executed on hardware that has degraded by 30-50% of its $T_1$ relaxation time (Carroll et al. 2022). The user has no way to know.

This is the **calibration-data gap**, and it is the problem Quantum Canary is built to solve.

### 1.2 Contributions

1. A three-circuit canary probe protocol that measures three orthogonal noise channels in approximately one second of total backend time.
2. A 10-seed MLP ensemble trained on a 21,000-sample hybrid dataset spanning physics-grounded simulations and real IBM calibration anchors.
3. Two rigorous statistical baselines (per-feature majority-vote threshold; Hotelling's $T^2$) demonstrating that linear methods plateau at $\text{AUC} \approx 0.76$, while the MLP achieves $0.9239$.
4. A 24-hour live deployment on `ibm_kingston` showing 3 of 3 real IBM recalibration events anticipated with an average lead time of 3.34 hours.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│            QUANTUM CANARY — END-TO-END PIPELINE             │
└─────────────────────────────────────────────────────────────┘

                    ┌────────────────────┐
                    │   IBM QUANTUM      │
                    │   (live backend)   │
                    └─────────┬──────────┘
                              │ submits 3 circuits
                              ▼
       ┌──────────────────────────────────────────┐
       │  CANARY CIRCUITS                          │
       │                                           │
       │   ┌───────────┐   ┌───────────┐   ┌──────┐│
       │   │ Bell state│   │Coherence  │   │ Gate ││
       │   │ (2-qubit) │   │ 20× H·H=I │   │ 20× X││
       │   └─────┬─────┘   └─────┬─────┘   └──┬───┘│
       │         │               │             │    │
       │         ▼               ▼             ▼    │
       │      F_bell        F_coherence     F_gate  │
       └─────────┬──────────────┬─────────────┬─────┘
                 │              │             │
                 └──────┬───────┴──────┬──────┘
                        ▼              ▼
                  ┌─────────────────────────────┐
                  │  StandardScaler             │
                  │  μ, σ from training set     │
                  └─────────────┬───────────────┘
                                │
                                ▼
       ┌─────────────────────────────────────────────┐
       │  MLP ENSEMBLE (10 seeded models)            │
       │                                             │
       │   3 → 64 → BN → Dropout(0.3) →              │
       │       32 → BN → Dropout(0.3) →              │
       │       16 → 1 (sigmoid)                      │
       └─────────────┬───────────────────────────────┘
                     │
                     ▼
              ┌──────────────────────┐
              │  Ensemble aggregation│
              │   mean = drift prob  │
              │   std  = uncertainty │
              └──────────┬───────────┘
                         │
                         ▼
                   ┌───────────────┐
                   │   VERDICT     │
                   │  STABLE /     │
                   │  DRIFTED      │
                   └───────────────┘
```

The three canary circuits are submitted in a single batched job to minimise queue overhead. Each circuit runs with 1000 shots. The full probe completes in approximately one second of backend time.

---

## 3. Canary Circuits

Quantum Canary uses three minimal circuits, each engineered to be sensitive to one specific noise channel.

### 3.1 Bell state circuit — 2-qubit entanglement probe

```
   q_0 ──── H ────●────────── M ───────►
                  │
   q_1 ───────── ⊕ ────────── M ───────►
```

A Hadamard on qubit 0 followed by a CNOT entangles the two qubits into the Bell state:

$$|\psi\rangle_{\text{ideal}} = \frac{1}{\sqrt{2}}\bigl(|00\rangle + |11\rangle\bigr)$$

On a perfect device, measurement yields exactly 50% $|00\rangle$ and 50% $|11\rangle$. Any $|01\rangle$ or $|10\rangle$ outcome is a noise event — typically caused by 2-qubit gate (CZ/ECR) error or asymmetric readout. The fidelity is extracted as the fraction of correct outcomes:

$$F_{\text{bell}} = \frac{N_{00} + N_{11}}{N_{\text{total}}}$$

This circuit is sensitive primarily to **two-qubit gate error** and **readout asymmetry**.

### 3.2 Coherence circuit — $T_2$ dephasing probe

```
   q_0 ──── H H H H H H H H H H ── ... ── H H ── M ───►
            └─── 10 H·H pairs (= identity) ────┘
```

Twenty Hadamard gates, applied as ten $H \cdot H$ pairs, form a logical identity since $H \cdot H = I$. The ideal output is $|0\rangle$ with 100% probability. On real Heron r2 hardware, each Hadamard takes approximately 56 ns, so the total runtime is:

$$t_{\text{coh}} = 20 \times t_{\text{sx}} \approx 1120~\text{ns}$$

During this 1120 ns window, the qubit spends substantial time in superposition states on the equator of the Bloch sphere — exactly where $T_2$ dephasing has maximum effect. Measured deviations from $|0\rangle$ therefore primarily reflect accumulated **phase decoherence**:

$$F_{\text{coherence}} = \frac{N_0}{N_{\text{total}}}$$

### 3.3 Gate-error circuit — single-qubit gate accumulation probe

```
   q_0 ──── X X X X X X X X X X ── ... ── X X ── M ───►
            └────── 20 X gates (= identity) ────┘
```

Twenty X gates compose to identity. Per-gate error accumulates exponentially:

$$F_{\text{gate}} \approx (1 - \epsilon_{\text{sx}})^{20}$$

For a healthy qubit with $\epsilon_{\text{sx}} \approx 3.5 \times 10^{-4}$, the expected fidelity is approximately $0.993$. For a drifted qubit with $\epsilon_{\text{sx}} \approx 2.8 \times 10^{-3}$, fidelity drops to approximately $0.945$. The 20-gate amplification turns sub-millipercent per-gate errors into easily measurable signals:

$$F_{\text{gate}} = \frac{N_0}{N_{\text{total}}}$$

### 3.4 Channel orthogonality

The three probes are deliberately orthogonal in the noise channels they measure:

| Channel                 | Bell     | Coherence | Gate     |
| ----------------------- | -------- | --------- | -------- |
| Two-qubit gate error    | dominant | —         | —        |
| $T_2$ dephasing         | partial  | dominant  | partial  |
| Single-qubit gate error | partial  | —         | dominant |
| Readout asymmetry       | yes      | partial   | partial  |

The MLP learns to weigh all three jointly to detect drift signatures that no individual probe can resolve.

---

## 4. Analytical Fidelity Approximations

For data labelling and synthetic generation, we use closed-form physics-based approximations of the three fidelities. These follow standard depolarizing-plus-thermal-relaxation noise models (Krantz et al. 2019; Bravo-Montes et al. 2024).

### 4.1 Bell state

$$F_{\text{bell}} \approx (1 - \epsilon_{\text{sx}})\,(1 - \epsilon_{cz})\,(1 - \epsilon_{\text{ro}})^2 \, e^{-t_{\text{bell}}/T_1}\, e^{-t_{\text{bell}}/T_2}$$

where $t_{\text{bell}} = t_{\text{sx}} + t_{cz}$ is the Bell-circuit duration, $\epsilon_{\text{sx}}$ is the single-qubit gate error, $\epsilon_{cz}$ is the two-qubit gate error, and $\epsilon_{\text{ro}}$ is the readout error.

### 4.2 Coherence circuit

For 20 H gates ($n=20$, total runtime $t_{\text{coh}} = 20\,t_{\text{sx}}$):

$$F_{\text{coherence}} \approx (1 - \epsilon_{\text{sx}})^{20} \,(1 - \epsilon_{\text{ro}}) \, \tfrac{1}{2}\left(1 + e^{-t_{\text{coh}}/T_1}\, e^{-t_{\text{coh}}/T_2}\right)$$

The factor $\tfrac{1}{2}(1 + e^{-t/T_1}e^{-t/T_2})$ captures the probability of recovering $|0\rangle$ after passing through superposition states subject to dephasing.

### 4.3 Gate-error circuit

For 20 X gates ($t_{\text{gate}} = 20\,t_{x}$):

$$F_{\text{gate}} \approx (1 - \epsilon_{x})^{20} \,(1 - \epsilon_{\text{ro}}) \, e^{-t_{\text{gate}}/T_1}\, e^{-t_{\text{gate}}/T_2}$$

These three formulas provide the link between IBM's published calibration parameters and the canary fidelities — allowing real IBM calibration snapshots to be converted into fidelity triplets for the hybrid training set.

---

## 5. Training Dataset

### 5.1 Hybrid construction

A drift detector cannot be trained on real IBM calibration data alone. IBM publishes calibration data **only at the moment of recalibration** — when hardware is at peak health. A model trained exclusively on this data would never see drifted samples.

Quantum Canary uses a hybrid construction:

```
┌──────────────────────────────────────────────────────────┐
│       21,000-SAMPLE HYBRID TRAINING DATASET              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   15,000 SYNTHETIC                  6,000 IBM ANCHORS    │
│   (Qiskit Aer simulation)           (real calibration    │
│   ┌──────────────────┐               published data)     │
│   │  5,000 stable   │               ┌──────────────────┐ │
│   ├──────────────────┤               │  3,000 stable   │ │
│   │ 5,000 borderline│               │   anchors        │ │
│   ├──────────────────┤               ├──────────────────┤ │
│   │  5,000 drifted  │               │  3,000 drifted  │ │
│   └──────────────────┘               │   anchors        │ │
│                                     └──────────────────┘ │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 5.2 Three-regime synthetic generator

Synthetic samples are generated by Qiskit Aer with noise models drawn from three physically motivated regimes:

| Regime         | $T_1$                         | $\epsilon_{\text{sx}}$    | $\epsilon_{cz}$           | Label |
| -------------- | ----------------------------- | ------------------------- | ------------------------- | ----- |
| **Stable**     | $\sim 175~\mu\text{s}$        | $\sim 3.5 \times 10^{-4}$ | $\sim 3.8 \times 10^{-3}$ | 0     |
| **Borderline** | $\sim 140~\mu\text{s}$ (-20%) | $\sim 8.0 \times 10^{-4}$ | $\sim 7.5 \times 10^{-3}$ | 1     |
| **Drifted**    | $\sim 90~\mu\text{s}$ (-49%)  | $\sim 2.8 \times 10^{-3}$ | $\sim 1.6 \times 10^{-2}$ | 1     |

The borderline regime is the key. It crosses the 15% $T_1$-drop labelling threshold, but its fidelity values overlap the stable distribution by design — forcing the model to learn a non-trivial decision boundary.

### 5.3 Real IBM anchors

Six thousand additional rows are extracted from the published calibration histories of `ibm_fez`, `ibm_kingston`, and `ibm_marrakesh`. They are converted to fidelity triplets via the analytical formulas in Section 4.

### 5.4 Labelling rule

A sample is labelled drifted ($y=1$) if **any** of the following hold relative to the qubit's 30-day rolling baseline:

$$\Delta T_1 \leq -15\%, \quad \Delta\epsilon_{cz} \geq +50\%, \quad \Delta\epsilon_{\text{ro}} \geq +30\%$$

The 15% $T_1$ threshold aligns with the borderline regime and matches the smallest drift level Carroll et al. (2022) document as having measurable circuit-fidelity impact.

---

## 6. The MLP Ensemble

### 6.1 Architecture

```
   Input: [F_bell, F_gate, F_coherence]                 (3 features)
       │
       ▼
   Dense(64) → BatchNorm → ReLU → Dropout(0.3)
       │
       ▼
   Dense(32) → BatchNorm → ReLU → Dropout(0.3)
       │
       ▼
   Dense(16) → ReLU
       │
       ▼
   Dense(1) → Sigmoid                       (drift probability ∈ [0,1])
```

### 6.2 Forward pass

For input $\mathbf{x} \in \mathbb{R}^3$, scaled by $\mathbf{z} = (\mathbf{x} - \boldsymbol{\mu})/\boldsymbol{\sigma}$, the network computes:

$$\mathbf{h}_1 = \text{Dropout}\bigl(\text{BN}(\text{ReLU}(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1))\bigr)$$
$$\mathbf{h}_2 = \text{Dropout}\bigl(\text{BN}(\text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2))\bigr)$$
$$\mathbf{h}_3 = \text{ReLU}(\mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3)$$
$$\hat{p} = \sigma(\mathbf{w}_4^{\top} \mathbf{h}_3 + b_4)$$

with sigmoid $\sigma(z) = 1/(1+e^{-z})$.

### 6.3 Training objective — binary cross-entropy

Per sample with true label $y \in \{0,1\}$ and predicted probability $\hat{p} \in (0,1)$:

$$\mathcal{L}_{\text{BCE}}(y, \hat{p}) = -\bigl[y\log\hat{p} + (1-y)\log(1-\hat{p})\bigr]$$

The network minimises the empirical mean of $\mathcal{L}_{\text{BCE}}$ over each minibatch.

### 6.4 Optimiser — Adam

Adam (Kingma & Ba 2014) is a per-parameter adaptive method maintaining first- and second-moment estimates of the gradient $g_t$ at step $t$:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

with bias-corrected estimates $\hat{m}_t = m_t/(1-\beta_1^t)$ and $\hat{v}_t = v_t/(1-\beta_2^t)$. The parameter update is:

$$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Training uses the standard hyperparameters $\alpha=10^{-3}$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, batch size 256, 50 epochs with early stopping on validation loss.

### 6.5 Why an ensemble of 10

A single MLP is a point estimate. An ensemble of $K=10$ differently seeded models produces a _distribution_ of predictions. The ensemble drift probability and uncertainty are:

$$\bar{p}(\mathbf{x}) = \frac{1}{K}\sum_{k=1}^{K}\hat{p}_k(\mathbf{x}), \qquad \sigma_p(\mathbf{x}) = \sqrt{\frac{1}{K}\sum_{k=1}^{K}\bigl(\hat{p}_k(\mathbf{x}) - \bar{p}(\mathbf{x})\bigr)^2}$$

The verdict is `DRIFTED` if $\bar{p} \geq 0.5$, `STABLE` otherwise. The standard deviation $\sigma_p$ is logged as **ensemble uncertainty** — a measure of internal disagreement that often rises before the mean crosses the decision threshold.

---

## 7. Statistical Baselines

Quantum Canary is benchmarked against two non-ML statistical methods of increasing rigour.

### 7.1 Per-feature majority-vote threshold

The simplest possible classifier. For each feature $f \in \{F_{\text{bell}}, F_{\text{gate}}, F_{\text{coherence}}\}$, the threshold $\mu_f$ is the training-set mean. Each feature votes "drifted" if its observed value falls below $\mu_f$. The final prediction is the majority across the three votes:

$$\hat{y}_{\text{thr}}(\mathbf{x}) = \mathbb{1}\!\left[\sum_{f \in \{\text{bell, gate, coh}\}} \mathbb{1}[x_f < \mu_f] \geq 2\right]$$

This baseline preserves per-feature information but produces only four discrete output scores ($0, 1, 2, 3$ drift votes), so the resulting ROC curve has at most four points.

**Test AUC: $0.7586$**

### 7.2 Hotelling's $T^2$ (Hotelling 1947)

The rigorous multivariate baseline. The healthy training distribution is modelled as a multivariate Gaussian:

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\!\left[-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{\top}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right]$$

with $\boldsymbol{\mu}, \boldsymbol{\Sigma}$ estimated from stable training samples only. The drift score is the squared Mahalanobis distance from the healthy centroid:

$$T^2(\mathbf{x}) = (\mathbf{x} - \boldsymbol{\mu})^{\top} \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$$

A small ridge regularisation handles high feature correlation ($F_{\text{gate}}$ and $F_{\text{coherence}}$ correlate at $\sim 0.98$):

$$\boldsymbol{\Sigma}_{\text{reg}} = \boldsymbol{\Sigma} + \epsilon\, \mathbf{I}, \qquad \epsilon = 10^{-6}$$

The decision threshold $h$ is tuned on validation data to maximise F1, and predictions are:

$$\hat{y}_{T^2}(\mathbf{x}) = \mathbb{1}[T^2(\mathbf{x}) > h]$$

**Test AUC: $0.7674$**

### 7.3 Why both baselines plateau near 0.76

The two baselines agree to within $0.009$ AUC. This convergence is the central scientific finding of the baseline analysis: **the bottleneck for linear methods on this problem is not feature-correlation modelling, but linear separability itself**. The borderline regime is constructed precisely to overlap the stable distribution. No quadratic decision surface (i.e., no $T^2$ ellipsoid, regardless of how well-conditioned $\boldsymbol{\Sigma}^{-1}$ is) can resolve it. The MLP wins by learning a **nonlinear** boundary that no analytic statistical method can express.

---

## 8. Results

### 8.1 Held-out test set

| Method                              | Test AUC          | F1              | Improvement vs MLP |
| ----------------------------------- | ----------------- | --------------- | ------------------ |
| Per-feature majority-vote threshold | $0.7586$          | $0.71$          | —                  |
| Hotelling's $T^2$ (1947)            | $0.7674$          | $0.71$          | $+0.9\%$           |
| **Quantum Canary MLP ensemble**     | $\mathbf{0.9239}$ | $\mathbf{0.86}$ | —                  |

The MLP delivers a $21.6\%$ AUC improvement over Hotelling's $T^2$ and a $21.8\%$ improvement over the threshold baseline.

### 8.2 ROC curve interpretation

- The threshold classifier is a 4-point step function (only 4 possible drift-vote scores).
- Hotelling's $T^2$ produces continuous scores — so its ROC is smooth but bounded by the linear decision surface.
- The MLP ensemble produces continuous, well-calibrated probabilities — its ROC hugs the upper-left corner.

Curve smoothness reflects information richness: more distinct scores produce smoother curves. Curve position reflects accuracy — the MLP is both smooth _and_ high.

### 8.3 Live deployment — `ibm_kingston`

A 24-hour continuous monitoring run on `ibm_kingston`, with 99 measurement rounds at 15-minute intervals, captured **three real IBM recalibration events**:

| Event | First DRIFTED verdict (Quantum Canary) | IBM recalibration timestamp | Lead time       |
| ----- | -------------------------------------- | --------------------------- | --------------- |
| 1     | 22:13 UTC                              | 02:18 UTC (next day)        | $4.07~\text{h}$ |
| 2     | 09:42 UTC                              | 13:45 UTC                   | $4.05~\text{h}$ |
| 3     | 16:30 UTC                              | 18:25 UTC                   | $1.91~\text{h}$ |

**Average lead time: $\mathbf{3.34~\text{hours}}$.**

In the same 99-round window, the threshold classifier produced **20 false-positive drift verdicts** during periods of stable hardware. Quantum Canary produced zero false positives.

---

## 9. Discussion

### 9.1 Operational interpretation

A researcher running a 12-hour variational algorithm on `ibm_kingston` would have approximately 3 hours of advance warning that their hardware was degrading — long enough to checkpoint, switch backends, or pause until recalibration.

### 9.2 Why the MLP outperforms the statistical baselines

The borderline regime is the smoking gun. Hardware that has drifted by 20% in $T_1$ is genuinely degraded but produces fidelity triplets that overlap the stable class. Linear methods see these as "stable enough" because no individual feature crosses a clean threshold. The MLP recognises the joint signature — a slight depression of $F_{\text{coherence}}$ combined with a slight increase in $F_{\text{gate}}$ variance — even when no single feature would trigger an alarm.

### 9.3 Limitations

- **Single backend tested live.** Generalisation to other Heron r2 chips, to Eagle-class chips, and to non-IBM hardware (IonQ, Rigetti) is future work.
- **Free-tier shot budget.** Continuous 15-minute monitoring uses approximately 288,000 shots over 24 hours, exceeding IBM's free-tier monthly allocation.
- **Three features.** A larger probe set might capture additional drift modes (e.g., readout-only drift) the current three circuits underweight.

---

## 10. Conclusion

Quantum Canary demonstrates that a small, physics-informed neural network can detect quantum hardware drift hours ahead of cloud providers' published calibration data. The system uses three lightweight quantum probes, runs entirely on free-tier IBM Quantum access, and outperforms the strongest classical statistical baseline (Hotelling's $T^2$) by $21.6\%$ on held-out data. In live deployment it caught every IBM recalibration event in a 24-hour window with an average $3.34$-hour lead time. The full pipeline — including pre-trained models — is released open-source under the MIT licence, deployable by any researcher alongside their existing IBM Quantum workflow.

---

## References

- Carroll, A. et al. (2022). _Dynamics of Superconducting Qubit Relaxation Times_. npj Quantum Information **8**, 132.
- Hotelling, H. (1947). _Multivariate Quality Control_. In Techniques of Statistical Analysis. McGraw-Hill.
- Krantz, P. et al. (2019). _A Quantum Engineer's Guide to Superconducting Qubits_. Applied Physics Reviews **6**, 021318.
- Kingma, D. P. & Ba, J. (2014). _Adam: A Method for Stochastic Optimization_. arXiv:1412.6980.
- Mahalanobis, P. C. (1936). _On the Generalized Distance in Statistics_. Proc. National Institute of Sciences of India.
- Preskill, J. (2018). _Quantum Computing in the NISQ Era and Beyond_. Quantum **2**, 79.
- Bravo-Montes, J. et al. (2024). _Combined depolarizing and thermal-relaxation noise models for cloud quantum hardware_. arXiv:2403.08129.
