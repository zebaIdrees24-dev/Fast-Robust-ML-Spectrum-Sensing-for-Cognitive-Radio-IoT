# Fast & Robust ML Spectrum Sensing for Cognitive-Radio IoT

A reproducible **Machine Learning, statistical signal processing, and Cognitive-Radio IoT (CR-IoT) framework** for reliable spectrum-occupancy detection under **low-SNR, noise-uncertain, and interference-prone wireless conditions**.

The project combines **complex I/Q signal processing, Energy Detection, Maximum-Minimum Eigenvalue (MME) sensing, Ledoit-Wolf robust covariance estimation, PCA/Lanczos dominant-component analysis, statistical feature engineering, Random Forest, and HistGradientBoosting** within a modular Python research implementation.

This repository is based on my published research:

**Z. Idrees et al., “Fast and Robust Spectrum Sensing for Cognitive Radio Enabled IoT,” IEEE Access, 2021.**

The repository translates the core research methodology into a reproducible Python implementation and extends it with robust covariance estimation, feature engineering, and supervised machine-learning workflows for Cognitive-Radio IoT spectrum sensing.

---

# Overview

Cognitive-Radio IoT networks require reliable identification of unused spectrum before secondary IoT devices can transmit.

Spectrum sensing becomes particularly difficult when:

- Signal-to-noise ratio is low
- Noise power is uncertain
- Interference is impulsive
- Observation windows are short
- Covariance matrices are poorly conditioned
- RF environments change over time
- IoT devices have limited computational resources
- False alarms reduce spectrum utilization
- Missed detections can interfere with primary users

This repository investigates a **hybrid signal-processing and machine-learning approach** to this problem.

Rather than relying on a single detection statistic, the framework extracts information from:

- received signal power
- covariance structure
- eigenvalue distributions
- principal components
- spectral entropy
- power quantiles
- temporal autocorrelation

and uses these features for robust spectrum-occupancy classification.

---

# Key Capabilities

- Synthetic complex I/Q signal generation
- Low-SNR spectrum-sensing experiments
- Noise-uncertainty simulation
- Impulsive-interference simulation
- Energy Detection baseline
- Maximum-Minimum Eigenvalue detection
- Ledoit-Wolf covariance shrinkage
- PCA/eigenvalue analysis
- Restarted Lanczos dominant-component extraction
- Eigen-spectrum feature engineering
- Spectral entropy features
- Power-quantile features
- Autocorrelation features
- Random Forest classification
- HistGradientBoosting classification
- Stratified model evaluation
- ROC-AUC analysis
- Model persistence
- Automated testing
- Docker containerization
- GitHub Actions CI
- Five-layer CR-IoT architecture mapping

---

# Problem Formulation

Spectrum sensing can be represented as a binary hypothesis-testing problem:

```text
H0: x[n] = w[n]

H1: x[n] = h[n]s[n] + w[n]
```

where:

- `H0` = spectrum is vacant
- `H1` = primary-user signal is present
- `x[n]` = received signal
- `s[n]` = transmitted signal
- `h[n]` = channel response
- `w[n]` = noise

The objective is to estimate:

```text
Spectrum State ∈ {Vacant, Occupied}
```

while maintaining high detection probability and low false-alarm probability under challenging channel conditions.

---

# End-to-End ML Pipeline

```text
Complex I/Q Samples
        |
        v
Signal Conditioning
        |
        v
Robust Covariance Estimation
        |
        +-----------------------------+
        |                             |
        v                             v
Classical Detectors             Feature Engineering
        |                             |
  Energy / MME                  +-----+----------------+
        |                       |      |       |        |
        |                       v      v       v        v
        |                    Eigen   Entropy  Power   Auto-
        |                   Spectrum         Stats   correlation
        |                       |      |       |        |
        |                       +------+-------+--------+
        |                              |
        |                              v
        |                       Feature Vector
        |                              |
        |                    +---------+----------+
        |                    |                    |
        |                    v                    v
        |              Random Forest     HistGradientBoosting
        |                    |                    |
        +--------------------+--------------------+
                             |
                             v
                   Spectrum Occupancy
                    Vacant / Occupied
```

---

# Why Robust Spectrum Sensing?

Conventional detectors can perform well when assumptions about the signal and noise environment are accurate.

Real wireless environments, however, may contain:

- uncertain noise variance
- interference bursts
- nonstationary signals
- finite-sample covariance errors
- low-power primary signals

The project therefore combines **robust statistical estimation with machine learning** to create a richer sensing representation.

---

# Synthetic Complex I/Q Signals

The default experiments use synthetic complex-valued baseband samples:

```text
x[n] = I[n] + jQ[n]
```

where:

- `I[n]` = in-phase component
- `Q[n]` = quadrature component

Complex I/Q representation preserves amplitude and phase information commonly available from software-defined-radio receivers.

---

# Challenging RF Conditions

The synthetic signal generator is designed to support difficult sensing conditions.

## Low SNR

Primary-user signals can be generated at low signal-to-noise ratios to evaluate detector sensitivity.

Example:

```text
Signal
  |
  |   Primary-user component
  v
~~~~~~~~~~~~~~
Noise
^^^^^^^^^^^^^^^^^^^^^^^^
```

At sufficiently low SNR, the signal may be visually indistinguishable from noise.

---

## Noise Uncertainty

Real receivers rarely know the exact noise floor.

The framework therefore allows variation in noise conditions rather than assuming a perfectly known fixed variance.

Noise uncertainty is particularly important because it can reduce the reliability of conventional Energy Detection.

---

## Impulsive Interference

Short high-energy disturbances can distort:

- signal power
- covariance estimates
- eigenvalue distributions
- detection thresholds

The synthetic framework can represent such interference to investigate detector robustness.

---

# Classical Baseline 1: Energy Detection

Energy Detection provides a simple and widely used baseline.

A typical energy statistic is:

```text
T_ED = (1/N) Σ |x[n]|²
```

The decision rule is:

```text
T_ED > threshold  →  H1
T_ED ≤ threshold  →  H0
```

### Advantages

- Low computational complexity
- No prior signal knowledge required
- Straightforward implementation

### Limitations

Performance can deteriorate under:

- noise uncertainty
- very low SNR
- interference
- inaccurate threshold selection

For this reason, Energy Detection is treated primarily as a reference baseline.

---

# Classical Baseline 2: Maximum-Minimum Eigenvalue Detection

The **MME detector** analyzes the covariance eigenvalue distribution.

A typical statistic is:

```text
T_MME = λmax / λmin
```

where:

- `λmax` = maximum covariance eigenvalue
- `λmin` = minimum covariance eigenvalue

Under noise-only conditions, eigenvalues tend to be more similar.

A structured signal can increase the separation between dominant and minimum eigenvalues.

---

# Robust Covariance Estimation

One of the key differences between this project and simpler eigenvalue-based sensing implementations is the use of **Ledoit-Wolf covariance shrinkage**.

The ordinary sample covariance matrix can become unstable when:

- the number of observations is limited
- features are highly correlated
- the signal is weak
- interference is present

Ledoit-Wolf shrinkage combines the sample covariance with a structured target.

Conceptually:

```text
Σ_robust = (1 - α) Σ_sample + αT
```

where:

- `Σ_sample` = empirical covariance
- `T` = shrinkage target
- `α` = estimated shrinkage intensity

This can provide a better-conditioned covariance estimate for downstream eigenvalue analysis.

---

# PCA and Dominant Components

Principal Component Analysis identifies directions associated with dominant variation in the received signal.

```text
Received Samples
       |
       v
Robust Covariance
       |
       v
Eigenvalue Analysis
       |
       v
Dominant Components
```

The resulting eigen-spectrum can reveal structure that is difficult to identify using signal power alone.

---

# Restarted Lanczos Method

Full eigendecomposition can become computationally expensive when only a few dominant components are required.

The Lanczos method iteratively approximates dominant eigeninformation using a Krylov subspace:

```text
K_m(A,q) =
span{q, Aq, A²q, ..., A^(m-1)q}
```

where:

- `A` = covariance matrix
- `q` = initial vector
- `m` = Krylov-subspace dimension

A restarted implementation limits subspace growth while retaining useful dominant-component information.

```text
Covariance Matrix
       |
       v
Lanczos Iteration
       |
       v
Krylov Subspace
       |
       v
Dominant Components
       |
       v
Restart / Convergence
```

This supports computationally efficient eigenfeature extraction.

---

# Feature Engineering

A major focus of this repository is converting raw complex I/Q observations into informative ML features.

The feature set combines several complementary signal characteristics.

---

## 1. Eigen-Spectrum Features

Potential features include:

- dominant eigenvalue
- minimum eigenvalue
- eigenvalue ratio
- eigenvalue spread
- normalized eigenvalues
- covariance trace
- dominant-component energy

These features capture structural information in the covariance matrix.

---

## 2. Entropy Features

Entropy measures characterize the distribution of signal energy or spectral information.

Conceptually:

```text
H = - Σ p_i log(p_i)
```

where `p_i` represents normalized energy or spectral contributions.

Entropy can help distinguish:

- structured transmissions
- broadband noise
- interference
- mixed signal conditions

---

## 3. Power-Quantile Features

Instead of representing signal power using only the mean, quantiles characterize different parts of the power distribution.

Examples include:

- lower power quantile
- median power
- upper power quantile
- interquantile range

These statistics can improve robustness to isolated high-energy samples.

---

## 4. Autocorrelation Features

Autocorrelation captures temporal structure:

```text
R_x[k] = E{x[n]x*[n-k]}
```

Noise and modulated signals can exhibit different correlation behavior.

Autocorrelation therefore provides information that is complementary to power and covariance features.

---

# Combined Feature Vector

The resulting ML representation can be viewed as:

```text
Feature Vector
|
├── Power Statistics
├── Eigenvalue Features
├── Principal-Component Features
├── Entropy Features
├── Quantile Features
└── Autocorrelation Features
```

This multidimensional representation allows ML classifiers to learn decision boundaries that cannot be expressed by a single hand-designed threshold.

---

# Machine Learning Models

## Random Forest

Random Forest combines multiple decision trees trained using randomized subsets of samples and features.

It is particularly useful here because it:

- captures nonlinear feature relationships
- handles mixed statistical features
- requires relatively little preprocessing
- supports feature-importance analysis
- provides a strong tabular ML baseline

---

## HistGradientBoosting

HistGradientBoosting builds an additive ensemble of decision trees using gradient-based optimization and histogram-based feature binning.

Potential advantages include:

- nonlinear classification
- efficient tabular learning
- interaction modeling
- strong predictive performance

The repository compares this approach with Random Forest rather than relying on a single ML algorithm.

---

# Classical Detection vs ML Detection

| Method | Information Used | Training | Main Role |
|---|---|---:|---|
| Energy Detector | Received power | No | Classical baseline |
| MME | Covariance eigenvalue ratio | No | Eigenvalue baseline |
| Robust PC/Lanczos | Dominant covariance structure | No | Robust signal representation |
| Random Forest | Multiple engineered features | Yes | ML occupancy classification |
| HistGradientBoosting | Multiple engineered features | Yes | Boosted ML classification |

This comparison demonstrates the progression from:

```text
Single Statistic
      ↓
Covariance Structure
      ↓
Robust Eigenfeatures
      ↓
Multidimensional Features
      ↓
Machine Learning
```

---

# Model Evaluation

The supervised ML models are evaluated using stratified data splitting and standard classification metrics.

## Accuracy

```text
Accuracy = Correct Predictions / Total Predictions
```

## Precision

Measures how often predicted occupied-spectrum observations are actually occupied.

## Recall

Measures how many occupied-spectrum observations are successfully detected.

## F1-Score

Balances precision and recall:

```text
F1 = 2 × Precision × Recall
         ------------------
         Precision + Recall
```

## ROC-AUC

Measures classification discrimination across different decision thresholds.

---

# Spectrum-Sensing Metrics

For RF sensing research, ML metrics should be complemented by traditional detection metrics.

## Probability of Detection

```text
Pd = P(decide H1 | H1)
```

## Probability of False Alarm

```text
Pfa = P(decide H1 | H0)
```

A useful detector should ideally achieve:

```text
High Pd + Low Pfa
```

especially under low-SNR conditions.

---

# Five-Layer CR-IoT Architecture

The repository maps spectrum sensing into a broader Cognitive-Radio IoT architecture.

```text
+--------------------------------------------------+
| Layer 5: Applications                            |
| Smart industry | Smart grid | Connected systems |
+--------------------------------------------------+
                       ↑
+--------------------------------------------------+
| Layer 4: Intelligence / Decision Layer           |
| ML classification | Spectrum decisions          |
+--------------------------------------------------+
                       ↑
+--------------------------------------------------+
| Layer 3: Cognitive-Radio Processing              |
| Features | PCA | Lanczos | MME | Detection       |
+--------------------------------------------------+
                       ↑
+--------------------------------------------------+
| Layer 2: Communication / Edge Layer              |
| CR gateway | SDR | Wireless connectivity         |
+--------------------------------------------------+
                       ↑
+--------------------------------------------------+
| Layer 1: Physical IoT / RF Layer                 |
| Sensors | RF signals | Primary/secondary users   |
+--------------------------------------------------+
```

This architecture illustrates how spectrum-sensing algorithms can fit within an intelligent IoT system rather than operating as isolated signal-processing routines.

---

# CR-IoT Decision Workflow

```text
RF Environment
      |
      v
SDR / RF Front End
      |
      v
Complex I/Q Samples
      |
      v
Edge Signal Processing
      |
      v
Robust Feature Extraction
      |
      v
ML Spectrum Classifier
      |
      v
Spectrum Occupancy Decision
      |
      +------------------+
      |                  |
      v                  v
Channel Vacant       Channel Occupied
      |                  |
      v                  v
Secondary Access     Protect Primary User
```

---

# Repository Structure

```text
Fast-Robust-ML-Spectrum-Sensing-for-Cognitive-Radio-IoT/
│
├── .github/
│   └── workflows/
│       └── CI configuration
│
├── docs/
│   └── architecture and project documentation
│
├── src/
│   └── cr_iot_sensing/
│       └── core signal-processing and ML implementation
│
├── tests/
│   └── automated tests
│
├── .gitignore
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

# Quick Start

## 1. Clone the Repository

```bash
git clone https://github.com/zebaIdrees24-dev/Fast-Robust-ML-Spectrum-Sensing-for-Cognitive-Radio-IoT.git

cd Fast-Robust-ML-Spectrum-Sensing-for-Cognitive-Radio-IoT
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

---

## 3. Install the Package

For the core package:

```bash
python -m pip install -e .
```

For development and plotting dependencies:

```bash
python -m pip install -e ".[dev,plot]"
```

---

# Run Tests

```bash
pytest -q
```

The automated tests provide a reproducible check of important repository functionality.

---

# Run a Spectrum-Sensing Experiment

The installed package provides the command:

```bash
cr-iot-sensing
```

A representative experiment can be launched with:

```bash
cr-iot-sensing --per-class 400 --samples 512
```

This generates synthetic spectrum-sensing observations and executes the configured ML workflow.

---

# Docker

Build the container:

```bash
docker build -t cr-iot-spectrum-sensing .
```

Run it:

```bash
docker run --rm cr-iot-spectrum-sensing
```

Docker provides a reproducible runtime environment independent of the host Python installation.

---

# Continuous Integration

The repository includes GitHub Actions configuration under:

```text
.github/workflows/
```

CI provides automated validation when repository changes are pushed.

Combined with:

- Python packaging
- automated tests
- Docker
- version-controlled configuration

this gives the project a software-engineering structure beyond a notebook-only research demonstration.

---

# Technology Stack

## Machine Learning

- Python
- scikit-learn
- Random Forest
- HistGradientBoosting
- supervised classification
- feature engineering
- model persistence
- ROC-AUC analysis

## Signal Processing

- Complex I/Q signals
- Energy Detection
- Maximum-Minimum Eigenvalue sensing
- covariance analysis
- Ledoit-Wolf shrinkage
- PCA
- Lanczos iteration
- eigen-spectrum analysis
- spectral entropy
- power statistics
- autocorrelation

## Scientific Computing

- NumPy
- SciPy
- pandas
- Matplotlib

## Wireless / IoT

- Cognitive Radio
- Dynamic Spectrum Access
- CR-IoT
- low-SNR detection
- spectrum occupancy
- noise uncertainty
- interference modeling

## Software Engineering

- Python packaging
- CLI
- pytest
- Docker
- GitHub Actions
- Git

---

# Difference from the Lanczos-PCA Repository

This repository complements, rather than duplicates, the separate **ML-Enhanced Cognitive Radio Spectrum Sensing with Lanczos-PCA** project.

| Repository | Primary Focus |
|---|---|
| **ML-Enhanced Cognitive Radio Spectrum Sensing with Lanczos-PCA** | Reproduction and extension of low-complexity principal-component/eigenvalue spectrum sensing |
| **Fast & Robust ML Spectrum Sensing for CR-IoT** | Robust covariance estimation, richer signal features, supervised ML, and CR-IoT-oriented spectrum classification |

### Lanczos-PCA Project

The primary emphasis is:

```text
Covariance
    ↓
Principal Components
    ↓
Lanczos Approximation
    ↓
Classical Spectrum Detection
```

### This CR-IoT Project

The primary emphasis is:

```text
Complex I/Q
    ↓
Robust Covariance
    ↓
Lanczos / PCA
    ↓
Rich Statistical Features
    ↓
Random Forest / HistGradientBoosting
    ↓
Robust CR-IoT Spectrum Classification
```

Together, the two repositories demonstrate both **classical statistical signal-processing expertise and modern ML-based wireless sensing**.

---

# Reproducibility Boundary

The associated IEEE Access study reports experimental evaluation involving wireless-microphone signals and USRP-based RF measurements.

The public research record does not provide the complete original raw I/Q dataset or reference source code required for exact bit-for-bit reproduction.

Therefore, this repository uses:

- synthetic complex I/Q observations
- configurable low-SNR conditions
- noise uncertainty
- impulsive interference
- transparent Python implementations

The following components should be understood as **engineering extensions** rather than claims of verbatim reproduction of the original experimental system:

- Ledoit-Wolf covariance shrinkage
- Random Forest classification
- HistGradientBoosting classification
- the current synthetic-data generator
- the current software-engineering architecture

For closer experimental comparison, users should replace the synthetic generator with governed complex-I/Q captures and match the acquisition conditions used in the published study.

---

# Research Questions

This framework can support investigation of questions such as:

1. How does spectrum-sensing performance change as SNR decreases?
2. How strongly does noise uncertainty affect Energy Detection?
3. Can robust covariance estimation improve eigenvalue stability?
4. How does MME compare with ML-based occupancy classification?
5. Which eigen-spectrum features contribute most to classification?
6. Do entropy features improve low-SNR sensing?
7. How useful are power quantiles under impulsive interference?
8. Does autocorrelation provide complementary information to covariance features?
9. How do Random Forest and HistGradientBoosting compare?
10. What is the trade-off between sensing accuracy and computational complexity?
11. Can these methods be deployed efficiently on edge CR-IoT devices?
12. How well do models trained on synthetic signals generalize to real SDR data?

---

# Scope and Limitations

This repository is intended for:

- Cognitive-Radio IoT research
- spectrum-sensing experimentation
- machine-learning research
- statistical signal processing
- robust covariance analysis
- reproducible algorithm development
- educational and portfolio use

It should **not** be interpreted as:

- a production RF monitoring system
- a regulatory spectrum-management platform
- an exact recreation of the complete original experimental campaign
- evidence of real-time SDR deployment
- evidence of field-level wireless performance
- a replacement for RF hardware validation

The default dataset is synthetic.

Real complex I/Q captures are required before making conclusions about field deployment.

---

# Research Lineage

This implementation is inspired by:

**Z. Idrees et al.  
“Fast and Robust Spectrum Sensing for Cognitive Radio Enabled IoT.”  
IEEE Access, 2021.**

DOI:

```text
10.1109/ACCESS.2021.3133336
```

The repository extends the research direction with a reproducible Python ML pipeline incorporating robust covariance estimation and supervised classification.

---

# Future Work

Potential extensions include:

- Real USRP / SDR datasets
- GNU Radio integration
- real-time RF streaming
- Rayleigh fading
- Rician fading
- multipath-channel models
- cooperative spectrum sensing
- multi-node CR-IoT sensing
- CNN-based raw-I/Q classification
- LSTM-based temporal spectrum sensing
- Transformer-based RF models
- self-supervised RF representation learning
- federated spectrum sensing
- online learning
- concept-drift detection
- adversarial robustness
- explainable ML
- model quantization
- ONNX deployment
- Raspberry Pi inference
- FPGA acceleration
- dynamic spectrum-access policies
- 5G/6G spectrum-sharing experiments

---

# Applications

Potential applications include:

- Cognitive-Radio IoT
- Dynamic Spectrum Access
- Intelligent wireless networks
- Industrial IoT
- Software-Defined Radio
- RF monitoring
- Spectrum sharing
- 5G/6G systems
- Smart-grid wireless communications
- Edge wireless intelligence
- Connected infrastructure
- Low-power IoT networks

---

## Usage and Attribution

This repository is provided for research, educational, and portfolio
demonstration purposes.

The project reflects collaborative research and engineering work, together
with subsequent implementation and development. Relevant publications and
research contributions are acknowledged in the references below.

No open-source license is currently granted for this repository.

---

# Author

**Zeba Idrees**

Research and engineering interests include:

- Machine Learning
- Artificial Intelligence
- Cognitive Radio
- Spectrum Sensing
- Wireless Communications
- Signal Processing
- Edge AI
- Embedded Systems
- Industrial IoT
- Cybersecurity
- Intelligent Wireless Systems

---

# Project Status

**Active research and portfolio implementation**

The repository demonstrates the integration of **complex-I/Q signal processing, robust covariance estimation, PCA/Lanczos methods, classical spectrum sensing, statistical feature engineering, supervised machine learning, Cognitive-Radio IoT architecture, automated testing, Docker, and CI workflows**.