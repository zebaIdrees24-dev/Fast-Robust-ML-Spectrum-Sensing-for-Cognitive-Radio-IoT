# Fast and Robust Spectrum Sensing for Cognitive-Radio IoT

Python/ML research implementation inspired by Idrees et al., *Fast and Robust Spectrum Sensing for Cognitive Radio Enabled IoT* (IEEE Access, 2021).

Included:

- synthetic low-SNR complex-IQ data with noise uncertainty and impulsive interference;
- energy and MME baselines;
- fast robust PC detector using Ledoit-Wolf covariance shrinkage and restarted Lanczos dominant components;
- eigen-spectrum, entropy, power-quantile and autocorrelation features;
- Random Forest and HistGradientBoosting classifiers;
- stratified evaluation with accuracy, precision, recall, F1 and ROC-AUC;
- model persistence, tests, and a five-layer CR-IoT architecture map.

```bash
python -m venv .venv
python -m pip install -e ".[dev,plot]"
pytest -q
cr-iot-sensing --per-class 400 --samples 512
```

## Reproducibility boundary

The paper reports experimental wireless-microphone/USRP evaluation, but the linked record does not provide the raw IQ dataset or reference source code. The robust covariance and supervised classifiers in this repository are transparent engineering extensions, **not claims of verbatim reproduction**. Replace the generator with governed `.npy` complex-IQ captures and match the paper's acquisition parameters for experimental comparison.

Paper: https://doi.org/10.1109/ACCESS.2021.3133336

