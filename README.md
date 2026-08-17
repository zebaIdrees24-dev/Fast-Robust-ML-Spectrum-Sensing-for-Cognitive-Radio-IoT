Fast-Robust-ML-Spectrum-Sensing-for-Cognitive-Radio-IoT

Machine-learning-enhanced spectrum-sensing system for Cognitive Radio IoT (CR-IoT) designed to detect primary-user activity under challenging low-SNR, noise-uncertain, and interference-prone wireless conditions. The project processes complex I/Q RF signals and combines robust statistical signal processing with supervised machine learning for spectrum-occupancy classification.

It implements Energy Detection, MME, PCA/Lanczos-based sensing, Ledoit-Wolf covariance estimation, Random Forest, and HistGradientBoosting, with engineered features including covariance eigenvalues, eigen-spectrum entropy, power statistics, quantiles, and autocorrelation.

Models are evaluated using Pd/Pfa, accuracy, precision, recall, F1-score, ROC-AUC, SNR sweeps, and comparative benchmarking across simulated operating conditions from −25 to −5 dB SNR.

Tools & Technologies: Python, NumPy, SciPy, pandas, scikit-learn, ARPACK/Lanczos, Ledoit-Wolf covariance estimation, Random Forest, HistGradientBoosting, joblib, Jupyter, pytest, Docker, Git, and GitHub Actions.
