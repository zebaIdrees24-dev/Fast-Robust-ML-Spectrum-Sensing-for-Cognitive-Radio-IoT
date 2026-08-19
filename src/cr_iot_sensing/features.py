from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


def lag_matrix(iq: np.ndarray, order: int = 8) -> np.ndarray:
    if len(iq) <= order:
        raise ValueError("IQ sequence is shorter than covariance order")
    return np.vstack([iq[offset : len(iq) - order + offset + 1] for offset in range(order)])


def robust_covariance(iq: np.ndarray, order: int = 8) -> np.ndarray:
    x = lag_matrix(iq, order)
    observations = np.c_[x.real.T, x.imag.T]
    return LedoitWolf().fit(observations).covariance_


def extract_features(iq: np.ndarray, order: int = 8) -> np.ndarray:
    covariance = robust_covariance(iq, order)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 1e-12)
    normalized = eigenvalues / eigenvalues.sum()
    power = np.abs(iq) ** 2
    autocorrelation = np.abs(np.vdot(iq[1:], iq[:-1])) / max(len(iq) - 1, 1)
    return np.r_[
        normalized,
        eigenvalues[-1] / eigenvalues[0],
        -np.sum(normalized * np.log(normalized)),
        np.mean(power),
        np.std(power),
        np.quantile(power, [0.5, 0.9, 0.99]),
        autocorrelation,
    ]

