from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import eigsh

from .features import lag_matrix, robust_covariance


@dataclass
class DetectionResult:
    statistic: float
    elapsed_ms: float


def energy(iq: np.ndarray) -> DetectionResult:
    start = time.perf_counter(); statistic = float(np.mean(np.abs(iq) ** 2))
    return DetectionResult(statistic, (time.perf_counter() - start) * 1000)


def mme(iq: np.ndarray, order: int = 8) -> DetectionResult:
    start = time.perf_counter(); values = np.linalg.eigvalsh(robust_covariance(iq, order))
    return DetectionResult(float(values[-1] / max(values[0], 1e-12)), (time.perf_counter() - start) * 1000)


def fast_robust_pc(iq: np.ndarray, order: int = 8, components: int = 2) -> DetectionResult:
    """Robust covariance plus dominant-only restarted-Lanczos projection."""
    start = time.perf_counter()
    x_complex = lag_matrix(iq, order)
    covariance = robust_covariance(iq, order)
    _, basis = eigsh(covariance, k=components, which="LM")
    x_real = np.c_[x_complex.real.T, x_complex.imag.T].T
    projected = basis.T @ x_real
    statistic = float(np.sum(projected**2) / projected.shape[1])
    return DetectionResult(statistic, (time.perf_counter() - start) * 1000)

