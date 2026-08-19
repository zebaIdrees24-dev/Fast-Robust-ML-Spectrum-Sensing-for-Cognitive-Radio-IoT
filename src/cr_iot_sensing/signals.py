from __future__ import annotations

import numpy as np


def generate_iq(samples: int, snr_db: float, present: bool, seed: int, noise_uncertainty_db: float = 1.0, impulse_rate: float = 0.002) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise_scale = 10 ** (rng.uniform(-noise_uncertainty_db, noise_uncertainty_db) / 20)
    noise = noise_scale * (rng.normal(size=samples) + 1j * rng.normal(size=samples)) / np.sqrt(2)
    impulses = rng.random(samples) < impulse_rate
    noise[impulses] += 5 * (rng.normal(size=impulses.sum()) + 1j * rng.normal(size=impulses.sum()))
    if not present:
        return noise
    symbols = rng.choice([-1, 1], size=(samples + 7) // 8)
    signal = np.repeat(symbols, 8)[:samples].astype(complex)
    signal *= np.exp(1j * 2 * np.pi * 0.07 * np.arange(samples))
    signal /= np.sqrt(np.mean(np.abs(signal) ** 2))
    return noise + np.sqrt(10 ** (snr_db / 10)) * signal


def load_iq(path: str) -> np.ndarray:
    values = np.load(path)
    if values.ndim != 1 or not np.iscomplexobj(values):
        raise ValueError("Expected a one-dimensional complex NumPy IQ array")
    return values

