from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .ml import dataset, evaluate_models
from .signals import generate_iq


def build_examples(per_class: int = 400, samples: int = 512, seed: int = 42):
    examples, labels = [], []
    rng = np.random.default_rng(seed)
    for label in (0, 1):
        for index in range(per_class):
            snr = float(rng.uniform(-25, -5))
            examples.append(generate_iq(samples, snr, bool(label), seed + label * 100_000 + index))
            labels.append(label)
    return examples, np.asarray(labels)


def run_ml_benchmark(per_class: int = 400, samples: int = 512, seed: int = 42):
    examples, labels = build_examples(per_class, samples, seed)
    x = dataset(examples)
    x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.25, stratify=labels, random_state=seed)
    model, reports = evaluate_models(x_train, y_train, x_test, y_test)
    return model, pd.DataFrame([report.__dict__ for report in reports])

