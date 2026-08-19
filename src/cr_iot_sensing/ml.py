from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import extract_features


@dataclass
class Benchmark:
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def dataset(iq_examples: list[np.ndarray]) -> np.ndarray:
    return np.vstack([extract_features(iq) for iq in iq_examples])


def candidate_models(seed: int = 42) -> dict[str, object]:
    return {
        "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1),
        "hist_gradient_boosting": Pipeline([("scale", StandardScaler()), ("model", HistGradientBoostingClassifier(max_iter=150, random_state=seed))]),
    }


def evaluate_models(x_train, y_train, x_test, y_test) -> tuple[object, list[Benchmark]]:
    reports, best_model, best_f1 = [], None, -1.0
    for name, model in candidate_models().items():
        model.fit(x_train, y_train); predicted = model.predict(x_test); probability = model.predict_proba(x_test)[:, 1]
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predicted, average="binary", zero_division=0)
        reports.append(Benchmark(name, accuracy_score(y_test, predicted), precision, recall, f1, roc_auc_score(y_test, probability)))
        if f1 > best_f1: best_model, best_f1 = model, f1
    return best_model, reports


def save_model(model, path: str) -> None:
    joblib.dump(model, path)

