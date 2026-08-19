from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation import run_ml_benchmark
from .ml import save_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=400)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--reports", default="reports/ml_benchmark.csv")
    parser.add_argument("--model", default="artifacts/spectrum_classifier.joblib")
    args = parser.parse_args()
    model, report = run_ml_benchmark(args.per_class, args.samples)
    report_path = Path(args.reports); report_path.parent.mkdir(parents=True, exist_ok=True); report.to_csv(report_path, index=False)
    model_path = Path(args.model); model_path.parent.mkdir(parents=True, exist_ok=True); save_model(model, str(model_path))
    print(report.to_string(index=False))

