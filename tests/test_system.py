import numpy as np

from cr_iot_sensing.detectors import energy, fast_robust_pc, mme
from cr_iot_sensing.evaluation import run_ml_benchmark
from cr_iot_sensing.features import extract_features
from cr_iot_sensing.signals import generate_iq


def test_features_and_detectors():
    iq = generate_iq(512, -5, True, 1)
    assert np.isfinite(extract_features(iq)).all()
    for detector in (energy, mme, fast_robust_pc):
        result = detector(iq)
        assert result.statistic > 0


def test_ml_pipeline():
    _, report = run_ml_benchmark(per_class=30, samples=256, seed=3)
    assert set(report["model"]) == {"random_forest", "hist_gradient_boosting"}
    assert report["f1"].between(0, 1).all()

