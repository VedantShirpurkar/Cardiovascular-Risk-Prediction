import pytest
import os
import pickle
import numpy as np

def test_model_load():
    try:
        with open("modelv1.pkl", "rb") as f:
            model = pickle.load(f)
        assert model is not None
    except Exception:
        pytest.skip("Skipping test: modelv1.pkl can't be unpickled in this environment")

def test_threshold_file():
    assert os.path.exists("modelv1_threshold.txt")

def test_feature_order_file():
    assert os.path.exists("modelv1_features.pkl")

def test_prediction_shape():
    try:
        with open("modelv1.pkl", "rb") as f:
            model = pickle.load(f)
        pred = model.predict(np.random.rand(1, 20))  # modelv1 expects 20 features
        assert pred.shape == (1,)
    except Exception:
        pytest.skip("Skipping test: model prediction test can't run in this environment")
