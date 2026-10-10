"""Basic tests for quant_system components."""

import pandas as pd
import numpy as np
from quant_system.features.indicators import FeatureEngineer
from quant_system.models.predictor import QuantPredictor

def test_feature_engineering():
    # Create synthetic OHLCV data
    dates = pd.date_range("2025-01-01", periods=100)
    data = {
        "$open": np.random.uniform(100, 150, 100),
        "$high": np.random.uniform(150, 200, 100),
        "$low": np.random.uniform(50, 100, 100),
        "$close": np.random.uniform(100, 150, 100),
        "$volume": np.random.uniform(1000, 5000, 100)
    }
    df = pd.DataFrame(data, index=dates)

    engineer = FeatureEngineer()
    enriched = engineer.add_technical_indicators(df)
    labeled = engineer.add_target_label(enriched, horizon=1)

    assert "rsi_14" in labeled.columns
    assert "macd" in labeled.columns
    assert "darvas_position" in labeled.columns
    assert "target_label" in labeled.columns

def test_predictor_train_predict():
    dates = pd.date_range("2025-01-01", periods=100)
    data = {
        "$open": np.random.uniform(100, 150, 100),
        "$high": np.random.uniform(150, 200, 100),
        "$low": np.random.uniform(50, 100, 100),
        "$close": np.random.uniform(100, 150, 100),
        "$volume": np.random.uniform(1000, 5000, 100)
    }
    df = pd.DataFrame(data, index=dates)
    engineer = FeatureEngineer()
    labeled = engineer.add_target_label(engineer.add_technical_indicators(df))

    predictor = QuantPredictor()
    metrics = predictor.train(labeled)
    assert "test_accuracy" in metrics

    pred = predictor.predict_latest(labeled, symbol="TEST")
    assert pred["signal"] in ["BULLISH", "BEARISH", "NEUTRAL"]
    assert 0.0 <= pred["confidence"] <= 1.0
