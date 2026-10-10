"""
Quant Predictor Module.
Trains Gradient Boosted Trees (LightGBM/Qlib) to predict forward market directions.
Generates structured signals for automated trading execution or AI assistant consumption.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "ema_12_26_ratio",
    "price_to_ema50",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi_14",
    "volatility_20d",
    "darvas_position",
    "volume_ratio",
]

class QuantPredictor:
    """Manages training, evaluating, and predicting signals with LightGBM."""

    def __init__(self, model_dir: str = "./models/checkpoints"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.feature_cols = FEATURE_COLUMNS

    def train(
        self,
        df: pd.DataFrame,
        test_ratio: float = 0.2,
        n_estimators: int = 150,
        learning_rate: float = 0.05
    ) -> Dict[str, float]:
        """Trains LightGBM model on historical feature data and returns performance metrics."""
        clean_df = df.dropna(subset=self.feature_cols + ["target_label"]).copy()
        
        if len(clean_df) < 60:
            raise ValueError(f"Insufficient samples to train model ({len(clean_df)} valid rows).")

        split_idx = int(len(clean_df) * (1.0 - test_ratio))
        train_data = clean_df.iloc[:split_idx]
        test_data = clean_df.iloc[split_idx:]

        X_train = train_data[self.feature_cols]
        y_train = train_data["target_label"]
        X_test = test_data[self.feature_cols]
        y_test = test_data["target_label"]

        # Use LightGBM (standard backend used by Qlib)
        try:
            import lightgbm as lgb
            clf = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=5,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
            clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])
            self.model = clf
        except ImportError:
            # Fallback to scikit-learn GradientBoostingClassifier if LightGBM not yet installed
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=4,
                random_state=42
            )
            clf.fit(X_train, y_train)
            self.model = clf

        # Calculate accuracy & ROC-AUC
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]

        accuracy = float((preds == y_test).mean())
        
        metrics = {
            "test_accuracy": accuracy,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }
        logger.info(f"Training completed. Test Accuracy: {accuracy:.4f}")
        return metrics

    def predict_latest(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Takes enriched features and produces a structured prediction for the latest candle.
        Returns signal, probability, and market context.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded yet. Call train() first.")

        clean_df = df.dropna(subset=self.feature_cols)
        if clean_df.empty:
            raise ValueError("No valid features found in latest data.")

        latest_row = clean_df.iloc[[-1]]
        latest_time = str(latest_row.index[-1])
        latest_price = float(latest_row["$close"].values[0])

        prob_up = float(self.model.predict_proba(latest_row[self.feature_cols])[0, 1])
        prob_down = 1.0 - prob_up

        if prob_up >= 0.60:
            signal = "BULLISH"
            confidence = prob_up
        elif prob_down >= 0.60:
            signal = "BEARISH"
            confidence = prob_down
        else:
            signal = "NEUTRAL"
            confidence = max(prob_up, prob_down)

        rsi_val = float(latest_row["rsi_14"].values[0]) if "rsi_14" in latest_row else 50.0
        darvas_pos = float(latest_row["darvas_position"].values[0]) if "darvas_position" in latest_row else 0.5

        return {
            "symbol": symbol,
            "timestamp": latest_time,
            "last_price": latest_price,
            "signal": signal,
            "confidence": round(confidence, 4),
            "prob_bullish": round(prob_up, 4),
            "prob_bearish": round(prob_down, 4),
            "indicators": {
                "rsi_14": round(rsi_val, 2),
                "darvas_channel_pos": round(darvas_pos, 2)
            }
        }
