"""
Feature Engineering Module.
Computes quantitative alpha factors, momentum, volatility, and Darvas Box levels
compatible with Machine Learning pipelines.
"""

import numpy as np
import pandas as pd

class FeatureEngineer:
    """Calculates quantitative features and target labels from OHLCV market data."""

    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Enriches OHLCV DataFrame with alpha factors, momentum, and volatility indicators."""
        out = df.copy()

        close = out["$close"]
        high = out["$high"]
        low = out["$low"]
        volume = out["$volume"]

        # 1. Log Returns
        out["return_1d"] = close.pct_change(1)
        out["return_5d"] = close.pct_change(5)
        out["return_20d"] = close.pct_change(20)

        # 2. Exponential Moving Averages (EMA Ratios)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()
        out["ema_12_26_ratio"] = (ema_12 / (ema_26 + 1e-8)) - 1.0
        out["price_to_ema50"] = (close / (ema_50 + 1e-8)) - 1.0

        # 3. MACD
        out["macd"] = ema_12 - ema_26
        out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
        out["macd_hist"] = out["macd"] - out["macd_signal"]

        # 4. RSI (Relative Strength Index - 14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        out["rsi_14"] = 100 - (100 / (1 + rs))

        # 5. Volatility (20-day realized volatility)
        out["volatility_20d"] = out["return_1d"].rolling(window=20).std()

        # 6. Darvas Box Indicators (High/Low Channel & Breakout distance)
        box_high = high.rolling(window=20).max()
        box_low = low.rolling(window=20).min()
        out["darvas_box_high"] = box_high
        out["darvas_box_low"] = box_low
        out["darvas_position"] = (close - box_low) / ((box_high - box_low) + 1e-8)

        # 7. Volume Momentum
        vol_ma20 = volume.rolling(window=20).mean()
        out["volume_ratio"] = volume / (vol_ma20 + 1e-8)

        return out

    @staticmethod
    def add_target_label(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0) -> pd.DataFrame:
        """
        Creates binary classification target:
        1 if price rises above threshold after `horizon` bars, 0 otherwise.
        """
        out = df.copy()
        future_return = out["$close"].shift(-horizon) / out["$close"] - 1.0
        out["target_return"] = future_return
        out["target_label"] = (future_return > threshold).astype(int)
        return out
