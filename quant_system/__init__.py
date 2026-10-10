"""
quant_system: Autonomous quantitative investment & ML prediction pipeline.
Combines zero-cost OpenBB data ingestion with Microsoft Qlib modeling.
"""

__version__ = "0.1.0"

from .data.collector import DataCollector
from .features.indicators import FeatureEngineer
from .models.predictor import QuantPredictor
from .bridge.jarvis_bridge import JarvisBridge

__all__ = [
    "DataCollector",
    "FeatureEngineer",
    "QuantPredictor",
    "JarvisBridge",
]
