"""
JARVIS Bridge Module.
Provides lightweight, ready-to-consume natural language and structured outputs
for the Mark-LIV AI assistant HUD and voice pipeline.
"""

from typing import Any, Dict
from ..data.collector import DataCollector
from ..features.indicators import FeatureEngineer
from ..models.predictor import QuantPredictor

class JarvisBridge:
    """Seamless connector for AI voice assistants (JARVIS / Mark-LIV)."""

    def __init__(self, cache_dir: str = "./data/cache"):
        self.collector = DataCollector(cache_dir=cache_dir)
        self.engineer = FeatureEngineer()
        self.predictor = QuantPredictor()
        self._trained_symbols = set()

    def analyze_asset(self, symbol: str = "SOL-USD", days: int = 365) -> Dict[str, Any]:
        """Runs the full pipeline (collect -> features -> train if needed -> predict)."""
        # 1. Fetch free data
        df = self.collector.fetch_historical(symbol=symbol, days=days)

        # 2. Extract quantitative features & labels
        features_df = self.engineer.add_technical_indicators(df)
        labeled_df = self.engineer.add_target_label(features_df, horizon=1)

        # 3. Train or update model if not trained for this session
        if symbol not in self._trained_symbols:
            self.predictor.train(labeled_df)
            self._trained_symbols.add(symbol)

        # 4. Generate structured prediction
        prediction = self.predictor.predict_latest(features_df, symbol=symbol)
        
        # 5. Add natural language summary for voice synthesis
        prediction["voice_summary"] = self._generate_voice_summary(prediction)
        return prediction

    def _generate_voice_summary(self, p: Dict[str, Any]) -> str:
        """Generates a natural speech sentence suitable for JARVIS TTS."""
        sym = p["symbol"]
        sig = p["signal"]
        conf = p["confidence"] * 100
        price = p["last_price"]
        rsi = p["indicators"]["rsi_14"]

        if sig == "BULLISH":
            return (
                f"Señor, el análisis cuantitativo de {sym} indica señal alcista "
                f"con un {conf:.1f}% de probabilidad. El precio actual es {price:.2f} y el RSI está en {rsi:.0f}."
            )
        elif sig == "BEARISH":
            return (
                f"Atención: {sym} muestra presión bajista con una probabilidad de caída del {conf:.1f}%. "
                f"El precio actual se encuentra en {price:.2f} con RSI en {rsi:.0f}."
            )
        else:
            return (
                f"El activo {sym} se mantiene en rango neutral (sin dirección clara). "
                f"Precio en {price:.2f}. Sugiero esperar confirmación de volumen antes de operar."
            )
