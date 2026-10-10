"""
Quickstart Example:
Demonstrates zero-cost data collection with OpenBB, feature engineering,
and machine learning prediction with Qlib/LightGBM.
"""

from quant_system import DataCollector, FeatureEngineer, QuantPredictor

def main():
    symbol = "SOL-USD"
    print(f"1. Obteniendo datos de mercado para {symbol} (100% Gratis)...")
    collector = DataCollector()
    df = collector.fetch_historical(symbol, days=365)
    print(f"   -> {len(df)} velas descargadas correctamente.")

    print("\n2. Generando indicadores técnicos y factores alfa...")
    engineer = FeatureEngineer()
    df_features = engineer.add_technical_indicators(df)
    df_labeled = engineer.add_target_label(df_features, horizon=1)
    print(f"   -> Factores calculados (RSI, MACD, Darvas, Retornos).")

    print("\n3. Entrenando modelo cuantitativo de Machine Learning...")
    predictor = QuantPredictor()
    metrics = predictor.train(df_labeled)
    print(f"   -> Precisión en conjunto de prueba: {metrics['test_accuracy'] * 100:.2f}%")

    print(f"\n4. Generando predicción para la siguiente vela de {symbol}...")
    pred = predictor.predict_latest(df_features, symbol=symbol)
    print(f"   -> Señal: {pred['signal']}")
    print(f"   -> Confianza: {pred['confidence'] * 100:.1f}%")
    print(f"   -> Probabilidad Alcista: {pred['prob_bullish'] * 100:.1f}%")
    print(f"   -> Probabilidad Bajista: {pred['prob_bearish'] * 100:.1f}%")

if __name__ == "__main__":
    main()
