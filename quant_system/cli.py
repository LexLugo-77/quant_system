"""
Command Line Interface for quant_system.
Usage:
    python -m quant_system.cli SOL-USD
    python -m quant_system.cli --serve
"""

import argparse
import json
import sys
from .bridge.jarvis_bridge import JarvisBridge
from .api.server import main as run_server

def main():
    parser = argparse.ArgumentParser(description="quant_system: Zero-Cost Quantitative ML Prediction Engine")
    parser.add_argument("symbol", nargs="?", default="SOL-USD", help="Ticker to analyze (default: SOL-USD)")
    parser.add_argument("--days", type=int, default=365, help="Days of history for training (default: 365)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--serve", action="store_true", help="Start local FastAPI REST server")

    args = parser.parse_args()

    if args.serve:
        print("Starting quant_system REST API server on http://127.0.0.1:8088...")
        run_server()
        return

    bridge = JarvisBridge()
    try:
        result = bridge.analyze_asset(symbol=args.symbol, days=args.days)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n" + "=" * 50)
            print(f"  QUANT_SYSTEM ANALYSIS: {result['symbol']}")
            print("=" * 50)
            print(f"Último Precio : ${result['last_price']:.2f}")
            print(f"Señal         : {result['signal']}")
            print(f"Confianza     : {result['confidence'] * 100:.1f}%")
            print(f"Prob. Alcista : {result['prob_bullish'] * 100:.1f}%")
            print(f"Prob. Bajista : {result['prob_bearish'] * 100:.1f}%")
            print(f"RSI (14)      : {result['indicators']['rsi_14']}")
            print("-" * 50)
            print(f"Voz JARVIS    : \"{result['voice_summary']}\"")
            print("=" * 50 + "\n")
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
