# quant_system 📈🤖

> **Autonomous Quantitative Investment & ML Prediction Engine**  
> Fusing **OpenBB** zero-cost data ingestion with **Microsoft Qlib** quantitative machine learning.

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Zero-Cost](https://img.shields.io/badge/APIs-100%25%20Free%20%26%20Open-success.svg)](#zero-cost-philosophy)

---

## 💡 Overview

**quant_system** is an open-source, standalone quantitative platform designed to bridge the gap between financial market data and machine learning intelligence without paying thousands of dollars for institutional subscriptions.

1. **OpenBB Data Layer (Zero-Cost):** Ingests historical and real-time market data (Crypto, Equities, Forex, Macro) relying exclusively on open, free providers (`yfinance`, `fred`, `sec`, `cboe`).
2. **Feature Engineering:** Calculates quantitative alpha factors, momentum indicators (RSI, MACD), realized volatility, and **Darvas Box** levels.
3. **Microsoft Qlib & LightGBM Brain:** Trains time-series models on historical features to predict forward return probabilities.
4. **Autonomous Bridge & REST API:** Delivers structured JSON signals and natural language summaries ready for trading bots, dashboards, or AI voice assistants (such as **JARVIS**).

---

## 🏛️ Architecture

```
 ┌─────────────────────────────────────────────────────────────┐
 │                      quant_system                           │
 ├─────────────────────────────────────────────────────────────┤
 │                                                             │
 │  1. Ingestion: OpenBB Free Providers (SOL, NEAR, BTC, AAPL) │
 │         │                                                   │
 │         ▼                                                   │
 │  2. Feature Engineering: Alpha Factors & Darvas Channels    │
 │         │                                                   │
 │         ▼                                                   │
 │  3. Model Core: Qlib LightGBM Direction Classifier          │
 │         │                                                   │
 │         ▼                                                   │
 │  4. Interfaces:                                             │
 │     ├── CLI Tool: python -m quant_system.cli SOL-USD        │
 │     ├── FastAPI REST Server: http://127.0.0.1:8088/predict │
 │     └── Voice Bridge: Formatted sentences for assistants    │
 └─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install the lightweight requirements:

```bash
git clone https://github.com/TU_USUARIO/quant_system.git
cd quant_system

# Create virtual environment (Python 3.10 - 3.12 recommended)
py -3.12 -m venv .venv
.\.venv\Scripts\activate   # En Windows PowerShell
# source .venv/bin/activate  # En Linux / macOS

pip install -r requirements.txt
```

### 2. Run from Command Line (CLI)

Analyze any asset directly in your terminal:

```bash
# Analyze Solana (SOL-USD)
python -m quant_system.cli SOL-USD

# Analyze NEAR Protocol or Equities
python -m quant_system.cli NEAR-USD --days 365
python -m quant_system.cli AAPL --json
```

**Example Output:**
```
==================================================
  QUANT_SYSTEM ANALYSIS: SOL-USD
==================================================
Último Precio : $174.20
Señal         : BULLISH
Confianza     : 78.4%
Prob. Alcista : 78.4%
Prob. Bajista : 21.6%
RSI (14)      : 58.40
--------------------------------------------------
Voz JARVIS    : "Señor, el análisis cuantitativo de SOL-USD indica señal alcista con un 78.4% de probabilidad. El precio actual es 174.20 y el RSI está en 58."
==================================================
```

### 3. Launch Local REST API

Launch the local API server for your dashboard or bots:

```bash
python -m quant_system.cli --serve
```
The server will run at `http://127.0.0.1:8088`. You can inspect the interactive documentation at `http://127.0.0.1:8088/docs`.

---

## 🔌 Connecting with JARVIS / AI Assistants

You can integrate `quant_system` into any voice assistant or trading bot in just 3 lines of Python:

```python
from quant_system.bridge import JarvisBridge

bridge = JarvisBridge()
analysis = bridge.analyze_asset("SOL-USD")

# For Voice TTS:
print(analysis["voice_summary"])

# For Algorithmic Orders:
if analysis["signal"] == "BULLISH" and analysis["confidence"] > 0.75:
    print(f"Executing Buy Order for {analysis['symbol']}")
```

---

## 📂 Project Structure

```
quant_system/
├── config/
│   └── config.yaml            # Default symbols, hyperparameters, intervals
├── quant_system/
│   ├── data/
│   │   └── collector.py       # OpenBB & yfinance zero-cost collector
│   ├── features/
│   │   └── indicators.py      # Quantitative alpha factors & Darvas boxes
│   ├── models/
│   │   └── predictor.py       # Qlib / LightGBM classification engine
│   ├── bridge/
│   │   └── jarvis_bridge.py   # Voice & structured output bridge
│   ├── api/
│   │   └── server.py          # FastAPI REST endpoints
│   └── cli.py                 # Terminal execution entrypoint
├── examples/
│   └── quickstart.py          # Runnable step-by-step example
├── tests/
│   └── test_quant.py          # Pytest suite
├── .gitignore
├── LICENSE                    # MIT
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 🛡️ Zero-Cost Philosophy

Unlike standard quantitative setups that depend on expensive Bloomberg or Polygon terminals, `quant_system` is architected to be 100% autonomous and free:
- **No API tokens or credit cards required** for basic cryptocurrency, equity, and index market data.
- **Local computation:** Runs directly on your CPU or GPU without cloud fees.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
