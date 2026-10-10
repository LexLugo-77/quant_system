"""
FastAPI Server Module.
Provides local REST API endpoints for external applications (JARVIS, HUD, TradingView Webhooks).
"""

import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ..bridge.jarvis_bridge import JarvisBridge

app = FastAPI(
    title="quant_system API",
    description="Zero-Cost Quantitative AI Data & Prediction Engine",
    version="0.1.0"
)

# Enable CORS for local dashboards & Mark-LIV HUD
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bridge = JarvisBridge()

class PredictResponse(BaseModel):
    symbol: str
    timestamp: str
    last_price: float
    signal: str
    confidence: float
    prob_bullish: float
    prob_bearish: float
    indicators: dict
    voice_summary: str

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "quant_system",
        "version": "0.1.0",
        "free_providers_only": True
    }

@app.get("/predict", response_model=PredictResponse)
def get_prediction(
    symbol: str = Query(default="SOL-USD", description="Ticker symbol (e.g., SOL-USD, NEAR-USD, AAPL)"),
    days: int = Query(default=365, description="Historical days for training window")
):
    """Analyzes market data with zero-cost feeds and returns quantitative prediction."""
    try:
        result = bridge.analyze_asset(symbol=symbol, days=days)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def main():
    import uvicorn
    port = int(os.environ.get("QUANT_PORT", 8088))
    uvicorn.run("quant_system.api.server:app", host="127.0.0.1", port=port, reload=False)

if __name__ == "__main__":
    main()
