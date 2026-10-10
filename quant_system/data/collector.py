"""
Data Collector Module.
Handles automated market data retrieval using 100% free providers (OpenBB & yfinance).
Normalizes formats to Qlib-ready quantitative standards.
"""

from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

class DataCollector:
    """Collects free market data with local caching and standard column formatting."""

    def __init__(self, cache_dir: str = "./data/cache", provider: str = "yfinance"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.provider = provider

    def fetch_historical(
        self,
        symbol: str,
        days: int = 365,
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a crypto or equity symbol.
        Returns a normalized DataFrame with ['$open', '$high', '$low', '$close', '$volume'].
        """
        cache_file = self.cache_dir / f"{symbol.replace('/', '_')}_{interval}_{days}d.parquet"

        if use_cache and cache_file.exists():
            # Check cache freshness (refresh if older than 4 hours for intraday, 12h for daily)
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(hours=6):
                logger.info(f"Loading {symbol} from cache ({cache_file})")
                return pd.read_parquet(cache_file)

        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        df: Optional[pd.DataFrame] = None

        # 1. Try fetching via OpenBB Platform
        try:
            from openbb import obb
            logger.info(f"Fetching {symbol} via OpenBB (provider: {self.provider})...")
            
            # Check if crypto or equity
            if "-" in symbol or any(c in symbol.upper() for c in ["BTC", "SOL", "NEAR", "ETH"]):
                res = obb.crypto.price.historical(
                    symbol=symbol,
                    provider=self.provider,
                    start_date=start_date,
                    interval=interval
                )
            else:
                res = obb.equity.price.historical(
                    symbol=symbol,
                    provider=self.provider,
                    start_date=start_date,
                    interval=interval
                )
            df = res.to_dataframe()
        except Exception as e:
            logger.warning(f"OpenBB fetch failed or not installed ({e}). Falling back to yfinance direct.")

        # 2. Fallback to direct yfinance (always 100% free)
        if df is None or df.empty:
            import yfinance as yf
            logger.info(f"Fetching {symbol} directly via yfinance...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, interval=interval)

        if df is None or df.empty:
            raise ValueError(f"No market data could be retrieved for symbol: {symbol}")

        # Normalize column names to quantitative standards
        df = self._standardize_dataframe(df)

        # Cache data locally
        if use_cache:
            try:
                df.to_parquet(cache_file)
            except Exception as e:
                logger.warning(f"Could not write cache file: {e}")

        return df

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lower-case dollar-prefixed standard ($open, $close, etc.)."""
        col_map = {}
        for col in df.columns:
            c_low = str(col).lower()
            if "open" in c_low:
                col_map[col] = "$open"
            elif "high" in c_low:
                col_map[col] = "$high"
            elif "low" in c_low:
                col_map[col] = "$low"
            elif "close" in c_low and "adj" not in c_low:
                col_map[col] = "$close"
            elif "volume" in c_low:
                col_map[col] = "$volume"

        df = df.rename(columns=col_map)
        
        # Ensure index is datetime and sorted
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        required_cols = ["$open", "$high", "$low", "$close", "$volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Standardized DataFrame missing required columns: {missing}")

        return df[required_cols]
