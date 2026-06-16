"""yfinance price layer — one fetch shared by the technical and Markov lenses."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf

import config


@dataclass
class MarketData:
    ticker: str
    sector: str | None
    sector_etf: str
    price: float | None
    shares_outstanding: float | None
    market_cap: float | None
    info: dict
    close: pd.Series           # adjusted close, ~6y daily
    spy_close: pd.Series
    sector_close: pd.Series

    @property
    def has_history(self) -> bool:
        return self.close is not None and len(self.close.dropna()) > 60


def _history(ticker: str, period: str = "6y") -> pd.Series:
    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if hist is None or hist.empty:
            return pd.Series(dtype="float64")
        return hist["Close"].dropna()
    except Exception:
        return pd.Series(dtype="float64")


def fetch_market_data(ticker: str) -> MarketData:
    t = yf.Ticker(ticker)
    info: dict = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    close = _history(ticker)
    sector = info.get("sector")
    etf = config.sector_etf(sector)

    spy_close = _history(config.BENCHMARK)
    sector_close = spy_close if etf == config.BENCHMARK else _history(etf)

    price = float(close.iloc[-1]) if len(close) else info.get("currentPrice")
    shares = info.get("sharesOutstanding")
    mcap = info.get("marketCap")
    if mcap is None and price and shares:
        mcap = price * shares

    return MarketData(
        ticker=ticker.upper(),
        sector=sector,
        sector_etf=etf,
        price=price,
        shares_outstanding=shares,
        market_cap=mcap,
        info=info,
        close=close,
        spy_close=spy_close,
        sector_close=sector_close,
    )
