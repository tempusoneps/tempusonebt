"""Sample OHLCV datasets and helper indicators."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).parent


def _read_file(filename: str) -> pd.DataFrame:
    return pd.read_csv(_DATA_DIR / filename, index_col=0, parse_dates=True)


BTCUSD = _read_file('BTCUSD.csv')
"""DataFrame of monthly BTC/USD historical index data from 2012 through 2024."""

GOOG = _read_file('GOOG.csv')
"""DataFrame of daily NASDAQ:GOOG stock price data from 2004 to 2013."""

EURUSD = _read_file('EURUSD.csv')
"""DataFrame of hourly EUR/USD forex data from April 2017 to February 2018."""


def SMA(arr: pd.Series, n: int) -> pd.Series:
    """Return `n`-period simple moving average of array `arr`."""
    return pd.Series(arr).rolling(n).mean()


__all__ = ['BTCUSD', 'EURUSD', 'GOOG', 'SMA']
