"""Compatibility facade re-exporting core backtesting classes."""

from __future__ import annotations

from .engine import Backtest
from .models import Order, Position, Trade
from .strategy import Strategy

__all__ = [
    'Backtest',
    'Order',
    'Position',
    'Strategy',
    'Trade',
]
