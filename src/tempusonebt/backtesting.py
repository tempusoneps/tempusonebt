"""Compatibility facade re-exporting core backtesting classes."""

from __future__ import annotations

from .broker import _Broker
from .engine import Backtest
from .models import Order, Position, Trade, _OutOfMoneyError
from .strategy import Strategy

__all__ = [
    'Backtest',
    'Order',
    'Position',
    'Strategy',
    'Trade',
]
