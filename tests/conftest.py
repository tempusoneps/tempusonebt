"""Shared fixtures and helpers for tempusonebt tests."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from tempfile import NamedTemporaryFile

from tempusonebt import Backtest, Strategy
from tempusonebt._util import try_
from tempusonebt.datasets import GOOG, SMA
from tempusonebt.lib import crossover

SHORT_DATA = GOOG.iloc[:20]


@contextmanager
def tempfile_html():
    with NamedTemporaryFile(suffix='.html') as f:
        if sys.platform.startswith('win'):
            f.close()
        yield f.name


class SmaCross(Strategy):
    fast = 10
    slow = 30

    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.fast)
        self.sma2 = self.I(SMA, self.data.Close, self.slow)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell()


class BaseStrategy(Strategy):
    def init(self):
        super().init()


def make_coroutine_backtest(strategy_coroutine, data=SHORT_DATA, **kwargs):
    class CoroutineStrategy(Strategy):
        def init(self):
            self.step = strategy_coroutine(self)

        def next(self):
            try_(self.step.__next__, None, StopIteration)

    return Backtest(data, CoroutineStrategy, **kwargs)
