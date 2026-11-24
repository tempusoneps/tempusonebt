"""Tests for Broker order execution, Position, Trade, hedging, and SL/TP."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tempusonebt import Backtest, Strategy
from tempusonebt.datasets import GOOG, SMA
from tempusonebt.lib import crossover

from .conftest import SHORT_DATA, BaseStrategy, make_coroutine_backtest


def test_position():
    def coroutine(self):
        yield self.buy()

        assert self.position
        assert self.position.is_long
        assert not self.position.is_short
        assert self.position.size > 0
        assert self.position.pl
        assert self.position.pl_pct
        assert np.isclose(self.position.pl_pct, self.trades[0].pl_pct)

        yield self.position.close()

        assert not self.position
        assert not self.position.is_long
        assert not self.position.is_short
        assert not self.position.size
        assert not self.position.pl
        assert not self.position.pl_pct

    make_coroutine_backtest(coroutine).run()


def test_broker_hedging():
    def coroutine(self):
        yield self.buy(size=2)
        assert len(self.trades) == 1
        yield self.sell(size=1)
        assert len(self.trades) == 2

    make_coroutine_backtest(coroutine, hedging=True).run()


def test_broker_exclusive_orders():
    def coroutine(self):
        yield self.buy(size=2)
        assert len(self.trades) == 1
        yield self.sell(size=3)
        assert len(self.trades) == 1
        assert self.trades[0].size == -3

    make_coroutine_backtest(coroutine, exclusive_orders=True).run()


def test_trade_multiple_close():
    def coroutine(self):
        yield self.buy()
        assert self.trades
        self.trades[-1].close(1)
        self.trades[-1].close(0.1)
        yield

    make_coroutine_backtest(coroutine).run()


def test_close_trade_leaves_needsize_0():
    def coroutine(self):
        self.buy(size=1)
        self.buy(size=1)
        yield
        if self.position:
            self.sell(size=1)

    make_coroutine_backtest(coroutine).run()


def test_stop_limit_order_price_is_stop_price():
    def coroutine(self):
        self.buy(stop=112, limit=113, size=1)
        self.sell(stop=107, limit=105, size=1)
        yield

    stats = make_coroutine_backtest(coroutine).run()
    assert stats._trades.filter(like='Price').stack().tolist() == [112, 107]


def test_autoclose_trades_on_finish():
    def coroutine(self):
        yield self.buy()

    stats = make_coroutine_backtest(coroutine, finalize_trades=True).run()
    assert len(stats._trades) == 1


def test_order_tag():
    def coroutine(self):
        yield self.buy(size=2, tag=1)
        yield self.sell(size=1, tag='s')
        yield self.sell(size=1)
        yield self.buy(tag=2)
        yield self.position.close()

    stats = make_coroutine_backtest(coroutine).run()
    assert list(stats._trades.Tag) == [1, 1, 2]


def test_trade_enter_hit_sl_on_same_day():
    the_day = pd.Timestamp('2012-10-17 00:00:00')

    class S1(BaseStrategy):
        def next(self):
            if self.data.index[-1] == the_day:
                self.buy(sl=720)

    assert Backtest(GOOG, S1).run()._trades.iloc[0].ExitPrice == 720

    class S2(BaseStrategy):
        def next(self):
            if self.data.index[-1] == the_day:
                self.buy(stop=758, sl=720)

    with pytest.warns(UserWarning):
        assert Backtest(GOOG, S2).run()._trades.iloc[0].ExitPrice == 705.58


def test_stop_price_between_sl_tp():
    class S(BaseStrategy):
        def next(self):
            if self.data.index[-1] == pd.Timestamp('2004-09-09 00:00:00'):
                self.buy(stop=104, sl=103, tp=110)

    with pytest.warns(UserWarning):
        assert Backtest(GOOG, S).run()._trades.iloc[0].EntryPrice == 104


def test_position_close_portion():
    class SmaCross1020(Strategy):
        def init(self):
            self.sma1 = self.I(SMA, self.data.Close, 10)
            self.sma2 = self.I(SMA, self.data.Close, 20)

        def next(self):
            if not self.position and crossover(self.sma1, self.sma2):
                self.buy(size=10)
            if self.position and crossover(self.sma2, self.sma1):
                self.position.close(portion=0.5)

    Backtest(GOOG, SmaCross1020, spread=0.002).run()


def test_close_orders_from_last_strategy_iteration():
    class S(BaseStrategy):
        def next(self):
            if not self.position:
                self.buy()
            elif len(self.data) == len(SHORT_DATA):
                self.position.close()

    with pytest.warns(UserWarning, match='finalize_trades'):
        assert Backtest(SHORT_DATA, S, finalize_trades=False).run()._trades.empty
    assert not Backtest(SHORT_DATA, S, finalize_trades=True).run()._trades.empty


def test_check_adjusted_price_when_placing_order():
    class S(BaseStrategy):
        def next(self):
            self.buy(tp=self.data.Close * 1.01)

    with pytest.raises(ValueError):
        Backtest(SHORT_DATA, S, spread=0.02).run()
