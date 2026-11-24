"""Regression tests for bug fixes and edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tempusonebt import Backtest
from tempusonebt.datasets import EURUSD, GOOG
from tempusonebt.lib import OHLCV_AGG

from .conftest import SHORT_DATA, BaseStrategy, SmaCross, make_coroutine_backtest


def test_gh_521():
    class S(BaseStrategy):
        def next(self):
            if self.data.Close[-1] == 100:
                self.buy(size=1, sl=90)

    arr = np.r_[100, 100, 100, 50, 50]
    df = pd.DataFrame({'Open': arr, 'High': arr, 'Low': arr, 'Close': arr})
    with pytest.warns(UserWarning, match='index is not datetime'):
        bt = Backtest(df, S, cash=100, trade_on_close=True)
    assert bt.run()._trades['ExitPrice'][0] == 50


def test_stats_annualized():
    stats = Backtest(GOOG.resample('W').agg(OHLCV_AGG), SmaCross).run()
    assert not np.isnan(stats['Return (Ann.) [%]'])
    assert round(stats['Return (Ann.) [%]']) == -3


def test_cancel_orders():
    class S(BaseStrategy):
        def next(self):
            self.buy(sl=1, tp=1e3)
            if self.position:
                self.position.close()
                for order in self.orders:
                    order.cancel()

    Backtest(SHORT_DATA, S).run()


def test_trade_on_close_closes_trades_on_close():
    def coro(strat):
        yield strat.buy(size=1, sl=90) and strat.buy(size=1, sl=80)
        assert len(strat.trades) == 2
        yield strat.trades[0].close()
        yield

    arr = np.r_[100, 101, 102, 50, 51]
    df = pd.DataFrame({'Open': arr - 10, 'Close': arr, 'High': arr, 'Low': arr})
    with pytest.warns(UserWarning, match='index is not datetime'):
        trades = make_coroutine_backtest(coro, df, cash=250, trade_on_close=True).run()._trades
        assert trades['EntryBar'].tolist() == [1, 1]
        assert trades['ExitBar'].tolist() == [2, 3]
        assert trades['EntryPrice'].tolist() == [101, 101]
        assert trades['ExitPrice'].tolist() == [102, 40]

    with pytest.warns(UserWarning, match='index is not datetime'):
        trades = make_coroutine_backtest(coro, df, cash=250, trade_on_close=False).run()._trades
        assert trades['EntryBar'].tolist() == [2, 2]
        assert trades['ExitBar'].tolist() == [3, 3]
        assert trades['EntryPrice'].tolist() == [92, 92]
        assert trades['ExitPrice'].tolist() == [40, 40]


def test_trades_dates_match_prices():
    bt = Backtest(EURUSD, SmaCross, trade_on_close=True)
    trades = bt.run()._trades
    assert EURUSD.Close[trades['ExitTime']].tolist() == trades['ExitPrice'].tolist()


def test_sl_always_before_tp():
    class S(BaseStrategy):
        def next(self):
            i = len(self.data.index)
            if i == 4:
                self.buy()
            if i == 5:
                t = self.trades[0]
                t.sl = 105
                t.tp = 107.9

    trades = Backtest(SHORT_DATA, S).run()._trades
    assert trades['ExitPrice'].iloc[0] == 104.95


def test_stop_entry_and_tp_in_same_bar():
    class S(BaseStrategy):
        def next(self):
            if len(self.data.index) == 3:
                self.sell(stop=108, tp=105, sl=113)

    trades = Backtest(SHORT_DATA, S).run()._trades
    assert trades['ExitBar'].iloc[0] == 3
    assert trades['ExitPrice'].iloc[0] == 105


def test_optimize_datetime_index_with_timezone():
    data = GOOG.iloc[:100].copy()
    data.index = data.index.tz_localize('Asia/Kolkata')
    res = Backtest(data, SmaCross).optimize(fast=range(2, 3), slow=range(4, 5))
    assert res['# Trades'] > 0


def test_sl_tp_values_in_trades_df():
    class S(BaseStrategy):
        def next(self):
            self.next = lambda: None
            self.buy(size=1, tp=111)
            self.buy(size=1, sl=99)

    trades = Backtest(SHORT_DATA, S).run()._trades
    assert trades['SL'].fillna(0).tolist() == [0, 99]
    assert trades['TP'].fillna(0).tolist() == [111, 0]


def test_exclusive_orders_cancels_all_pending():
    class S(BaseStrategy):
        def next(self):
            if len(self.data) == 2:
                self.buy(limit=1, size=1)
                self.buy(limit=2, size=1)
                self.buy(limit=3, size=1)
                assert len(self.orders) == 1

    Backtest(SHORT_DATA, S, exclusive_orders=True).run()


def test_out_of_money_closes_all_trades():
    class S(BaseStrategy):
        def next(self):
            if len(self.data) == 2:
                self.buy(size=10)
                self.buy(size=10)
                self.buy(size=10)

    arr = np.r_[100, 100, 100, 10, 10]
    df = pd.DataFrame({'Open': arr, 'High': arr, 'Low': arr, 'Close': arr})
    with pytest.warns(UserWarning, match='index is not datetime'):
        stats = Backtest(df, S, cash=1000, margin=0.2).run()
    assert stats['Equity Final [$]'] == 0
    assert stats['# Trades'] == 3


def test_sl_preserved_when_gapped_through():
    class S(BaseStrategy):
        def next(self):
            if len(self.data) == 2:
                self.buy(size=1, sl=95)

    arr = np.r_[100, 100, 100, 80, 80]
    df = pd.DataFrame({'Open': arr, 'High': arr, 'Low': arr, 'Close': arr})
    with pytest.warns(UserWarning, match='index is not datetime'):
        trades = Backtest(df, S, cash=1000).run()._trades
    assert trades['ExitPrice'].iloc[0] == 80
    assert trades['SL'].iloc[0] == 95


def test_partial_close_fixed_commission_matches_equity():
    class S(BaseStrategy):
        def next(self):
            if len(self.data) == 2:
                self.buy(size=10)
            elif len(self.data) == 3:
                self.trades[0].close(0.5)
            elif len(self.data) == 4:
                self.trades[0].close(1.0)

    arr = np.r_[100, 100, 100, 110, 120, 120]
    df = pd.DataFrame({'Open': arr, 'High': arr, 'Low': arr, 'Close': arr})
    with pytest.warns(UserWarning, match='index is not datetime'):
        stats = Backtest(df, S, cash=5000, commission=(10, 0.01)).run()
    assert stats['Equity Final [$]'] - 5000 == pytest.approx(stats['_trades']['PnL'].sum())


def test_kelly_criterion_edge_cases():
    class S(BaseStrategy):
        def next(self):
            if len(self.data) == 2:
                self.buy(size=1)
            elif len(self.data) == 4:
                self.position.close()

    arr_up = np.r_[100, 100, 100, 110, 120, 120]
    arr_dn = np.r_[100, 100, 100, 90, 80, 80]
    with pytest.warns(UserWarning, match='index is not datetime'):
        stats_win = Backtest(
            pd.DataFrame({'Open': arr_up, 'High': arr_up, 'Low': arr_up, 'Close': arr_up}), S
        ).run()
        stats_loss = Backtest(
            pd.DataFrame({'Open': arr_dn, 'High': arr_dn, 'Low': arr_dn, 'Close': arr_dn}), S
        ).run()
    assert stats_win['Kelly Criterion'] == 1.0
    assert stats_loss['Kelly Criterion'] == 0.0
