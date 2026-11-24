"""Tests for Backtest engine initialization, execution, and commissions."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tempusonebt import Backtest, Strategy
from tempusonebt._util import _Indicator, try_
from tempusonebt.datasets import EURUSD, GOOG, SMA
from tempusonebt.lib import crossover, resample_apply

from .conftest import SHORT_DATA, BaseStrategy, SmaCross, tempfile_html


def test_run():
    bt = Backtest(EURUSD, SmaCross)
    bt.run()


def test_run_invalid_param():
    bt = Backtest(GOOG, SmaCross)
    with pytest.raises(AttributeError):
        bt.run(foo=3)


def test_run_speed():
    bt = Backtest(GOOG, SmaCross)
    start = time.process_time()
    bt.run()
    end = time.process_time()
    assert end - start < 0.3


def test_data_missing_columns():
    df = GOOG.copy(deep=False)
    del df['Open']
    with pytest.raises(ValueError):
        Backtest(df, SmaCross).run()


def test_data_nan_columns():
    df = GOOG.copy()
    df['Open'] = np.nan
    with pytest.raises(ValueError):
        Backtest(df, SmaCross).run()


def test_data_extra_columns():
    df = GOOG.copy(deep=False)
    df['P/E'] = np.arange(len(df))
    df['MCap'] = np.arange(len(df))

    class S(Strategy):
        def init(self):
            assert len(self.data.MCap) == len(self.data.Close)
            assert len(self.data['P/E']) == len(self.data.Close)

        def next(self):
            assert len(self.data.MCap) == len(self.data.Close)
            assert len(self.data['P/E']) == len(self.data.Close)

    Backtest(df, S).run()


def test_data_invalid():
    with pytest.raises(TypeError):
        Backtest(GOOG.index, SmaCross).run()
    with pytest.raises(ValueError):
        Backtest(GOOG.iloc[:0], SmaCross).run()


def test_strategy_abstract():
    class MyStrategy(Strategy):
        pass

    with pytest.raises(TypeError):
        MyStrategy(None, None)


def test_strategy_str():
    bt = Backtest(GOOG.iloc[:100], SmaCross)
    assert str(bt.run()._strategy) == SmaCross.__name__
    assert str(bt.run(fast=11)._strategy) == f'{SmaCross.__name__}(fast=11)'


def test_dont_overwrite_data():
    df = EURUSD.copy()
    bt = Backtest(df, SmaCross)
    bt.run()
    bt.optimize(fast=4, slow=[6, 8])
    with tempfile_html() as f:
        bt.plot(filename=f, plot_drawdown=True, open_browser=False)
    assert df.equals(EURUSD)


def test_assertions():
    class Assertive(Strategy):
        def init(self):
            self.sma = self.I(SMA, self.data.Close, 10)
            self.remains_indicator = np.r_[2] * np.cumsum(self.sma * 5 + 1) * np.r_[2]
            self.transpose_invalid = self.I(
                lambda: np.column_stack((self.data.Open, self.data.Close))
            )

            resampled = resample_apply('W', SMA, self.data.Close, 3)
            resampled_ind = resample_apply('W', SMA, self.sma, 3)
            assert np.unique(resampled[-5:]).size == 1
            assert np.unique(resampled[-6:]).size == 2
            assert resampled in self._indicators
            assert resampled_ind in self._indicators

            assert try_(lambda: self.data.X, 1, AttributeError) == 1
            assert try_(lambda: self.data['X'], 1, KeyError) == 1
            assert self.data.pip == 0.01
            assert float(self.data.Close) == self.data.Close[-1]

        def next(self, _few_days=pd.Timedelta('3 days')):
            assert self.equity >= 0
            assert isinstance(self.sma, _Indicator)
            assert isinstance(self.remains_indicator, _Indicator)
            assert self.remains_indicator.name
            assert isinstance(self.remains_indicator._opts, dict)

            if crossover(self.sma, self.data.Close):
                for order in self.orders:
                    if not order.is_contingent:
                        order.cancel()
                price = self.data.Close[-1]
                sl, tp = 1.05 * price, 0.9 * price
                n_orders = len(self.orders)
                self.sell(size=0.21, limit=price, stop=price, sl=sl, tp=tp)
                assert len(self.orders) == n_orders + 1
                order = self.orders[-1]
                assert order.limit == price
                assert order.stop == price
                assert order.size == -0.21
                assert order.sl == sl
                assert order.tp == tp
                assert not order.is_contingent
            elif self.position:
                assert not self.position.is_long
                assert self.position.is_short
                assert self.position.pl
                assert self.position.pl_pct
                assert self.position.size < 0
                trade = self.trades[0]
                if self.data.index[-1] - self.data.index[trade.entry_bar] > _few_days:
                    self.position.close(0.5)
                    self.position.close(0.5)
                    self.position.close(0.5)
                    self.position.close()
                    self.position.close()

    bt = Backtest(GOOG, Assertive)
    with pytest.warns(UserWarning):
        stats = bt.run()
    assert stats['# Trades'] == 131


def test_broker_params():
    bt = Backtest(
        GOOG.iloc[:100], SmaCross, cash=1000, spread=0.01, margin=0.1, trade_on_close=True
    )
    bt.run()


def test_spread_commission():
    class S(Strategy):
        def init(self):
            self.done = False

        def next(self):
            if not self.position:
                self.buy()
            else:
                self.position.close()
                self.next = lambda: None

    spread = 0.01
    commission = 0.01
    cash = 10_000
    order_bar = 2

    stats = Backtest(SHORT_DATA, S, cash=cash, spread=spread, commission=commission).run()
    trade_open_price = SHORT_DATA['Open'].iloc[order_bar]
    assert stats['_trades']['EntryPrice'].iloc[0] == trade_open_price * (1 + spread)
    assert stats['_equity_curve']['Equity'].iloc[2:4].round(2).tolist() == [9685.31, 9749.33]

    stats = Backtest(SHORT_DATA, S, cash=cash, commission=(100, commission)).run()
    assert stats['_equity_curve']['Equity'].iloc[2:4].round(2).tolist() == [9683.74, 9647.77]

    stats = Backtest(SHORT_DATA, S, cash=cash, commission=lambda size, price: size * price * commission).run()
    assert stats['_equity_curve']['Equity'].iloc[2:4].round(2).tolist() == [9781.28, 9846.04]


def test_commissions():
    fixed_commission, commission = 10, 0.01
    cash, size, price_entry, price_exit = 5000, 100, 1, 4

    class S(BaseStrategy):
        def next(self):
            if len(self.data) == 2:
                self.buy(size=size, tp=3)

    arr = np.r_[1, price_entry, 1, 2, price_exit, 1, 2]
    df = pd.DataFrame({'Open': arr, 'High': arr, 'Low': arr, 'Close': arr})
    with pytest.warns(UserWarning, match='index is not datetime'):
        stats = Backtest(df, S, cash=cash, commission=(fixed_commission, commission)).run()

    expected_paid_commission = (
        fixed_commission
        + commission * size * price_entry
        + fixed_commission
        + commission * size * price_exit
    )
    assert stats['Commissions [$]'] == expected_paid_commission
    assert stats._trades['Commission'][0] == expected_paid_commission
    assert stats['Equity Final [$]'] == cash + (price_exit - price_entry) * size - expected_paid_commission


def test_backtest_run_docstring_and_readme_contain_stats_keys():
    stats = Backtest(SHORT_DATA, SmaCross).run()
    readme = (Path(__file__).parent.parent / 'README.md').read_text(encoding='utf-8')
    for key in stats.index:
        assert key in Backtest.run.__doc__
        assert key in readme
