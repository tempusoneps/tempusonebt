"""Tests for tempusonebt.lib utilities and composable strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from tempusonebt import Backtest
from tempusonebt.datasets import BTCUSD, EURUSD, GOOG, SMA
from tempusonebt.lib import (
    FractionalBacktest,
    SignalStrategy,
    TrailingStrategy,
    barssince,
    compute_stats,
    cross,
    crossover,
    plot_heatmaps,
    quantile,
    random_ohlc_data,
    resample_apply,
)

from .conftest import SmaCross, tempfile_html


def test_barssince():
    assert barssince(np.r_[1, 0, 0]) == 2
    assert barssince(np.r_[0, 0, 0]) == np.inf
    assert barssince(np.r_[0, 0, 0], 0) == 0


def test_cross():
    assert cross([0, 1], [1, 0])
    assert cross([1, 0], [0, 1])
    assert not cross([1, 0], [1, 0])


def test_crossover():
    assert crossover([0, 1], [1, 0])
    assert crossover([0, 1], 0.5)
    assert crossover([0, 1], pd.Series([0.5, 0.5], index=[5, 6]))
    assert not crossover([1, 0], [1, 0])
    assert not crossover([0], [1])


def test_quantile():
    assert quantile(np.r_[1, 3, 2], 0.5) == 2
    assert quantile(np.r_[1, 3, 2]) == 0.5


def test_resample_apply():
    res = resample_apply('D', SMA, EURUSD.Close, 10)
    assert res.name == 'C[D]'
    assert res.count() / res.size == 0.9634
    np.testing.assert_almost_equal(
        res.iloc[-48:].unique().tolist(), [1.242643, 1.242381, 1.242275], decimal=6
    )

    def resets_index(*args):
        return pd.Series(SMA(*args).values)

    res2 = resample_apply('D', resets_index, EURUSD.Close, 10)
    assert (res.dropna() == res2.dropna()).all()
    assert (res.index == res2.index).all()

    res3 = resample_apply('D', None, EURUSD)
    assert 'Volume' in res3

    res4 = resample_apply('D', lambda df: (df.Close, df.Close), EURUSD)
    assert isinstance(res4, pd.DataFrame)


def test_plot_heatmaps():
    bt = Backtest(GOOG, SmaCross)
    _, heatmap = bt.optimize(fast=range(2, 7, 2), slow=range(7, 15, 2), return_heatmap=True)
    with tempfile_html() as f:
        for agg in ('mean', lambda x: np.percentile(x, 75)):
            plot_heatmaps(heatmap, agg, filename=f, open_browser=False)


def test_random_ohlc_data():
    generator = random_ohlc_data(GOOG, frac=1)
    new_data = next(generator)
    assert list(new_data.index) == list(GOOG.index)
    assert new_data.shape == GOOG.shape
    assert list(new_data.columns) == list(GOOG.columns)


def test_compute_stats():
    stats = Backtest(GOOG, SmaCross).run()
    only_long_trades = stats._trades[stats._trades.Size > 0]
    long_stats = compute_stats(
        stats=stats, trades=only_long_trades, data=GOOG, risk_free_rate=0.02
    )
    assert list(stats._equity_curve.Equity) != list(long_stats._equity_curve.Equity)
    assert stats['Sharpe Ratio'] != long_stats['Sharpe Ratio']
    assert long_stats['# Trades'] == len(only_long_trades)
    assert stats._strategy == long_stats._strategy
    assert_frame_equal(long_stats._trades, only_long_trades)
    first_exit = only_long_trades['ExitBar'].min()
    assert (
        long_stats._equity_curve.Equity.iloc[:first_exit] == stats._equity_curve.Equity.iloc[0]
    ).all()


def test_signal_strategy():
    class S(SignalStrategy):
        def init(self):
            sma = self.data.Close.s.rolling(10).mean()
            self.set_signal(self.data.Close > sma, self.data.Close < sma)

    stats = Backtest(GOOG, S).run()
    assert stats['# Trades'] in (1179, 1180)


def test_trailing_strategy():
    class S(TrailingStrategy):
        def init(self):
            super().init()
            self.set_atr_periods(40)
            self.set_trailing_pct(0.1)
            self.set_trailing_sl(3)
            self.sma = self.I(lambda: self.data.Close.s.rolling(10).mean())

        def next(self):
            super().next()
            if not self.position and self.data.Close > self.sma:
                self.buy()

    stats = Backtest(GOOG, S).run()
    assert stats['# Trades'] == 56


def test_fractional_backtest():
    ubtc_bt = FractionalBacktest(BTCUSD['2015':], SmaCross, fractional_unit=1 / 1e6, cash=100)
    stats = ubtc_bt.run(fast=2, slow=3)
    assert stats['# Trades'] == 41
    trades = stats['_trades']
    assert len(trades) == 41
    trade = trades.iloc[0]
    assert np.isclose(trade['EntryPrice'], 236.69)
    assert np.isclose(stats['_strategy']._indicators[0][trade['EntryBar']], 234.14)

    no_vol = BTCUSD['2015':].drop(columns=['Volume'])
    stats_no_vol = FractionalBacktest(
        no_vol, SmaCross, fractional_unit=1 / 1e6, cash=100
    ).run(fast=2, slow=3)
    assert stats_no_vol['# Trades'] == 41
