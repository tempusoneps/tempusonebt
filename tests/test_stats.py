"""Tests for statistics computation and drawdown metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tempusonebt import Backtest, Strategy
from tempusonebt._stats import compute_drawdown_duration_peaks
from tempusonebt.datasets import GOOG

from .conftest import BaseStrategy, SmaCross


def test_compute_drawdown():
    dd = pd.Series([0, 1, 7, 0, 4, 0, 0])
    durations, peaks = compute_drawdown_duration_peaks(dd)
    np.testing.assert_array_equal(durations, pd.Series([3, 2], index=[3, 5]).reindex(dd.index))
    np.testing.assert_array_equal(peaks, pd.Series([7, 4], index=[3, 5]).reindex(dd.index))


def test_compute_stats():
    stats = Backtest(GOOG, SmaCross, finalize_trades=True).run()
    expected = pd.Series({
        '# Trades': 66,
        'Avg. Drawdown Duration': pd.Timedelta('41 days 00:00:00'),
        'Avg. Drawdown [%]': -5.925851581948801,
        'Avg. Trade Duration': pd.Timedelta('46 days 00:00:00'),
        'Avg. Trade [%]': 2.531715975158555,
        'Best Trade [%]': 53.59595229490424,
        'Buy & Hold Return [%]': 522.0601851851852,
        'Calmar Ratio': 0.4414380935608377,
        'Duration': pd.Timedelta('3116 days 00:00:00'),
        'End': pd.Timestamp('2013-03-01 00:00:00'),
        'Equity Final [$]': 51422.98999999996,
        'Equity Peak [$]': 75787.44,
        'Expectancy [%]': 3.2748078066748834,
        'Exposure Time [%]': 96.74115456238361,
        'Max. Drawdown Duration': pd.Timedelta('584 days 00:00:00'),
        'Max. Drawdown [%]': -47.98012705007589,
        'Max. Trade Duration': pd.Timedelta('183 days 00:00:00'),
        'Profit Factor': 1.4778410020970645,
        'Return (Ann.) [%]': 21.180255813792282,
        'Return [%]': 414.2298999999996,
        'Volatility (Ann.) [%]': 36.49390889140787,
        'CAGR [%]': 21.160245705962623,
        'SQN': 1.0766187356697705,
        'Kelly Criterion': 0.1518705127029717,
        'Sharpe Ratio': 0.5803778344714113,
        'Sortino Ratio': 1.0847880675854096,
        'Start': pd.Timestamp('2004-08-19 00:00:00'),
        'Win Rate [%]': 46.96969696969697,
        'Worst Trade [%]': -18.39887353835481,
        'Alpha [%]': 393.8351032363473,
        'Beta': 0.03906598768189504,
    })

    def almost_equal(a, b):
        try:
            return bool(np.isclose(a, b, rtol=1e-8))
        except TypeError:
            return bool(a == b)

    diff = {
        key: value
        for key, value in stats.filter(regex='^[^_]').items()
        if not almost_equal(value, expected[key])
    }
    assert diff == {}

    assert sorted(stats['_equity_curve'].columns) == sorted(
        ['Equity', 'DrawdownPct', 'DrawdownDuration']
    )
    assert len(stats['_trades']) == 66

    indicator_columns = [
        f'{entry}_SMA(C,{n})' for entry in ('Entry', 'Exit') for n in (SmaCross.fast, SmaCross.slow)
    ]
    assert sorted(stats['_trades'].columns) == sorted(
        [
            'Size',
            'EntryBar',
            'ExitBar',
            'EntryPrice',
            'ExitPrice',
            'SL',
            'TP',
            'PnL',
            'ReturnPct',
            'EntryTime',
            'ExitTime',
            'Duration',
            'Tag',
            'Commission',
            *indicator_columns,
        ]
    )


@pytest.mark.parametrize('strategy_kind', ['sma', 'single_trade', 'single_position', 'no_trade'])
def test_compute_stats_bordercase(strategy_kind):
    class SingleTrade(Strategy):
        def init(self):
            self._done = False

        def next(self):
            if not self._done:
                self.buy()
                self._done = True
            if self.position:
                self.position.close()

    class SinglePosition(BaseStrategy):
        def next(self):
            if not self.position:
                self.buy()

    class NoTrade(BaseStrategy):
        def next(self):
            pass

    strategies = {
        'sma': SmaCross,
        'single_trade': SingleTrade,
        'single_position': SinglePosition,
        'no_trade': NoTrade,
    }
    strategy = strategies[strategy_kind]
    stats = Backtest(GOOG.iloc[:100], strategy).run()
    assert not np.isnan(stats['Equity Final [$]'])
    assert not stats['_equity_curve']['Equity'].isnull().any()
    assert stats['_strategy'].__class__ is strategy
