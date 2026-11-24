"""Tests for Bokeh plotting output."""

from __future__ import annotations

import os

import pandas as pd
import pytest

import tempusonebt._plotting
from tempusonebt import Backtest, Strategy
from tempusonebt._util import patch
from tempusonebt.datasets import EURUSD, GOOG, SMA
from tempusonebt.lib import OHLCV_AGG

from .conftest import BaseStrategy, SmaCross, tempfile_html


def test_plot_before_run():
    bt = Backtest(GOOG, SmaCross)
    with pytest.raises(RuntimeError):
        bt.plot()


def test_file_size():
    bt = Backtest(GOOG, SmaCross)
    bt.run()
    with tempfile_html() as f:
        bt.plot(filename=f[: -len('.html')], open_browser=False)
        assert os.path.getsize(f) < 500_000


@pytest.mark.parametrize(
    ('param', 'value'),
    [
        ('plot_volume', False),
        ('plot_equity', False),
        ('plot_return', True),
        ('plot_pl', False),
        ('plot_drawdown', True),
        ('plot_trades', False),
        ('superimpose', False),
        ('resample', '1W'),
        ('smooth_equity', False),
        ('relative_equity', False),
        ('reverse_indicators', True),
        ('show_legend', False),
    ],
)
def test_params(param, value):
    bt = Backtest(GOOG.iloc[:100], SmaCross)
    bt.run()
    with tempfile_html() as f:
        bt.plot(**{param: value}, filename=f, open_browser=False)


@pytest.mark.parametrize('rule', 'ms s min h D W ME'.split())
def test_resolutions(rule):
    with tempfile_html() as f:
        df = EURUSD.iloc[:2].resample(rule).agg(OHLCV_AGG).dropna().iloc[:1100]
        bt = Backtest(df, SmaCross)
        bt.run()
        bt.plot(filename=f, open_browser=False)


def test_range_axis():
    df = GOOG.iloc[:100].reset_index(drop=True)
    with pytest.warns(UserWarning):
        bt = Backtest(df, SmaCross)
    bt.run()
    with tempfile_html() as f:
        bt.plot(filename=f, open_browser=False)


def test_wellknown():
    class S(BaseStrategy):
        def next(self):
            date = self.data.index[-1]
            if date == pd.Timestamp('Thu 19 Oct 2006'):
                self.buy(stop=484, limit=466, size=100)
            elif date == pd.Timestamp('Thu 30 Oct 2007'):
                self.position.close()
            elif date == pd.Timestamp('Tue 11 Nov 2008'):
                self.sell(stop=self.data.Low, limit=324.90, size=200)

    bt = Backtest(GOOG, S, margin=0.1)
    stats = bt.run()
    trades = stats['_trades']

    assert stats['Equity Peak [$]'] == pytest.approx(46961)
    assert stats['Equity Final [$]'] == 0
    assert len(trades) == 2
    assert trades[['EntryTime', 'ExitTime']].equals(
        pd.DataFrame(
            {
                'EntryTime': pd.to_datetime(['2006-11-01', '2008-11-14']),
                'ExitTime': pd.to_datetime(['2007-10-31', '2009-09-21']),
            }
        )
    )
    assert trades['PnL'].round().equals(pd.Series([23469.0, -34420.0]))

    with tempfile_html() as f:
        bt.plot(filename=f, plot_drawdown=True, smooth_equity=False, open_browser=False)


def test_resample():
    class S(SmaCross):
        def init(self):
            self.I(lambda: ['x'] * len(self.data))
            super().init()

    bt = Backtest(GOOG, S)
    bt.run()
    with tempfile_html() as f, patch(tempusonebt._plotting, '_MAX_CANDLES', 10):
        with pytest.warns(UserWarning):
            bt.plot(filename=f, resample=True, open_browser=False)


def test_indicator_name_color_scatter():
    class S(Strategy):
        def init(self):
            def _sma():
                return SMA(self.data.Close, 5), SMA(self.data.Close, 10)

            with pytest.raises(TypeError):
                self.I(_sma, name=42)
            with pytest.raises(ValueError):
                self.I(_sma, name=('SMA One',))
            with pytest.raises(ValueError):
                self.I(_sma, name=('SMA One', 'SMA Two', 'SMA Three'))

            a = self.I(SMA, self.data.Close, 5, overlay=True, color='red', scatter=True)
            b = self.I(SMA, self.data.Close, 10, overlay=False, color='blue', scatter=True)
            self.I(lambda: (a, b), overlay=False, color=('green', 'orange'))

        def next(self):
            pass

    bt = Backtest(GOOG, S)
    bt.run()
    with tempfile_html() as f:
        bt.plot(
            filename=f,
            plot_drawdown=False,
            plot_equity=False,
            plot_pl=False,
            plot_volume=False,
            open_browser=False,
        )
