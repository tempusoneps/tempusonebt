"""Tests for internal utilities and data structures."""

from __future__ import annotations

from concurrent.futures.process import ProcessPoolExecutor

import pandas as pd

from tempusonebt import Backtest, Strategy
from tempusonebt._util import _Array, _as_str, _Indicator, patch
from tempusonebt.datasets import GOOG

from .conftest import SHORT_DATA, SmaCross, tempfile_html


def test_as_str():
    def func():
        pass

    class Class:
        def __call__(self):
            pass

    assert _as_str('4') == '4'
    assert _as_str(4) == '4'
    assert _as_str(_Indicator([1, 2], name='x')) == 'x'
    assert _as_str(func) == 'func'
    assert _as_str(Class) == 'Class'
    assert _as_str(Class()) == 'Class'
    assert _as_str(pd.Series([1, 2], name='x')) == 'x'
    assert _as_str(pd.DataFrame()) == 'df'
    assert _as_str(lambda x: x) == 'λ'
    for s in ('Open', 'High', 'Low', 'Close', 'Volume'):
        assert _as_str(_Array([1], name=s)) == s[0]


def test_patch():
    class Obj:
        pass

    o = Obj()
    o.attr = False
    with patch(o, 'attr', True):
        assert o.attr is True
    assert o.attr is False


def test_pandas_accessors():
    class S(Strategy):
        def init(self):
            close, index = self.data.Close, self.data.index
            assert close.s.equals(pd.Series(close, index=index))
            assert self.data.df['Close'].equals(pd.Series(close, index=index))
            self.data.df['new_key'] = 2 * close

        def next(self):
            close, index = self.data.Close, self.data.index
            assert close.s.equals(pd.Series(close, index=index))
            assert self.data.df['Close'].equals(pd.Series(close, index=index))
            assert self.data.df['new_key'].equals(pd.Series(self.data.new_key, index=index))

    Backtest(GOOG.iloc[:20], S).run()


def test_indicators_picklable():
    bt = Backtest(SHORT_DATA, SmaCross)
    with ProcessPoolExecutor() as executor:
        stats = executor.submit(Backtest.run, bt).result()
    assert stats._strategy._indicators[0]._opts
    with tempfile_html() as f:
        bt.plot(results=stats, filename=f, resample='2D', open_browser=False)
