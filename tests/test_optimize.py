"""Tests for Backtest.optimize (grid and SAMBO)."""

from __future__ import annotations

import inspect
import sys
import time

import numpy as np
import pandas as pd
import pytest

from tempusonebt import Backtest
from tempusonebt.datasets import GOOG

from .conftest import SmaCross, tempfile_html


def test_optimize():
    bt = Backtest(GOOG.iloc[:100], SmaCross)
    opt_params = {'fast': range(2, 5, 2), 'slow': [2, 5, 7, 9]}

    with pytest.raises(ValueError):
        bt.optimize()
    with pytest.raises(ValueError):
        bt.optimize(maximize='missing key', **opt_params)
    with pytest.raises(TypeError):
        bt.optimize(maximize=15, **opt_params)
    with pytest.raises(TypeError):
        bt.optimize(constraint=15, **opt_params)
    with pytest.raises(ValueError):
        bt.optimize(constraint=lambda d: False, **opt_params)
    with pytest.raises(ValueError):
        bt.optimize(return_optimization=True, **opt_params)

    res = bt.optimize(**opt_params)
    assert isinstance(res, pd.Series)

    default_maximize = inspect.signature(Backtest.optimize).parameters['maximize'].default
    res2 = bt.optimize(**opt_params, maximize=lambda s: s[default_maximize])
    assert res.filter(regex='^[^_]').fillna(-1).to_dict() == res2.filter(regex='^[^_]').fillna(-1).to_dict()

    _, heatmap = bt.optimize(
        **opt_params, return_heatmap=True, constraint=lambda d: d.slow > 2 * d.fast
    )
    assert isinstance(heatmap, pd.Series)
    assert len(heatmap) == 4
    assert heatmap.name == default_maximize

    with tempfile_html() as f:
        bt.plot(filename=f, open_browser=False)


def test_method_sambo():
    bt = Backtest(GOOG.iloc[:100], SmaCross, finalize_trades=True)
    res, heatmap, sambo_results = bt.optimize(
        fast=range(2, 20),
        slow=np.arange(2, 20, dtype=object),
        constraint=lambda p: p.fast < p.slow,
        max_tries=30,
        method='sambo',
        return_optimization=True,
        return_heatmap=True,
        random_state=2,
    )
    assert isinstance(res, pd.Series)
    assert isinstance(heatmap, pd.Series)
    assert heatmap.max() > 1.1
    assert heatmap.min() > -2
    assert -sambo_results.fun == heatmap.max()
    assert heatmap.index.tolist() == heatmap.dropna().index.unique().tolist()


@pytest.mark.parametrize(
    ('method', 'max_tries', 'random_state'),
    [('grid', 5, 0), ('grid', 0.3, 0), ('sambo', 6, 0), ('sambo', 0.42, 0)],
)
def test_max_tries(method, max_tries, random_state):
    bt = Backtest(GOOG.iloc[:100], SmaCross)
    opt_params = {'fast': range(2, 10, 2), 'slow': [2, 5, 7, 9]}
    _, heatmap = bt.optimize(
        max_tries=max_tries,
        method=method,
        random_state=random_state,
        return_heatmap=True,
        **opt_params,
    )
    assert len(heatmap) == 6


def test_optimize_invalid_param():
    bt = Backtest(GOOG.iloc[:100], SmaCross)
    with pytest.raises(AttributeError):
        bt.optimize(foo=range(3))
    with pytest.raises(ValueError):
        bt.optimize(fast=[])


def test_optimize_no_trades():
    bt = Backtest(GOOG, SmaCross)
    stats = bt.optimize(fast=[3], slow=[3])
    assert stats.isnull().any()


def test_optimize_speed():
    bt = Backtest(GOOG.iloc[:100], SmaCross)
    start = time.process_time()
    bt.optimize(fast=range(2, 20, 2), slow=range(10, 40, 2))
    end = time.process_time()
    handicap = 5 if 'win' in sys.platform else 0.1
    assert end - start < 0.3 + handicap
