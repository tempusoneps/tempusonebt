"""Backtest engine for running, optimizing, and plotting trading strategies."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from copy import copy
from functools import lru_cache, partial
from itertools import product, repeat
from numbers import Number

import numpy as np
import pandas as pd
from numpy.random import default_rng

from ._plotting import plot  # noqa: I001
from ._stats import compute_stats, dummy_stats
from ._util import (
    SharedMemoryManager,
    _batch,
    _Data,
    _indicator_warmup_nbars,
    _strategy_indicators,
    _tqdm,
    patch,
    try_,
)
from .broker import _Broker
from .models import _OutOfMoneyError
from .strategy import Strategy


class Backtest:
    """
    Backtest a particular (parameterized) strategy on particular data.

    After initialization, call `Backtest.run` to run a backtest instance,
    `Backtest.optimize` to optimize parameters, or `Backtest.plot` to visualize results.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: type[Strategy],
        *,
        cash: float = 10_000,
        spread: float = 0.0,
        commission: float | tuple[float, float] | Callable[[float, float], float] = 0.0,
        margin: float = 1.0,
        trade_on_close: bool = False,
        hedging: bool = False,
        exclusive_orders: bool = False,
        finalize_trades: bool = False,
    ):
        if not (isinstance(strategy, type) and issubclass(strategy, Strategy)):
            raise TypeError('`strategy` must be a Strategy sub-type')
        if not isinstance(data, pd.DataFrame):
            raise TypeError('`data` must be a pandas.DataFrame with columns')
        if not isinstance(spread, Number):
            raise TypeError('`spread` must be a float value, percent of entry order price')
        if not isinstance(commission, (Number, tuple)) and not callable(commission):
            raise TypeError(
                '`commission` must be a float percent of order value, '
                'a tuple of `(fixed, relative)` commission, '
                'or a function that takes `(order_size, price)` '
                'and returns commission dollar value'
            )

        data = data.copy(deep=False)

        if (
            not isinstance(data.index, pd.DatetimeIndex)
            and not isinstance(data.index, pd.RangeIndex)
            and (
                pd.api.types.is_numeric_dtype(data.index)
                and (data.index > pd.Timestamp('1975').timestamp()).mean() > 0.8
            )
        ):
            try:
                data.index = pd.to_datetime(data.index)
            except ValueError:
                pass

        if 'Volume' not in data:
            data['Volume'] = np.nan

        if len(data) == 0:
            raise ValueError('OHLC `data` is empty')
        if len(data.columns.intersection({'Open', 'High', 'Low', 'Close', 'Volume'})) != 5:
            raise ValueError(
                '`data` must be a pandas.DataFrame with columns '
                "'Open', 'High', 'Low', 'Close', and (optionally) 'Volume'"
            )
        if data[['Open', 'High', 'Low', 'Close']].isnull().values.any():
            raise ValueError(
                'Some OHLC values are missing (NaN). '
                'Please strip those lines with `df.dropna()` or '
                'fill them in with `df.interpolate()` or whatever.'
            )
        if np.any(data['Close'] > cash):
            warnings.warn(
                'Some prices are larger than initial cash value. Note that fractional '
                'trading is not supported by this class. If you want to trade Bitcoin, '
                'increase initial cash, or trade μBTC or satoshis instead (see e.g. class '
                '`tempusonebt.lib.FractionalBacktest`.',
                stacklevel=2,
            )
        if not data.index.is_monotonic_increasing:
            warnings.warn('Data index is not sorted in ascending order. Sorting.', stacklevel=2)
            data = data.sort_index()
        if not isinstance(data.index, pd.DatetimeIndex):
            warnings.warn(
                'Data index is not datetime. Assuming simple periods, '
                'but `pd.DateTimeIndex` is advised.',
                stacklevel=2,
            )

        self._data: pd.DataFrame = data
        self._broker = partial(
            _Broker,
            cash=cash,
            spread=spread,
            commission=commission,
            margin=margin,
            trade_on_close=trade_on_close,
            hedging=hedging,
            exclusive_orders=exclusive_orders,
            index=data.index,
        )
        self._strategy = strategy
        self._results: pd.Series | None = None
        self._finalize_trades = bool(finalize_trades)

    def run(self, **kwargs) -> pd.Series:
        """
        Run the backtest. Returns `pd.Series` with results and statistics.

        Keyword arguments are interpreted as strategy parameters.

            >>> Backtest(GOOG, SmaCross).run()
            Start                     2004-08-19 00:00:00
            End                       2013-03-01 00:00:00
            Duration                   3116 days 00:00:00
            Exposure Time [%]                    96.74115
            Equity Final [$]                     51422.99
            Equity Peak [$]                      75787.44
            Return [%]                           414.2299
            Buy & Hold Return [%]               703.45824
            Return (Ann.) [%]                    21.18026
            Volatility (Ann.) [%]                36.49391
            CAGR [%]                             21.16025
            Sharpe Ratio                          0.58038
            Sortino Ratio                         1.08479
            Calmar Ratio                          0.44144
            Alpha [%]                           393.83510
            Beta                                  0.03907
            Max. Drawdown [%]                   -47.98013
            Avg. Drawdown [%]                    -5.92585
            Max. Drawdown Duration      584 days 00:00:00
            Avg. Drawdown Duration       41 days 00:00:00
            # Trades                                   66
            Win Rate [%]                          46.9697
            Best Trade [%]                       53.59595
            Worst Trade [%]                     -18.39887
            Avg. Trade [%]                        2.53172
            Max. Trade Duration         183 days 00:00:00
            Avg. Trade Duration          46 days 00:00:00
            Profit Factor                         1.47784
            Expectancy [%]                        3.27481
            SQN                                   1.07662
            Kelly Criterion                       0.15187
            _strategy                            SmaCross
            _equity_curve                           Eq...
            _trades                       Size  EntryB...
            dtype: object
        """
        data = _Data(self._data.copy(deep=False))
        broker: _Broker = self._broker(data=data)
        strategy: Strategy = self._strategy(broker, data, kwargs)

        strategy.init()
        data._update()

        indicator_attrs = _strategy_indicators(strategy)
        start = 1 + _indicator_warmup_nbars(strategy)

        with np.errstate(invalid='ignore'):
            for i in _tqdm(
                range(start, len(self._data)),
                desc=self.run.__qualname__,
                unit='bar',
                mininterval=2,
                miniters=100,
            ):
                data._set_length(i + 1)
                for attr, indicator in indicator_attrs:
                    setattr(strategy, attr, indicator[..., : i + 1])

                try:
                    broker.next()
                except _OutOfMoneyError:
                    break

                strategy.next()
            else:
                if self._finalize_trades is True:
                    for trade in reversed(broker.trades):
                        trade.close()

                    if start < len(self._data):
                        try_(broker.next, exception=_OutOfMoneyError)
                elif len(broker.trades):
                    warnings.warn(
                        'Some trades remain open at the end of backtest. Use '
                        '`Backtest(..., finalize_trades=True)` to close them and '
                        'include them in stats.',
                        stacklevel=2,
                    )

            data._set_length(len(self._data))

            equity = pd.Series(broker._equity).bfill().fillna(broker._cash).values
            self._results = compute_stats(
                trades=broker.closed_trades,
                equity=equity,
                ohlc_data=self._data,
                risk_free_rate=0.0,
                strategy_instance=strategy,
            )

        return self._results

    def optimize(
        self,
        *,
        maximize: str | Callable[[pd.Series], float] = 'SQN',
        method: str = 'grid',
        max_tries: int | float | None = None,
        constraint: Callable[[dict], bool] | None = None,
        return_heatmap: bool = False,
        return_optimization: bool = False,
        random_state: int | None = None,
        **kwargs,
    ) -> pd.Series | tuple[pd.Series, pd.Series] | tuple[pd.Series, pd.Series, dict]:
        """Optimize strategy parameters to an optimal combination."""
        if not kwargs:
            raise ValueError('Need some strategy parameters to optimize')

        maximize_key = None
        if isinstance(maximize, str):
            maximize_key = str(maximize)
            if maximize not in dummy_stats().index:
                raise ValueError(
                    '`maximize`, if str, must match a key in pd.Series result of backtest.run()'
                )

            def _maximize_by_key(stats: pd.Series, _key: str = maximize) -> float:
                return stats[_key]

            maximize_fn: Callable[[pd.Series], float] = _maximize_by_key
        elif callable(maximize):
            maximize_fn = maximize
        else:
            raise TypeError(
                '`maximize` must be str (a field of backtest.run() result '
                'Series) or a function that accepts result Series '
                'and returns a number; the higher the better'
            )

        have_constraint = bool(constraint)
        if constraint is None:
            constraint_fn: Callable[[dict], bool] = lambda _: True  # noqa: E731
        elif callable(constraint):
            constraint_fn = constraint
        else:
            raise TypeError(
                '`constraint` must be a function that accepts a dict '
                'of strategy parameters and returns a bool whether '
                'the combination of parameters is admissible or not'
            )

        if method == 'skopt':
            method = 'sambo'
            warnings.warn(
                '`Backtest.optimize(method="skopt")` is deprecated. Use `method="sambo"`.',
                DeprecationWarning,
                stacklevel=2,
            )
        if return_optimization and method != 'sambo':
            raise ValueError("return_optimization=True only valid if method='sambo'")

        def _tuple(x):
            return x if isinstance(x, Sequence) and not isinstance(x, str) else (x,)

        for k, v in kwargs.items():
            if len(_tuple(v)) == 0:
                raise ValueError(
                    f"Optimization variable '{k}' is passed no optimization values: {k}={v}"
                )

        class AttrDict(dict):
            def __getattr__(self, item):
                return self[item]

        def _grid_size() -> int:
            size = int(np.prod([len(_tuple(v)) for v in kwargs.values()]))
            if size < 10_000 and have_constraint:
                size = sum(
                    1
                    for p in product(*(zip(repeat(k), _tuple(v)) for k, v in kwargs.items()))
                    if constraint_fn(AttrDict(p))
                )
            return size

        def _optimize_grid() -> pd.Series | tuple[pd.Series, pd.Series]:
            rand = default_rng(random_state).random
            grid_frac = (
                1
                if max_tries is None
                else max_tries
                if 0 < max_tries <= 1
                else max_tries / _grid_size()
            )
            param_combos = [
                dict(params)
                for params in (
                    AttrDict(params)
                    for params in product(*(zip(repeat(k), _tuple(v)) for k, v in kwargs.items()))
                )
                if constraint_fn(params) and rand() <= grid_frac
            ]
            if not param_combos:
                raise ValueError('No admissible parameter combinations to test')

            if len(param_combos) > 300:
                warnings.warn(
                    f'Searching for best of {len(param_combos)} configurations.',
                    stacklevel=2,
                )

            heatmap = pd.Series(
                np.nan,
                name=maximize_key,
                index=pd.MultiIndex.from_tuples(
                    [p.values() for p in param_combos],
                    names=next(iter(param_combos)).keys(),
                ),
            )

            from . import Pool

            with Pool() as pool, SharedMemoryManager() as smm:
                with patch(self, '_data', None):
                    bt = copy(self)
                results = _tqdm(
                    pool.imap(
                        Backtest._mp_task,
                        (
                            (bt, smm.df2shm(self._data), params_batch)
                            for params_batch in _batch(param_combos)
                        ),
                    ),
                    total=len(param_combos),
                    desc='Backtest.optimize',
                )
                for param_batch, result in zip(_batch(param_combos), results):
                    for params, stats in zip(param_batch, result):
                        if stats is not None:
                            heatmap[tuple(params.values())] = maximize_fn(stats)

            if pd.isnull(heatmap).all():
                stats = self.run(**param_combos[0])
            else:
                best_params = heatmap.idxmax(skipna=True)
                stats = self.run(**dict(zip(heatmap.index.names, best_params)))

            if return_heatmap:
                return stats, heatmap
            return stats

        def _optimize_sambo() -> (
            pd.Series | tuple[pd.Series, pd.Series] | tuple[pd.Series, pd.Series, dict]
        ):
            try:
                import sambo
            except ImportError:
                raise ImportError(
                    "Need package 'sambo' for method='sambo'. pip install sambo"
                ) from None

            nonlocal max_tries
            max_tries = (
                200
                if max_tries is None
                else max(1, int(max_tries * _grid_size()))
                if 0 < max_tries <= 1
                else max_tries
            )

            dimensions = []
            for values in kwargs.values():
                arr = np.asarray(values)
                if arr.dtype.kind in 'mM':
                    arr = arr.astype(np.int64)

                if arr.dtype.kind in 'iumM':
                    dimensions.append((arr.min(), arr.max() + 1))
                elif arr.dtype.kind == 'f':
                    dimensions.append((arr.min(), arr.max()))
                else:
                    dimensions.append(arr.tolist())

            @lru_cache()
            def memoized_run(tup):
                stats = self.run(**dict(tup))
                return -maximize_fn(stats)

            progress = iter(
                _tqdm(
                    repeat(None),
                    total=max_tries,
                    leave=False,
                    desc=self.optimize.__qualname__,
                    mininterval=2,
                )
            )
            _names = tuple(kwargs.keys())

            def objective_function(x):
                next(progress)
                value = memoized_run(tuple(zip(_names, x)))
                return 0 if np.isnan(value) else value

            def cons(x):
                return constraint_fn(AttrDict(zip(_names, x)))

            res = sambo.minimize(
                fun=objective_function,
                bounds=dimensions,
                constraints=cons,
                max_iter=max_tries,
                method='sceua',
                rng=random_state,
            )

            stats = self.run(**dict(zip(kwargs.keys(), res.x)))
            output: list = [stats]

            if return_heatmap:
                heatmap = pd.Series(
                    dict(zip(map(tuple, res.xv), -res.funv)),
                    name=maximize_key,
                )
                heatmap.index.names = kwargs.keys()
                heatmap.sort_index(inplace=True)
                output.append(heatmap)

            if return_optimization:
                output.append(res)

            return stats if len(output) == 1 else tuple(output)

        if method == 'grid':
            return _optimize_grid()
        if method in ('sambo', 'skopt'):
            return _optimize_sambo()
        raise ValueError(f"Method should be 'grid' or 'sambo', not {method!r}")

    @staticmethod
    def _mp_task(arg):
        bt, data_shm, params_batch = arg
        bt._data, shm = SharedMemoryManager.shm2df(data_shm)
        try:
            return [
                stats.filter(regex='^[^_]') if stats['# Trades'] else None
                for stats in (bt.run(**params) for params in params_batch)
            ]
        finally:
            for shmem in shm:
                shmem.close()

    def plot(
        self,
        *,
        results: pd.Series | None = None,
        filename=None,
        plot_width=None,
        plot_equity: bool = True,
        plot_return: bool = False,
        plot_pl: bool = True,
        plot_volume: bool = True,
        plot_drawdown: bool = False,
        plot_trades: bool = True,
        smooth_equity: bool = False,
        relative_equity: bool = True,
        superimpose: bool | str = True,
        resample: bool | str = True,
        reverse_indicators: bool = False,
        show_legend: bool = True,
        open_browser: bool = True,
    ):
        """Plot the progression of the last backtest run."""
        if results is None:
            if self._results is None:
                raise RuntimeError('First issue `backtest.run()` to obtain results.')
            results = self._results

        return plot(
            results=results,
            df=self._data,
            indicators=results._strategy._indicators,
            filename=filename,
            plot_width=plot_width,
            plot_equity=plot_equity,
            plot_return=plot_return,
            plot_pl=plot_pl,
            plot_volume=plot_volume,
            plot_drawdown=plot_drawdown,
            plot_trades=plot_trades,
            smooth_equity=smooth_equity,
            relative_equity=relative_equity,
            superimpose=superimpose,
            resample=resample,
            reverse_indicators=reverse_indicators,
            show_legend=show_legend,
            open_browser=open_browser,
        )
