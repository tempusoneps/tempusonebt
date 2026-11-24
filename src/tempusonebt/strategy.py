"""Abstract base class for trading strategies."""

from __future__ import annotations

import sys
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from difflib import get_close_matches
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._util import _as_str, _Data, _Indicator, try_

if TYPE_CHECKING:
    from .broker import _Broker
    from .models import Order, Position, Trade


class Strategy(metaclass=ABCMeta):
    """
    A trading strategy base class. Extend this class and override methods
    `Strategy.init` and `Strategy.next` to define your own strategy.
    """

    def __init__(self, broker: _Broker, data: _Data, params: dict):
        self._indicators: list[_Indicator] = []
        self._broker: _Broker = broker
        self._data: _Data = data
        self._params = self._check_params(params)

    def __repr__(self) -> str:
        return '<Strategy ' + str(self) + '>'

    def __str__(self) -> str:
        params = ','.join(
            f'{k}={v}' for k, v in zip(self._params.keys(), map(_as_str, self._params.values()))
        )
        if params:
            params = '(' + params + ')'
        return f'{self.__class__.__name__}{params}'

    def _check_params(self, params: dict) -> dict:
        for k, v in params.items():
            if not hasattr(self, k):
                suggestions = get_close_matches(
                    k, (attr for attr in dir(self) if not attr.startswith('_'))
                )
                hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ''
                raise AttributeError(
                    f"Strategy '{self.__class__.__name__}' is missing parameter '{k}'. "
                    'Strategy class should define parameters as class variables before they '
                    'can be optimized or run with.' + hint
                )
            setattr(self, k, v)
        return params

    def I(  # noqa: E743
        self,
        func: Callable,
        *args,
        name=None,
        plot: bool = True,
        overlay: bool | None = None,
        color=None,
        scatter: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Declare an indicator array (or tuple of arrays) revealed gradually in `Strategy.next`.
        """
        def _format_name(template: str) -> str:
            return template.format(
                *map(_as_str, args),
                **dict(zip(kwargs.keys(), map(_as_str, kwargs.values()))),
            )

        if name is None:
            params = ','.join(filter(None, map(_as_str, chain(args, kwargs.values()))))
            func_name = _as_str(func)
            name = f'{func_name}({params})' if params else f'{func_name}'
        elif isinstance(name, str):
            name = _format_name(name)
        elif try_(lambda: all(isinstance(item, str) for item in name), False):
            name = [_format_name(item) for item in name]
        else:
            raise TypeError(
                f'Unexpected `name=` type {type(name)}; expected `str` or `Sequence[str]`'
            )

        try:
            value = func(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f'Indicator "{name}" error. See traceback above.') from e

        if isinstance(value, pd.DataFrame):
            value = value.values.T

        if value is not None:
            value = try_(lambda: np.asarray(value, order='C'), None)
        is_arraylike = bool(value is not None and value.shape)

        # Optionally flip the array if the user returned e.g. `df.values`
        if is_arraylike and np.argmax(value.shape) == 0:
            value = value.T

        if isinstance(name, list) and (np.atleast_2d(value).shape[0] != len(name)):
            raise ValueError(
                f'Length of `name=` ({len(name)}) must agree with the number '
                f'of arrays the indicator returns ({value.shape[0]}).'
            )

        if not is_arraylike or not 1 <= value.ndim <= 2 or value.shape[-1] != len(self._data.Close):
            raise ValueError(
                'Indicators must return (optionally a tuple of) numpy.arrays of same '
                f'length as `data` (data shape: {self._data.Close.shape}; indicator "{name}" '
                f'shape: {getattr(value, "shape", "")}, returned value: {value})'
            )

        if overlay is None and np.issubdtype(value.dtype, np.number):
            x = value / self._data.Close
            with np.errstate(invalid='ignore'):
                overlay = bool(((x < 1.4) & (x > 0.6)).mean() > 0.6)

        indicator = _Indicator(
            value,
            name=name,
            plot=plot,
            overlay=overlay,
            color=color,
            scatter=scatter,
            index=self.data.index,
        )
        self._indicators.append(indicator)
        return indicator

    @abstractmethod
    def init(self) -> None:
        """Initialize the strategy and declare indicators with `Strategy.I`."""

    @abstractmethod
    def next(self) -> None:
        """Main strategy runtime method, called on each new candlestick bar."""

    class __FULL_EQUITY(float):  # noqa: N801
        def __repr__(self) -> str:
            return '.9999'

    _FULL_EQUITY = __FULL_EQUITY(1 - sys.float_info.epsilon)

    def buy(
        self,
        *,
        size: float = _FULL_EQUITY,
        limit: float | None = None,
        stop: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
        tag: object = None,
    ) -> Order:
        """Place a new long order and return it."""
        if not (0 < size < 1 or round(size) == size >= 1):
            raise ValueError(
                'size must be a positive fraction of equity, or a positive whole number of units'
            )
        return self._broker.new_order(size, limit, stop, sl, tp, tag)

    def sell(
        self,
        *,
        size: float = _FULL_EQUITY,
        limit: float | None = None,
        stop: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
        tag: object = None,
    ) -> Order:
        """Place a new short order and return it."""
        if not (0 < size < 1 or round(size) == size >= 1):
            raise ValueError(
                'size must be a positive fraction of equity, or a positive whole number of units'
            )
        return self._broker.new_order(-size, limit, stop, sl, tp, tag)

    @property
    def equity(self) -> float:
        """Current account equity (cash plus assets)."""
        return self._broker.equity

    @property
    def data(self) -> _Data:
        """Price data accessor."""
        return self._data

    @property
    def position(self) -> Position:
        """Current open `Position`."""
        return self._broker.position

    @property
    def orders(self) -> tuple[Order, ...]:
        """Orders waiting for execution."""
        return tuple(self._broker.orders)

    @property
    def trades(self) -> tuple[Trade, ...]:
        """Currently active trades."""
        return tuple(self._broker.trades)

    @property
    def closed_trades(self) -> tuple[Trade, ...]:
        """Settled trades."""
        return tuple(self._broker.closed_trades)
