"""Core trading domain models: Position, Order, and Trade."""

from __future__ import annotations

from copy import copy
from math import copysign
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._util import try_

if TYPE_CHECKING:
    from .broker import _Broker


class _OutOfMoneyError(Exception):
    pass


class Position:
    """
    Currently held asset position, available as `Strategy.position` within `Strategy.next`.
    Can be used in boolean contexts, e.g.:

        if self.position:
            ...  # we have a position, either long or short
    """

    def __init__(self, broker: _Broker):
        self._broker = broker

    def __bool__(self) -> bool:
        return self.size != 0

    @property
    def size(self) -> float:
        """Position size in units of asset. Negative if position is short."""
        return sum(trade.size for trade in self._broker.trades)

    @property
    def pl(self) -> float:
        """Profit (positive) or loss (negative) of the current position in cash units."""
        return sum(trade.pl for trade in self._broker.trades)

    @property
    def pl_pct(self) -> float:
        """Profit (positive) or loss (negative) of the current position as a fraction."""
        total_invested = sum(trade.entry_price * abs(trade.size) for trade in self._broker.trades)
        return self.pl / total_invested if total_invested else 0

    @property
    def is_long(self) -> bool:
        """True if the position is long (position size is positive)."""
        return self.size > 0

    @property
    def is_short(self) -> bool:
        """True if the position is short (position size is negative)."""
        return self.size < 0

    def close(self, portion: float = 1.0) -> None:
        """Close portion of position by closing `portion` of each active trade."""
        for trade in self._broker.trades:
            trade.close(portion)

    def __repr__(self) -> str:
        return f'<Position: {self.size} ({len(self._broker.trades)} trades)>'


class Order:
    """
    Place new orders through `Strategy.buy()` and `Strategy.sell()`.
    Query existing orders through `Strategy.orders`.

    When an order is executed (filled), it results in a `Trade`.
    All placed orders are Good 'Til Canceled (GTC).
    """

    def __init__(
        self,
        broker: _Broker,
        size: float,
        limit_price: float | None = None,
        stop_price: float | None = None,
        sl_price: float | None = None,
        tp_price: float | None = None,
        parent_trade: Trade | None = None,
        tag: object = None,
    ):
        if size == 0:
            raise ValueError('Order size must be non-zero')
        self._broker = broker
        self._size = size
        self._limit_price = limit_price
        self._stop_price = stop_price
        self._sl_price = sl_price
        self._tp_price = tp_price
        self._parent_trade = parent_trade
        self._tag = tag

    def _replace(self, **kwargs) -> Order:
        for k, v in kwargs.items():
            setattr(self, f'_{k}', v)
        return self

    def __repr__(self) -> str:
        fields = (
            ('size', self._size),
            ('limit', self._limit_price),
            ('stop', self._stop_price),
            ('sl', self._sl_price),
            ('tp', self._tp_price),
            ('contingent', self.is_contingent),
            ('tag', self._tag),
        )
        formatted = ', '.join(
            f'{param}={try_(lambda: round(value, 5), value)!r}'
            for param, value in fields
            if value is not None
        )
        return f'<Order {formatted}>'

    def cancel(self) -> None:
        """Cancel the order."""
        self._broker.orders.remove(self)
        trade = self._parent_trade
        if trade:
            if self is trade._sl_order:
                trade._replace(sl_order=None)
            elif self is trade._tp_order:
                trade._replace(tp_order=None)

    @property
    def size(self) -> float:
        """Order size (negative for short orders)."""
        return self._size

    @property
    def limit(self) -> float | None:
        """Order limit price for limit orders, or None for market orders."""
        return self._limit_price

    @property
    def stop(self) -> float | None:
        """Order stop price for stop-limit/stop-market orders, or None."""
        return self._stop_price

    @property
    def sl(self) -> float | None:
        """Stop-loss price for the contingent bracket order."""
        return self._sl_price

    @property
    def tp(self) -> float | None:
        """Take-profit price for the contingent bracket order."""
        return self._tp_price

    @property
    def parent_trade(self) -> Trade | None:
        return self._parent_trade

    @property
    def tag(self) -> object:
        """Arbitrary value inherited by the resulting `Trade`."""
        return self._tag

    @property
    def is_long(self) -> bool:
        """True if the order is long (order size is positive)."""
        return self._size > 0

    @property
    def is_short(self) -> bool:
        """True if the order is short (order size is negative)."""
        return self._size < 0

    @property
    def is_contingent(self) -> bool:
        """True for contingent OCO stop-loss and take-profit bracket orders."""
        return bool(
            (parent := self._parent_trade)
            and (self is parent._sl_order or self is parent._tp_order)
        )


class Trade:
    """
    When an `Order` is filled, it results in an active `Trade`.
    Find active trades in `Strategy.trades` and settled trades in `Strategy.closed_trades`.
    """

    def __init__(
        self,
        broker: _Broker,
        size: int,
        entry_price: float,
        entry_bar: int,
        tag: object,
    ):
        self._broker = broker
        self._size = size
        self._entry_price = entry_price
        self._exit_price: float | None = None
        self._entry_bar: int = entry_bar
        self._exit_bar: int | None = None
        self._sl_order: Order | None = None
        self._tp_order: Order | None = None
        self._tag = tag
        self._commissions: float = 0
        self._open_commission: float = 0

    def __repr__(self) -> str:
        tag_str = f' tag={self._tag}' if self._tag is not None else ''
        return (
            f'<Trade size={self._size} time={self._entry_bar}-{self._exit_bar or ""} '
            f'price={self._entry_price}-{self._exit_price or ""} pl={self.pl:.0f}{tag_str}>'
        )

    def _replace(self, **kwargs) -> Trade:
        for k, v in kwargs.items():
            setattr(self, f'_{k}', v)
        return self

    def _copy(self, **kwargs) -> Trade:
        return copy(self)._replace(**kwargs)

    def close(self, portion: float = 1.0) -> None:
        """Place new `Order` to close `portion` of the trade at next market price."""
        if not 0 < portion <= 1:
            raise ValueError('portion must be a fraction between 0 and 1')
        size = copysign(max(1, int(round(abs(self._size) * portion))), -self._size)
        order = Order(self._broker, size, parent_trade=self, tag=self._tag)
        self._broker.orders.insert(0, order)

    @property
    def size(self) -> int:
        """Trade size (volume; negative for short trades)."""
        return self._size

    @property
    def entry_price(self) -> float:
        """Trade entry price."""
        return self._entry_price

    @property
    def exit_price(self) -> float | None:
        """Trade exit price (or None if the trade is still active)."""
        return self._exit_price

    @property
    def entry_bar(self) -> int:
        """Candlestick bar index of when the trade was entered."""
        return self._entry_bar

    @property
    def exit_bar(self) -> int | None:
        """Candlestick bar index of when the trade was exited (or None if still active)."""
        return self._exit_bar

    @property
    def tag(self) -> object:
        """Tag value inherited from the `Order` that opened this trade."""
        return self._tag

    @property
    def entry_time(self) -> pd.Timestamp | int:
        """Datetime of when the trade was entered."""
        return self._broker._data.index[self._entry_bar]

    @property
    def exit_time(self) -> pd.Timestamp | int | None:
        """Datetime of when the trade was exited."""
        if self._exit_bar is None:
            return None
        return self._broker._data.index[self._exit_bar]

    @property
    def is_long(self) -> bool:
        """True if the trade is long (trade size is positive)."""
        return self._size > 0

    @property
    def is_short(self) -> bool:
        """True if the trade is short (trade size is negative)."""
        return not self.is_long

    @property
    def pl(self) -> float:
        """
        Trade profit (positive) or loss (negative) in cash units.
        Commissions are reflected only after the Trade is closed.
        """
        price = self._exit_price or self._broker.last_price
        return (self._size * (price - self._entry_price)) - self._commissions

    @property
    def pl_pct(self) -> float:
        """Trade profit (positive) or loss (negative) as a fraction relative to entry price."""
        price = self._exit_price or self._broker.last_price
        gross_pl_pct = copysign(1, self._size) * (price / self._entry_price - 1)
        commission_pct = self._commissions / (abs(self._size) * self._entry_price)
        return gross_pl_pct - commission_pct

    @property
    def value(self) -> float:
        """Trade total value in cash (volume * price)."""
        price = self._exit_price or self._broker.last_price
        return abs(self._size) * price

    @property
    def sl(self) -> float | None:
        """Stop-loss price at which to close the trade."""
        return self._sl_order.stop if self._sl_order else None

    @sl.setter
    def sl(self, price: float | None) -> None:
        self._set_contingent('sl', price)

    @property
    def tp(self) -> float | None:
        """Take-profit price at which to close the trade."""
        return self._tp_order.limit if self._tp_order else None

    @tp.setter
    def tp(self, price: float | None) -> None:
        self._set_contingent('tp', price)

    def _set_contingent(self, kind: str, price: float | None) -> None:
        if kind not in ('sl', 'tp'):
            raise ValueError(f'Unexpected contingent order type: {kind!r}')
        if price is not None and not (0 < price < np.inf):
            raise ValueError(f'Make sure 0 < price < inf! price: {price}')
        attr = f'_{kind}_order'
        order: Order | None = getattr(self, attr)
        if order:
            order.cancel()
        if price:
            kwargs = {'stop': price} if kind == 'sl' else {'limit': price}
            order = self._broker.new_order(-self.size, trade=self, tag=self.tag, **kwargs)
            setattr(self, attr, order)
