"""Simulated broker handling order execution, margin, spread, and commissions."""

from __future__ import annotations

import warnings
from math import copysign

import numpy as np

from ._util import _Data
from .models import Order, Position, Trade, _OutOfMoneyError


class _Broker:
    def __init__(
        self,
        *,
        data: _Data,
        cash: float,
        spread: float,
        commission,
        margin: float,
        trade_on_close: bool,
        hedging: bool,
        exclusive_orders: bool,
        index,
    ):
        if cash <= 0:
            raise ValueError(f'cash should be > 0, is {cash}')
        if not (0 < margin <= 1):
            raise ValueError(f'margin should be between 0 and 1, is {margin}')
        self._data: _Data = data
        self._cash = cash

        if callable(commission):
            self._commission = commission
        else:
            try:
                self._commission_fixed, self._commission_relative = commission
            except TypeError:
                self._commission_fixed, self._commission_relative = 0, commission
            if self._commission_fixed < 0:
                raise ValueError('Need fixed cash commission in $ >= 0')
            if not (-0.1 <= self._commission_relative < 0.1):
                raise ValueError(
                    'commission should be between -10% '
                    f"(e.g. market-maker's rebates) and 10% (fees), is {self._commission_relative}"
                )
            self._commission = self._commission_func

        self._spread = spread
        self._leverage = 1 / margin
        self._trade_on_close = trade_on_close
        self._hedging = hedging
        self._exclusive_orders = exclusive_orders

        self._equity = np.tile(np.nan, len(index))
        self.orders: list[Order] = []
        self.trades: list[Trade] = []
        self.position = Position(self)
        self.closed_trades: list[Trade] = []
        self._i: int = 0

    def _commission_func(self, order_size: float, price: float) -> float:
        return self._commission_fixed + abs(order_size) * price * self._commission_relative

    def __repr__(self) -> str:
        return f'<Broker: {self._cash:.0f}{self.position.pl:+.1f} ({len(self.trades)} trades)>'

    def new_order(
        self,
        size: float,
        limit: float | None = None,
        stop: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
        tag: object = None,
        *,
        trade: Trade | None = None,
    ) -> Order:
        """Argument size indicates whether the order is long or short."""
        size = float(size)
        stop = stop and float(stop)
        limit = limit and float(limit)
        sl = sl and float(sl)
        tp = tp and float(tp)

        is_long = size > 0
        if size == 0:
            raise ValueError('Order size must be non-zero')
        adjusted_price = self._adjusted_price(size)

        if is_long:
            if not (sl or -np.inf) < (limit or stop or adjusted_price) < (tp or np.inf):
                raise ValueError(
                    'Long orders require: '
                    f'SL ({sl}) < LIMIT ({limit or stop or adjusted_price}) < TP ({tp})'
                )
        else:
            if not (tp or -np.inf) < (limit or stop or adjusted_price) < (sl or np.inf):
                raise ValueError(
                    'Short orders require: '
                    f'TP ({tp}) < LIMIT ({limit or stop or adjusted_price}) < SL ({sl})'
                )

        order = Order(self, size, limit, stop, sl, tp, trade, tag)

        if not trade:
            if self._exclusive_orders:
                for o in list(self.orders):
                    if not o.is_contingent:
                        o.cancel()
                for t in self.trades:
                    t.close()

        # Put the new order in the order queue, ensuring SL orders are processed first
        self.orders.insert(0 if trade and stop else len(self.orders), order)

        return order

    @property
    def last_price(self) -> float:
        """Price at the last (current) close."""
        return self._data.Close[-1]

    def _adjusted_price(self, size: float = 1, price: float | None = None) -> float:
        """
        Long/short `price`, adjusted for spread.
        In long positions, the adjusted price is a fraction higher, and vice versa.
        """
        return (price or self.last_price) * (1 + copysign(self._spread, size))

    @property
    def equity(self) -> float:
        return self._cash + sum(trade.pl for trade in self.trades)

    @property
    def margin_available(self) -> float:
        margin_used = sum(trade.value / self._leverage for trade in self.trades)
        return max(0, self.equity - margin_used)

    def next(self) -> None:
        i = self._i = len(self._data) - 1
        self._process_orders()

        equity = self.equity
        self._equity[i] = equity

        if equity <= 0:
            assert self.margin_available <= 0
            for trade in list(self.trades):
                self._close_trade(trade, self._data.Close[-1], i)
            self._cash = 0
            self._equity[i:] = 0
            raise _OutOfMoneyError

    def _process_orders(self) -> None:
        data = self._data
        open_, high, low = data.Open[-1], data.High[-1], data.Low[-1]
        reprocess_orders = False

        for order in list(self.orders):
            if order not in self.orders:
                continue

            stop_price = order.stop
            if stop_price:
                is_stop_hit = (high >= stop_price) if order.is_long else (low <= stop_price)
                if not is_stop_hit:
                    continue
                order._replace(stop_price=None)

            if order.limit:
                is_limit_hit = low <= order.limit if order.is_long else high >= order.limit
                is_limit_hit_before_stop = is_limit_hit and (
                    order.limit <= (stop_price or -np.inf)
                    if order.is_long
                    else order.limit >= (stop_price or np.inf)
                )
                if not is_limit_hit or is_limit_hit_before_stop:
                    continue

                price = (
                    min(stop_price or open_, order.limit)
                    if order.is_long
                    else max(stop_price or open_, order.limit)
                )
            else:
                prev_close = data.Close[-2]
                price = prev_close if self._trade_on_close and not order.is_contingent else open_
                if stop_price:
                    price = max(price, stop_price) if order.is_long else min(price, stop_price)

            is_market_order = not order.limit and not stop_price
            time_index = (
                (self._i - 1)
                if is_market_order and self._trade_on_close and not order.is_contingent
                else self._i
            )

            if order.parent_trade:
                trade = order.parent_trade
                _prev_size = trade.size
                size = copysign(min(abs(_prev_size), abs(order.size)), order.size)
                if trade in self.trades:
                    self._reduce_trade(trade, price, size, time_index)
                    assert order.size != -_prev_size or trade not in self.trades
                    if order is trade._sl_order:
                        trade._sl_order._replace(stop_price=stop_price)
                if order in (trade._sl_order, trade._tp_order):
                    assert order.size == -trade.size
                    assert order not in self.orders
                else:
                    assert abs(_prev_size) >= abs(size) >= 1
                    self.orders.remove(order)
                continue

            adjusted_price = self._adjusted_price(order.size, price)

            size = order.size
            if -1 < size < 1:
                available_margin = self.margin_available * abs(size)
                if self._commission == self._commission_func:
                    margin_per_unit = (
                        adjusted_price / self._leverage + price * self._commission_relative
                    )
                    units = (
                        int(max(0, (available_margin - self._commission_fixed) // margin_per_unit))
                        if margin_per_unit > 0
                        else 0
                    )
                else:
                    lo, hi = 0, int((available_margin * self._leverage) // adjusted_price)
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        req = mid * adjusted_price / self._leverage + self._commission(
                            copysign(mid, size), price
                        )
                        if req <= available_margin:
                            lo = mid
                        else:
                            hi = mid - 1
                    units = lo
                size = copysign(units, size)
                if not size:
                    warnings.warn(
                        f'time={self._i}: Broker canceled the relative-sized order due to insufficient margin '
                        f'(equity={self.equity:.2f}, margin_available={self.margin_available:.2f}).',
                        category=UserWarning,
                    )
                    self.orders.remove(order)
                    continue
            assert size == round(size)
            need_size = int(size)

            if not self._hedging:
                for trade in list(self.trades):
                    if trade.is_long == order.is_long:
                        continue
                    assert trade.size * order.size < 0

                    if abs(need_size) >= abs(trade.size):
                        self._close_trade(trade, price, time_index)
                        need_size += trade.size
                    else:
                        self._reduce_trade(trade, price, need_size, time_index)
                        need_size = 0

                    if not need_size:
                        break

            if need_size and (
                abs(need_size) * adjusted_price / self._leverage
                + self._commission(need_size, price)
                > self.margin_available
            ):
                warnings.warn(
                    f'time={self._i}: Broker canceled the order due to insufficient margin '
                    f'(equity={self.equity:.2f}, margin_available={self.margin_available:.2f}).',
                    category=UserWarning,
                )
                self.orders.remove(order)
                continue

            if need_size:
                self._open_trade(
                    adjusted_price,
                    need_size,
                    order.sl,
                    order.tp,
                    time_index,
                    order.tag,
                )

                if order.sl or order.tp:
                    if is_market_order:
                        reprocess_orders = True
                    elif (
                        stop_price
                        and not order.limit
                        and order.tp
                        and (
                            (order.is_long and order.tp <= high and (order.sl or -np.inf) < low)
                            or (order.is_short and order.tp >= low and (order.sl or np.inf) > high)
                        )
                    ):
                        reprocess_orders = True
                    elif (
                        low <= (order.sl or -np.inf) <= high
                        or low <= (order.tp or -np.inf) <= high
                    ):
                        warnings.warn(
                            f'({data.index[-1]}) A contingent SL/TP order would execute in the '
                            'same bar its parent stop/limit order was turned into a trade. '
                            "Since we can't assert the precise intra-candle "
                            'price movement, the affected SL/TP order will instead be executed on '
                            'the next (matching) price/bar, making the result (of this trade) '
                            'somewhat dubious.',
                            UserWarning,
                        )

            self.orders.remove(order)

        if reprocess_orders:
            self._process_orders()

    def _reduce_trade(self, trade: Trade, price: float, size: float, time_index: int) -> None:
        assert trade.size * size < 0
        assert abs(trade.size) >= abs(size)

        size_left = trade.size + size
        assert size_left * trade.size >= 0
        if not size_left:
            close_trade = trade
        else:
            closed_open_commission = trade._open_commission * (abs(size) / abs(trade.size))
            trade._open_commission -= closed_open_commission

            trade._replace(size=size_left)
            if trade._sl_order:
                trade._sl_order._replace(size=-trade.size)
            if trade._tp_order:
                trade._tp_order._replace(size=-trade.size)

            close_trade = trade._copy(size=-size, sl_order=None, tp_order=None)
            close_trade._open_commission = closed_open_commission
            self.trades.append(close_trade)

        self._close_trade(close_trade, price, time_index)

    def _close_trade(self, trade: Trade, price: float, time_index: int) -> None:
        self.trades.remove(trade)
        if trade._sl_order:
            self.orders.remove(trade._sl_order)
        if trade._tp_order:
            self.orders.remove(trade._tp_order)

        closed_trade = trade._replace(exit_price=price, exit_bar=time_index)
        self.closed_trades.append(closed_trade)
        commission = self._commission(trade.size, price)
        self._cash += trade.pl - commission
        closed_trade._commissions = commission + closed_trade._open_commission

    def _open_trade(
        self,
        price: float,
        size: int,
        sl: float | None,
        tp: float | None,
        time_index: int,
        tag: object,
    ) -> None:
        trade = Trade(self, size, price, time_index, tag)
        self.trades.append(trade)
        entry_commission = self._commission(size, price)
        trade._open_commission = entry_commission
        self._cash -= entry_commission
        if tp:
            trade.tp = tp
        if sl:
            trade.sl = sl
