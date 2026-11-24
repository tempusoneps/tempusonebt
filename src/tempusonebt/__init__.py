"""tempusonebt — Backtest trading strategies in Python."""

from __future__ import annotations

try:
    from ._version import version as __version__
except ImportError:
    __version__ = '?.?.?'

from . import lib
from ._plotting import set_bokeh_output
from ._util import try_
from .engine import Backtest
from .models import Order, Position, Trade
from .strategy import Strategy

__all__ = [
    'Backtest',
    'Order',
    'Pool',
    'Position',
    'Strategy',
    'Trade',
    '__version__',
    'lib',
    'set_bokeh_output',
]


def Pool(processes=None, initializer=None, initargs=()):
    """Overridable process/thread pool used for parallel optimization."""
    import multiprocessing as mp
    import sys

    if sys.platform.startswith('linux') and mp.get_start_method(allow_none=True) != 'fork':
        try_(lambda: mp.set_start_method('fork'))
    if mp.get_start_method() == 'spawn':
        import warnings

        warnings.warn(
            "If you want to use multi-process optimization with "
            "`multiprocessing.get_start_method() == 'spawn'` (e.g. on Windows), "
            "set `tempusonebt.Pool = multiprocessing.Pool` (or of the desired context) "
            "and hide `bt.optimize()` call behind a `if __name__ == '__main__'` guard. "
            'Currently using thread-based parallelism, '
            'which might be slightly slower for non-numpy / non-GIL-releasing code.',
            category=RuntimeWarning,
            stacklevel=3,
        )
        from multiprocessing.dummy import Pool as ThreadPool

        return ThreadPool(processes, initializer, initargs)
    return mp.Pool(processes, initializer, initargs)
