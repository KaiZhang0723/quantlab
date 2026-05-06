Backtesting
===========

The :mod:`quantlab.compute.backtest` module runs a per-ticker strategy in
parallel via ``multiprocessing.Pool.map``. The default reference strategy
is 12-1 *time-series* momentum (per ticker, long when that ticker's prior
12 months excluding the most recent month delivered positive returns).

CLI
---

.. code-block:: bash

   quantlab backtest \
       --prices data/prices.csv \
       --workers 4 \
       --out reports/backtest_metrics.csv \
       --lookback 252 --skip 21

The output CSV contains per-ticker Sharpe, annualised return, annualised
volatility, and maximum drawdown.

Python API
----------

.. code-block:: python

   import pandas as pd
   from quantlab.compute.backtest import run_backtest, momentum_strategy

   prices = pd.read_csv("data/prices.csv", parse_dates=["date"])
   result = run_backtest(prices, strategy=momentum_strategy, n_workers=4)

   metrics = result.metrics_table()
   equity = result.portfolio_equity

Theoretical upper bound
-----------------------

The :mod:`quantlab.compute.optimal_execution` module computes the
perfect-foresight maximum profit a trader could achieve on a given price
path with a per-trade fee. It is a useful upper bound for realistic
strategies::

   from quantlab.compute.optimal_execution import max_profit_with_fee
   upper_bound = max_profit_with_fee(prices.tolist(), fee=0.001)
