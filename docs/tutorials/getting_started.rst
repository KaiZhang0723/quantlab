Getting started
===============

Install
-------

.. code-block:: bash

   git clone https://github.com/kzhang-cornell/quantlab.git
   cd quantlab
   pip install -e ".[dev,docs]"

Run the test suite to confirm a clean install::

   pytest --cov=quantlab

Fetch some data
---------------

The CLI can download a small price panel and cache it locally:

.. code-block:: bash

   quantlab fetch \
       --tickers AAPL MSFT GOOGL \
       --start 2020-01-01 --end 2024-12-31 \
       --out data/prices.csv \
       --cache-dir .cache

The same is available from Python:

.. code-block:: python

   from datetime import date
   from quantlab.data.cache import CSVCache
   from quantlab.data.yfinance_source import YFinanceSource

   src = CSVCache(YFinanceSource(), root=".cache")
   prices = src.fetch(["AAPL", "MSFT"], date(2020, 1, 1), date(2024, 12, 31))

Streaming statistics
--------------------

Per-ticker online vol and median estimators avoid look-ahead bias::

   from quantlab.streaming.welford import OnlineMoments
   from quantlab.streaming.median import RunningMedian

   m = OnlineMoments()
   m.update_many([0.001, -0.003, 0.002, 0.0])
   print(m.mean, m.std)

   rm = RunningMedian()
   rm.add_many([0.01, -0.02, 0.005, 0.0])
   print(rm.median)
