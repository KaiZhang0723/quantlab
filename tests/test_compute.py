"""Tests for the compute subpackage."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quantlab.compute.backtest import _backtest_one, momentum_strategy, run_backtest
from quantlab.compute.montecarlo import (
    _split_evenly,
    historical_simulation,
    monte_carlo_var,
)
from quantlab.compute.optimal_execution import (
    _greedy_unlimited,
    max_profit_with_fee,
    max_profit_with_k_trades,
)
from quantlab.compute.rolling import (
    cumulative_returns,
    drawdown_series,
    log_returns,
    max_drawdown,
    rolling_sharpe,
    rolling_volatility,
)
from quantlab.compute.sector_aggregate_mr import (
    SectorVolumeMR,
    emit_csv,
    run_inline,
)
from quantlab.exceptions import InsufficientHistoryError


class RollingTests(unittest.TestCase):
    def test_log_returns_basic(self) -> None:
        prices = pd.Series([100.0, 110.0, 121.0])
        rets = log_returns(prices)
        self.assertEqual(len(rets), 2)
        self.assertAlmostEqual(rets.iloc[0], np.log(1.1))

    def test_log_returns_requires_two(self) -> None:
        with self.assertRaises(InsufficientHistoryError):
            log_returns(pd.Series([100.0]))

    def test_cumulative_returns(self) -> None:
        rets = pd.Series([0.0, 0.1, -0.05])
        cum = cumulative_returns(rets)
        self.assertAlmostEqual(cum.iloc[-1], np.exp(0.05))

    def test_rolling_volatility_window_validation(self) -> None:
        with self.assertRaises(ValueError):
            rolling_volatility(pd.Series([1.0, 2.0, 3.0]), window=1)

    def test_rolling_sharpe_window_validation(self) -> None:
        with self.assertRaises(ValueError):
            rolling_sharpe(pd.Series([1.0]), window=1)

    def test_drawdown_series_known_path(self) -> None:
        equity = pd.Series([1.0, 1.2, 0.9, 1.3])
        dd = drawdown_series(equity)
        self.assertAlmostEqual(dd.iloc[2], 0.9 / 1.2 - 1.0)

    def test_max_drawdown(self) -> None:
        equity = pd.Series([1.0, 1.5, 0.75, 1.0])
        self.assertAlmostEqual(max_drawdown(equity), -0.5)

    def test_max_drawdown_empty(self) -> None:
        self.assertEqual(max_drawdown(pd.Series([], dtype=float)), 0.0)

    def test_rolling_volatility_annualised_factor(self) -> None:
        rets = pd.Series(np.random.default_rng(0).normal(0, 0.01, 200))
        vol_d = rolling_volatility(rets, window=20, annualised=False).dropna()
        vol_a = rolling_volatility(rets, window=20, annualised=True).dropna()
        ratio = float((vol_a / vol_d).mean())
        self.assertAlmostEqual(ratio, np.sqrt(252), places=4)


class OptimalExecutionTests(unittest.TestCase):
    def test_max_profit_with_fee_known(self) -> None:
        self.assertEqual(max_profit_with_fee([1, 3, 2, 8, 4, 9], fee=2), 8)

    def test_max_profit_with_fee_zero_fee_matches_unlimited(self) -> None:
        prices = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        self.assertEqual(max_profit_with_fee(prices, fee=0.0), _greedy_unlimited(prices))

    def test_max_profit_with_fee_invalid_inputs(self) -> None:
        with self.assertRaises(InsufficientHistoryError):
            max_profit_with_fee([])
        with self.assertRaises(ValueError):
            max_profit_with_fee([1, 2, 3], fee=-1)

    def test_max_profit_with_k_trades_known(self) -> None:
        # Classic LeetCode example.
        self.assertEqual(max_profit_with_k_trades([3, 2, 6, 5, 0, 3], k=2), 7)

    def test_max_profit_with_k_unlimited_branch(self) -> None:
        prices = [1, 2, 3, 4, 5]
        self.assertEqual(max_profit_with_k_trades(prices, k=10), 4)

    def test_max_profit_with_k_zero_trades(self) -> None:
        self.assertEqual(max_profit_with_k_trades([1, 5, 3], k=0), 0.0)

    def test_max_profit_with_k_invalid(self) -> None:
        with self.assertRaises(ValueError):
            max_profit_with_k_trades([1, 2], k=-1)


class MonteCarloTests(unittest.TestCase):
    def test_split_evenly_handles_remainder(self) -> None:
        self.assertEqual(_split_evenly(10, 3), [4, 3, 3])
        self.assertEqual(sum(_split_evenly(101, 7)), 101)

    def test_split_evenly_low_n(self) -> None:
        # Mirrors Ed #66 — when N < workers we should still emit non-zero chunks.
        out = _split_evenly(2, 4)
        self.assertEqual(sum(out), 2)
        self.assertTrue(all(c > 0 for c in out))

    def test_split_evenly_invalid(self) -> None:
        with self.assertRaises(ValueError):
            _split_evenly(10, 0)

    def test_monte_carlo_var_serial(self) -> None:
        res = monte_carlo_var(mu=0.0005, sigma=0.012, horizon_days=10,
                              n_paths=4000, confidence=0.95, n_workers=1, seed=1)
        self.assertGreater(res.var, 0)
        self.assertGreaterEqual(res.cvar, res.var)
        self.assertEqual(res.n_paths, 4000)

    def test_monte_carlo_var_parallel_matches_serial_within_tolerance(self) -> None:
        # The seeds are split per-chunk, so parallel != serial exactly, but the
        # statistic should be very close on a large sample.
        s = monte_carlo_var(mu=0.0, sigma=0.01, horizon_days=5,
                            n_paths=20_000, confidence=0.99, n_workers=1, seed=7)
        p = monte_carlo_var(mu=0.0, sigma=0.01, horizon_days=5,
                            n_paths=20_000, confidence=0.99, n_workers=2, seed=7)
        self.assertLess(abs(s.var - p.var), 0.01)

    def test_monte_carlo_var_invalid(self) -> None:
        with self.assertRaises(ValueError):
            monte_carlo_var(mu=0, sigma=0.01, horizon_days=1, n_paths=10, confidence=1.0)
        with self.assertRaises(ValueError):
            monte_carlo_var(mu=0, sigma=0.01, horizon_days=1, n_paths=0)
        with self.assertRaises(ValueError):
            monte_carlo_var(mu=0, sigma=0.01, horizon_days=0, n_paths=10)

    def test_historical_simulation(self) -> None:
        rng = np.random.default_rng(0)
        rets = rng.normal(0, 0.01, 5_000)
        res = historical_simulation(rets, confidence=0.95)
        self.assertGreater(res.var, 0)

    def test_historical_simulation_too_few(self) -> None:
        with self.assertRaises(InsufficientHistoryError):
            historical_simulation([0.01])


class BacktestTests(unittest.TestCase):
    def test_run_backtest_serial(self, ) -> None:
        from tests.conftest import _synthetic_panel
        prices = _synthetic_panel(("AAA", "BBB"), n_days=400)
        result = run_backtest(prices, n_workers=1)
        self.assertEqual({"AAA", "BBB"}, set(result.per_ticker.keys()))
        self.assertFalse(result.metrics_table().empty)

    def test_run_backtest_parallel(self) -> None:
        from tests.conftest import _synthetic_panel
        prices = _synthetic_panel(("AAA", "BBB", "CCC"), n_days=400)
        result = run_backtest(prices, n_workers=2)
        self.assertEqual(3, len(result.per_ticker))

    def test_run_backtest_empty_panel(self) -> None:
        with self.assertRaises(InsufficientHistoryError):
            run_backtest(pd.DataFrame())

    def test_strategy_short_history_returns_zeros(self) -> None:
        from tests.conftest import _synthetic_panel
        prices = _synthetic_panel(("AAA",), n_days=100).set_index("date")
        sig = momentum_strategy(prices, lookback=252)
        self.assertTrue((sig == 0).all())

    def test_backtest_one_short_history_raises(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2))
        with self.assertRaises(InsufficientHistoryError):
            _backtest_one(("X", df, momentum_strategy))


class SectorAggregateMRTests(unittest.TestCase):
    def test_run_inline_sums_per_sector(self) -> None:
        rows = [
            ("2024-01-02", "AAPL", "Tech", 1000.0),
            ("2024-01-02", "MSFT", "Tech", 500.0),
            ("2024-01-02", "JPM", "Finance", 750.0),
        ]
        out = run_inline(rows)
        self.assertAlmostEqual(out["Tech"], 1500.0)
        self.assertAlmostEqual(out["Finance"], 750.0)

    def test_mapper_emits_correct_kv(self) -> None:
        # Pass empty args explicitly so MRJob doesn't read pytest's sys.argv.
        job = SectorVolumeMR(args=[])
        out = list(job.mapper_emit_sector_volume(None, "2024-01-02,AAPL,Tech,1000"))
        self.assertEqual(out, [("Tech", 1000.0)])

    def test_mapper_skips_header_and_garbage(self) -> None:
        job = SectorVolumeMR(args=[])
        self.assertEqual(list(job.mapper_emit_sector_volume(None, "date,ticker,sector,volume")), [])
        self.assertEqual(list(job.mapper_emit_sector_volume(None, "")), [])
        self.assertEqual(list(job.mapper_emit_sector_volume(None, "bad,row")), [])

    def test_reducer_sums(self) -> None:
        job = SectorVolumeMR(args=[])
        out = list(job.reducer_sum_volume("Tech", iter([1.0, 2.0, 3.0])))
        self.assertEqual(out, [("Tech", 6.0)])

    def test_emit_csv(self) -> None:
        text = emit_csv([("d", "t", "s", 1.0)])
        self.assertIn("date,ticker,sector,volume", text)
        self.assertIn("d,t,s,1.0", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
