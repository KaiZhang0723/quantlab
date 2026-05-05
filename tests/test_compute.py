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
    MRJOB_AVAILABLE,
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
        self.assertAlmostEqual(rets.iloc[1], np.log(121.0 / 110.0))

    def test_log_returns_requires_two(self) -> None:
        with self.assertRaises(InsufficientHistoryError):
            log_returns(pd.Series([100.0]))

    def test_log_returns_rejects_nonpositive(self) -> None:
        with self.assertRaises(ValueError):
            log_returns(pd.Series([100.0, 0.0, 1.0]))
        with self.assertRaises(ValueError):
            log_returns(pd.Series([100.0, -5.0, 1.0]))

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
        self.assertEqual(res.horizon_days, 1)

    def test_historical_simulation_too_few(self) -> None:
        with self.assertRaises(InsufficientHistoryError):
            historical_simulation([0.01])

    def test_historical_simulation_horizon_aggregates_returns(self) -> None:
        # 10-day VaR should be roughly sqrt(10) larger than 1-day VaR for
        # iid returns (square-root scaling).
        rng = np.random.default_rng(0)
        rets = rng.normal(0, 0.01, 10_000)
        one_day = historical_simulation(rets, confidence=0.99, horizon_days=1)
        ten_day = historical_simulation(rets, confidence=0.99, horizon_days=10)
        self.assertEqual(ten_day.horizon_days, 10)
        ratio = ten_day.var / one_day.var
        self.assertAlmostEqual(ratio, np.sqrt(10), delta=0.5)

    def test_historical_simulation_invalid_horizon(self) -> None:
        with self.assertRaises(ValueError):
            historical_simulation([0.01, 0.02, 0.03], horizon_days=0)


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

    def test_backtest_pnl_math_matches_hand_computation(self) -> None:
        # Five-day price path: 100, 101, 99, 105, 102.
        # Strategy = always long (position = 1.0). Under the no-look-ahead
        # convention (positions.shift(1) * rets), each day's return enters PnL.
        prices = pd.DataFrame(
            {"close": [100.0, 101.0, 99.0, 105.0, 102.0]},
            index=pd.date_range("2024-01-02", periods=5, freq="B"),
        )

        def always_long(df: pd.DataFrame) -> pd.Series:
            return pd.Series(1.0, index=df.index)

        res = _backtest_one(("X", prices, always_long))
        # Expected log returns (the 4 inter-day moves):
        expected_logrets = np.log([101 / 100, 99 / 101, 105 / 99, 102 / 105])
        # Final equity = exp(sum of expected log returns).
        expected_equity = float(np.exp(expected_logrets.sum()))
        self.assertAlmostEqual(float(res.equity_curve.iloc[-1]), expected_equity, places=10)
        self.assertEqual(len(res.equity_curve), 4)

    def test_backtest_lookahead_protected(self) -> None:
        # A "cheating" strategy that knows today's close should NOT earn
        # the absolute mean return: under the lagged-position convention,
        # signals from day t are applied to day t+1's return, which the
        # strategy cannot see.
        rng = np.random.default_rng(7)
        path = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
        prices = pd.DataFrame(
            {"close": path},
            index=pd.date_range("2024-01-02", periods=200, freq="B"),
        )

        def perfect_foresight_today(df: pd.DataFrame) -> pd.Series:
            # Position at t = sign of return realised at t (uses close[t]).
            rets_today = np.log(df["close"] / df["close"].shift(1))
            return rets_today.fillna(0).apply(lambda r: 1.0 if r > 0 else -1.0)

        res = _backtest_one(("X", prices, perfect_foresight_today))
        # If alignment were wrong (positions[t] * rets[t]), this strategy
        # would earn the sum of |return|s — a very high Sharpe.
        # With correct alignment (positions[t-1] * rets[t]), the cheat's
        # signal is one day stale, so Sharpe should NOT explode to >50.
        self.assertLess(res.sharpe, 5.0)


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

    @unittest.skipUnless(MRJOB_AVAILABLE, "mrjob unavailable on this Python version")
    def test_mapper_emits_correct_kv(self) -> None:
        # Pass empty args explicitly so MRJob doesn't read pytest's sys.argv.
        job = SectorVolumeMR(args=[])
        out = list(job.mapper_emit_sector_volume(None, "2024-01-02,AAPL,Tech,1000"))
        self.assertEqual(out, [("Tech", 1000.0)])

    @unittest.skipUnless(MRJOB_AVAILABLE, "mrjob unavailable on this Python version")
    def test_mapper_skips_header_and_garbage(self) -> None:
        job = SectorVolumeMR(args=[])
        self.assertEqual(list(job.mapper_emit_sector_volume(None, "date,ticker,sector,volume")), [])
        self.assertEqual(list(job.mapper_emit_sector_volume(None, "")), [])
        self.assertEqual(list(job.mapper_emit_sector_volume(None, "bad,row")), [])

    @unittest.skipUnless(MRJOB_AVAILABLE, "mrjob unavailable on this Python version")
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
