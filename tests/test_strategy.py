import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import pathlib

# sys.path.append(
#     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "new_src")
# )

from src.data_processor import DataProcessor
from src.factor_model import FactorModel
from src.alpha_signals import AlphaSignals
from src.backtester import Backtester
from src.risk_analyzer import RiskAnalyzer
from src.optimizer import Optimizer
from src.utils.logger import get_logger


class TestStrategyComponents(unittest.TestCase):
    """Unit tests for critical strategy functions."""

    def setUp(self):
        """Set up test data."""
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")

        np.random.seed(42)
        n_industries = 10
        self.test_returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, size=(len(dates), n_industries)),
            index=dates,
            columns=[f"Industry_{i}" for i in range(n_industries)],
        )

        self.test_factors = pd.DataFrame(
            {
                "MKT_RF": np.random.normal(0.0005, 0.015, len(dates)),
                "SMB": np.random.normal(0.0002, 0.01, len(dates)),
                "HML": np.random.normal(0.0003, 0.012, len(dates)),
                "RMW": np.random.normal(0.0001, 0.008, len(dates)),
                "CMA": np.random.normal(0.0002, 0.009, len(dates)),
                "RF": np.random.normal(0.0001, 0.001, len(dates)),
                "MOMENTUM": np.random.normal(0.0004, 0.011, len(dates)),
                "VOLATILITY": np.random.normal(0.02, 0.005, len(dates)),
            },
            index=dates,
        )

        # Test configuration
        self.test_config = {
            "transaction_cost_bps": 5,
            "risk_free_rate": 0.02,
            "lookback_periods": {
                "momentum": 252,
                "volatility": 63,
                "macro": 126,
                "sentiment": 21,
            },
        }

    def test_data_processor(self):
        """Test data processor functionality."""
        # Test data processor initialization
        processor = DataProcessor("/fake/path")
        self.assertIsNotNone(processor)
        # Fix: Compare as string since data_path is a Path object
        self.assertEqual(str(processor.data_path), "/fake/path")

    def test_factor_model(self):
        """Test factor model creation."""
        factor_model = FactorModel(self.test_config)

        # Test factor creation
        factors, industries = factor_model.build_comprehensive_model(
            {
                "industries": self.test_returns,
                "ff5": self.test_factors,
                "fedfunds": pd.DataFrame(
                    {"FEDFUNDS": [2.0] * len(self.test_returns)},
                    index=self.test_returns.index,
                ),
                "output_gap": pd.DataFrame(
                    {"GDPC1_GDPPOT": [0.5] * len(self.test_returns)},
                    index=self.test_returns.index,
                ),
                "nber": pd.DataFrame(
                    {"USREC": [0] * len(self.test_returns)},
                    index=self.test_returns.index,
                ),
            }
        )

        self.assertIsNotNone(factors)
        self.assertGreater(len(factors.columns), 0)
        self.assertEqual(len(factors), len(self.test_returns))

    def test_alpha_signals(self):
        """Test alpha signal creation."""
        signals = AlphaSignals(self.test_config)

        test_signals = signals.build_all_signals(self.test_returns, self.test_factors)

        self.assertIsNotNone(test_signals)
        self.assertGreater(len(test_signals), 0)

        combined = signals.combine_signals()
        self.assertIsNotNone(combined)
        self.assertEqual(combined.shape, self.test_returns.shape)

    def test_backtester(self):
        """Test backtesting functionality."""
        backtester = Backtester(self.test_config)

        test_signal = pd.DataFrame(
            np.random.normal(0, 1, size=self.test_returns.shape),
            index=self.test_returns.index,
            columns=self.test_returns.columns,
        )

        # Test backtest
        results = backtester.run_backtest(test_signal, self.test_returns)

        self.assertIsNotNone(results)
        self.assertIn("returns", results)
        self.assertIn("weights", results)
        self.assertIn("metrics", results)

        metrics = results["metrics"]
        self.assertIn("sharpe", metrics)
        self.assertIn("total_return", metrics)
        self.assertIn("max_drawdown", metrics)
        self.assertIn("volatility", metrics)

    def test_risk_analyzer(self):
        """Test risk analysis functionality."""
        analyzer = RiskAnalyzer(self.test_config)

        test_weights = pd.DataFrame(
            np.random.normal(0, 0.1, size=self.test_returns.shape),
            index=self.test_returns.index,
            columns=self.test_returns.columns,
        )

        attribution = analyzer.factor_attribution(
            self.test_returns.iloc[:, 0],
            self.test_factors,
            test_weights,
        )

        self.assertIsNotNone(attribution)
        self.assertIn("factor_exposures", attribution)
        self.assertIn("factor_contributions", attribution)
        self.assertIn("marginal_contributions", attribution)

    def test_optimizer(self):
        """Test optimizer functionality."""
        optimizer = Optimizer(self.test_config)

        test_signals = {
            "signal1": pd.DataFrame(
                np.random.normal(0, 1, size=self.test_returns.shape),
                index=self.test_returns.index,
                columns=self.test_returns.columns,
            ),
            "signal2": pd.DataFrame(
                np.random.normal(0, 1, size=self.test_returns.shape),
                index=self.test_returns.index,
                columns=self.test_returns.columns,
            ),
        }

        # Test signal combination
        combined = optimizer.combine_signals(
            test_signals, {"signal1": 0.6, "signal2": 0.4}
        )
        self.assertIsNotNone(combined)
        self.assertEqual(combined.shape, self.test_returns.shape)

        test_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        metrics = optimizer.calculate_performance_metrics(test_returns)

        self.assertIn("sharpe", metrics)
        self.assertIn("total_return", metrics)
        self.assertIn("volatility", metrics)

    def test_data_integrity(self):
        """Test data integrity and alignment."""
        # Test that returns and factors have same index
        self.assertEqual(len(self.test_returns), len(self.test_factors))
        self.assertTrue(all(self.test_returns.index == self.test_factors.index))

        self.assertTrue(self.test_returns.std().mean() > 0)
        self.assertTrue(
            self.test_returns.std().mean() < 1
        )

    def test_signal_properties(self):
        """Test signal mathematical properties."""
        signals = AlphaSignals(self.test_config)
        test_signals = signals.build_all_signals(self.test_returns, self.test_factors)

        for signal_name, signal_data in test_signals.items():
            self.assertTrue(
                signal_data.isna().sum().sum()
                < len(signal_data) * len(signal_data.columns) * 0.5
            )

            # Test that signals have reasonable scale
            signal_std = signal_data.std().mean()
            # Fix: Handle case where signal might have zero std (e.g., constant signals)
            if not np.isnan(signal_std):
                self.assertTrue(signal_std >= 0)  # Should be non-negative
                self.assertTrue(signal_std < 10)  # Should not be extremely large

    def test_backtest_consistency(self):
        """Test backtest consistency."""
        backtester = Backtester(self.test_config)

        # Create two identical signals
        signal1 = pd.DataFrame(
            np.random.normal(0, 1, size=self.test_returns.shape),
            index=self.test_returns.index,
            columns=self.test_returns.columns,
        )
        signal2 = signal1.copy()

        # Run backtests
        results1 = backtester.run_backtest(signal1, self.test_returns)
        results2 = backtester.run_backtest(signal2, self.test_returns)

        # Results should be identical
        np.testing.assert_array_almost_equal(
            results1["returns"].values, results2["returns"].values, decimal=10
        )

    def test_risk_metrics_bounds(self):
        """Test that risk metrics are within reasonable bounds."""
        backtester = Backtester(self.test_config)

        test_signal = pd.DataFrame(
            np.random.normal(0, 1, size=self.test_returns.shape),
            index=self.test_returns.index,
            columns=self.test_returns.columns,
        )

        results = backtester.run_backtest(test_signal, self.test_returns)
        metrics = results["metrics"]

        # Test Sharpe ratio bounds
        self.assertGreater(metrics["sharpe"], -10)
        self.assertLess(metrics["sharpe"], 10)

        # Test volatility bounds
        self.assertGreater(metrics["volatility"], 0)
        self.assertLess(metrics["volatility"], 2)

        # Test drawdown bounds
        self.assertLessEqual(metrics["max_drawdown"], 0)
        self.assertGreater(
            metrics["max_drawdown"], -1
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
