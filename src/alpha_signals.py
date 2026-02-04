import pandas as pd
import numpy as np
import warnings
from utils.logger import create_logger

warnings.filterwarnings("ignore")

class AlphaSignals:
    def __init__(self, config):
        self.config = config
        self.signals = {}
        self.optimized_weights = None
        self.optimized_lookbacks = None
        self.logger = create_logger("logs/alpha_signals.log")

    def generate_optimized_signals(
        self, aligned_data, optimize_weights=True, optimize_lookbacks=True
    ):
        self.logger.info("Generating optimized alpha signals")

        returns = aligned_data["industries"]
        factors = aligned_data.get("ff5", pd.DataFrame())

        self.logger.info("Creating momentum signal")
        momentum_signal = self.create_momentum_signal(returns)

        self.logger.info("Creating mean reversion signal")
        mean_reversion_signal = self.create_mean_reversion_signal(returns)

        self.logger.info("Creating volatility signal")
        volatility_signal = self.create_volatility_signal(returns)

        self.logger.info("Creating quality signal")
        quality_signal = self.create_quality_signal(returns)

        self.logger.info("Creating macro-regime signal")
        macro_regime_signal = self.create_macro_regime_signal(returns, factors)

        self.logger.info("Creating regime-adaptive signal")
        regime_adaptive_signal = self.create_regime_adaptive_signal(returns, factors)

        self.logger.info("Creating defensive signal")
        defensive_signal = self.create_defensive_signal(returns, factors)

        all_signals = {
            "momentum": momentum_signal,
            "mean_reversion": mean_reversion_signal,
            "volatility": volatility_signal,
            "quality": quality_signal,
            "macro_regime": macro_regime_signal,
            "regime_adaptive": regime_adaptive_signal,
            "defensive": defensive_signal,
        }

        initial_weights = {
            "momentum": 0.15,
            "mean_reversion": 0.15,
            "volatility": 0.15,
            "quality": 0.15,
            "macro_regime": 0.1,
            "regime_adaptive": 0.1,
            "defensive": 0.2,
        }

        if optimize_weights:
            self.logger.info("Optimizing signal weights with Optuna")
            from optimizer import Optimizer

            optimizer = Optimizer(self.config)

            optimized_weights = optimizer.optimize_signal_weights(
                all_signals, returns, factors, n_trials=30
            )

            combined_signal = pd.DataFrame(
                0, index=returns.index, columns=returns.columns
            )
            for signal_name, weight in optimized_weights.items():
                if signal_name in all_signals:
                    combined_signal += all_signals[signal_name] * weight

            self.optimized_weights = optimized_weights
        else:
            combined_signal = (
                momentum_signal * initial_weights["momentum"]
                + mean_reversion_signal * initial_weights["mean_reversion"]
                + volatility_signal * initial_weights["volatility"]
                + quality_signal * initial_weights["quality"]
                + macro_regime_signal * initial_weights["macro_regime"]
                + regime_adaptive_signal * initial_weights["regime_adaptive"]
                + defensive_signal * initial_weights["defensive"]
            )
            self.optimized_weights = initial_weights

        if combined_signal.std().max() < 1e-8:
            self.logger.warning(
                "Combined signal has no variation, using individual signals"
            )
            signal_vars = {
                name: signal.std().mean() for name, signal in all_signals.items()
            }
            best_signal = max(signal_vars.items(), key=lambda x: x[1])
            self.logger.info(
                f"Using {best_signal[0]} signal with std {best_signal[1]:.6f}"
            )
            combined_signal = all_signals[best_signal[0]]

        final_signal = self.apply_cross_sectional_ranking(combined_signal)

        self.optimized_lookbacks = {
            "momentum_short": 63,
            "momentum_long": 252,
            "mean_reversion": 252,
            "volatility": 21,
            "quality": 126,
            "valuation": 252,
        }

        self.logger.info("Generated optimized alpha signals")
        self.logger.info(f"Signal shape: {final_signal.shape}")
        self.logger.info(f"Optimized weights: {self.optimized_weights}")
        self.logger.info(f"Optimized lookbacks: {self.optimized_lookbacks}")
        self.logger.info(
            f"Signal range: {final_signal.min().min():.3f} to {final_signal.max().max():.3f}"
        )
        self.logger.info(f"Signal std: {final_signal.std().mean():.3f}")
        self.logger.info(
            f"Non-zero signals: {(final_signal != 0).sum().sum()} out of {final_signal.size}"
        )

        return final_signal

    def create_momentum_signal(self, returns):
        short_window = 63
        long_window = 252

        short_momentum = returns.rolling(short_window, min_periods=1).mean()
        long_momentum = returns.rolling(long_window, min_periods=1).mean()

        momentum_signal = short_momentum - long_momentum

        momentum_signal = momentum_signal.rolling(252, min_periods=1).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8)
        )

        return momentum_signal.fillna(0)

    def create_mean_reversion_signal(self, returns):
        lookback = 252

        rolling_mean = returns.rolling(lookback, min_periods=1).mean()
        mean_reversion = returns - rolling_mean

        mean_reversion_signal = mean_reversion.rolling(252, min_periods=1).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8)
        )

        return mean_reversion_signal.fillna(0)

    def create_volatility_signal(self, returns):
        lookback = 21

        rolling_vol = returns.rolling(lookback, min_periods=1).std()

        volatility_signal = -rolling_vol

        volatility_signal = volatility_signal.rolling(252, min_periods=1).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8)
        )

        return volatility_signal.fillna(0)

    def create_quality_signal(self, returns):
        lookback = 126

        rolling_mean = returns.rolling(lookback, min_periods=1).mean()
        rolling_std = returns.rolling(lookback, min_periods=1).std()
        quality_signal = rolling_mean / (rolling_std + 1e-8)

        quality_signal = quality_signal.rolling(252, min_periods=1).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8)
        )

        return quality_signal.fillna(0)

    def create_macro_regime_signal(self, returns, factors):
        if factors.empty:
            return pd.DataFrame(0, index=returns.index, columns=returns.columns)

        market_factor = factors.get("MKT_RF", pd.Series(0, index=factors.index))

        regime_signal = pd.DataFrame(0, index=returns.index, columns=returns.columns)

        for col in returns.columns:
            correlation = returns[col].rolling(252, min_periods=1).corr(market_factor)
            regime_signal[col] = correlation

        return regime_signal.fillna(0)

    def create_regime_adaptive_signal(self, returns, factors):
        if factors.empty:
            return pd.DataFrame(0, index=returns.index, columns=returns.columns)

        market_factor = factors.get("MKT_RF", pd.Series(0, index=factors.index))
        momentum_factor = factors.get("MOMENTUM", pd.Series(0, index=factors.index))

        regime_signal = pd.DataFrame(0, index=returns.index, columns=returns.columns)

        for col in returns.columns:
            market_corr = returns[col].rolling(126, min_periods=1).corr(market_factor)
            momentum_corr = (
                returns[col].rolling(126, min_periods=1).corr(momentum_factor)
            )
            regime_signal[col] = (market_corr + momentum_corr) / 2

        return regime_signal.fillna(0)

    def create_defensive_signal(self, returns, factors):
        self.logger.info(
            "Creating defensive signal for industries based on market stress"
        )

        market_factor = factors.get("MKT_RF", pd.Series(0, index=factors.index))

        market_vol = market_factor.rolling(63, min_periods=1).std()
        market_momentum = market_factor.rolling(252, min_periods=1).mean()
        market_drawdown = market_factor.rolling(252, min_periods=1).min()

        stress_indicator = (
            -market_vol.rolling(21, min_periods=1).mean()
            - market_momentum.rolling(21, min_periods=1).mean()
            - market_drawdown.rolling(21, min_periods=1).mean()
        )

        stress_normalized = stress_indicator.rolling(252, min_periods=1).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8)
        )

        defensive_signal = pd.DataFrame(0, index=returns.index, columns=returns.columns)

        defensive_industries = [
            "Food",
            "Soda",
            "Beer",
            "Smoke",
            "Hlth",
            "MedEq",
            "Drugs",
        ]

        for col in returns.columns:
            if col in defensive_industries:
                defensive_signal[col] = stress_normalized
            else:
                correlation = (
                    returns[col].rolling(126, min_periods=1).corr(stress_indicator)
                )
                defensive_signal[col] = -correlation

        defensive_signal = defensive_signal.fillna(0)

        self.logger.info("Created defensive signal for industries")
        self.logger.info(
            f"Defensive industries signal range: {defensive_signal[defensive_industries].min().min():.3f} to {defensive_signal[defensive_industries].max().max():.3f}"
        )
        self.logger.info(
            f"Average defensive signal: {defensive_signal[defensive_industries].mean().mean():.3f}"
        )

        return defensive_signal

    def apply_cross_sectional_ranking(self, signal):
        signal = signal.fillna(method="ffill").fillna(0)

        if signal.std().max() < 1e-8:
            self.logger.warning("Signal has no variation, skipping ranking")
            return signal

        ranked_signal = signal.rank(axis=1, pct=True) * 2 - 1

        if ranked_signal.std().max() < 1e-8:
            self.logger.warning("Ranked signal has no variation, using raw signal")
            return signal

        return ranked_signal

    def map_industry_signals_to_etfs(self, industry_signals, etf_returns):
        self.logger.info("Mapping industry signals to ETFs")

        common_dates = industry_signals.index.intersection(etf_returns.index)
        if len(common_dates) == 0:
            self.logger.warning("No common dates for signal mapping")
            return pd.DataFrame()

        industry_aligned = industry_signals.loc[common_dates]
        etf_aligned = etf_returns.loc[common_dates]

        self.logger.info(f"Aligned {len(common_dates)} dates for signal mapping")
        self.logger.info(f"Industry signals: {industry_aligned.shape}")
        self.logger.info(f"ETF returns: {etf_aligned.shape}")

        etf_signals = pd.DataFrame(0, index=common_dates, columns=etf_aligned.columns)

        for etf in etf_aligned.columns:
            if etf == "SPY":
                base_signals = industry_aligned.mean(axis=1)
                defensive_industries = [
                    "Food",
                    "Soda",
                    "Beer",
                    "Smoke",
                    "Hlth",
                    "MedEq",
                    "Drugs",
                ]
                defensive_mask = industry_aligned.columns.isin(defensive_industries)
                if defensive_mask.any():
                    defensive_signals = industry_aligned.loc[:, defensive_mask].mean(
                        axis=1
                    )
                    contrarian_factor = 1 - (defensive_signals.abs() * 0.3)
                    etf_signals[etf] = base_signals * contrarian_factor
                else:
                    etf_signals[etf] = base_signals
            elif etf in ["XLP", "XLU", "XLV"]:
                defensive_industries = [
                    "Food",
                    "Soda",
                    "Beer",
                    "Smoke",
                    "Hlth",
                    "MedEq",
                    "Drugs",
                ]
                defensive_mask = industry_aligned.columns.isin(defensive_industries)
                if defensive_mask.any():
                    defensive_signals = industry_aligned.loc[:, defensive_mask].mean(
                        axis=1
                    )
                    other_signals = industry_aligned.loc[:, ~defensive_mask].mean(
                        axis=1
                    )
                    etf_signals[etf] = 0.6 * defensive_signals + 0.4 * other_signals
                else:
                    etf_signals[etf] = industry_aligned.mean(axis=1)
            elif etf == "QQQ":
                tech_industries = ["Hardw", "Softw", "Chips"]
                tech_mask = industry_aligned.columns.isin(tech_industries)
                if tech_mask.any():
                    tech_signals = industry_aligned.loc[:, tech_mask].mean(axis=1)
                    other_signals = industry_aligned.loc[:, ~tech_mask].mean(axis=1)
                    etf_signals[etf] = 0.7 * tech_signals + 0.3 * other_signals
                else:
                    etf_signals[etf] = industry_aligned.mean(axis=1)
            else:
                etf_signals[etf] = industry_aligned.mean(axis=1)

        signal_std = etf_signals.std()
        if signal_std.max() > 1e-8:
            etf_signals = (etf_signals - etf_signals.mean()) / signal_std
        else:
            self.logger.warning(
                "All signals have zero standard deviation, using raw signals"
            )
            pass

        self.logger.info(f"Mapped signals to {len(etf_signals.columns)} ETFs")
        self.logger.info(
            f"Signal range: {etf_signals.min().min():.3f} to {etf_signals.max().max():.3f}"
        )
        self.logger.info(f"Signal std: {etf_signals.std().mean():.3f}")

        if "SPY" in etf_signals.columns:
            self.logger.info(f"SPY signal: {etf_signals['SPY'].mean():.3f} (mean)")
        else:
            self.logger.info("SPY not included in portfolio")

        if "QQQ" in etf_signals.columns:
            self.logger.info(f"QQQ signal: {etf_signals['QQQ'].mean():.3f} (mean)")
        else:
            self.logger.info("QQQ not included in portfolio")

        defensive_etfs = ["XLP", "XLU", "XLV"]
        existing_defensive = [
            etf for etf in defensive_etfs if etf in etf_signals.columns
        ]
        if existing_defensive:
            self.logger.info(
                f"Defensive ETFs average: {etf_signals[existing_defensive].mean().mean():.3f}"
            )
        else:
            self.logger.info("No defensive ETFs in portfolio")

        return etf_signals
