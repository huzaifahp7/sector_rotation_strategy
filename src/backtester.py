import pandas as pd
import numpy as np
import warnings
from utils.logger import create_logger
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")


class Backtester:
    def __init__(self, config):
        self.config = config
        self.results = None
        self.prev_weights = None
        self.logger = create_logger("logs/backtester.log")

    def calculate_weights(self, alpha_signal, returns, method="quintile"):
        self.logger.info("Calculating weights using quintile method...")

        common_dates = alpha_signal.index.intersection(returns.index)
        if len(common_dates) == 0:
            self.logger.warning("No common dates for weight calculation")
            return pd.DataFrame()

        signal_aligned = alpha_signal.loc[common_dates]
        returns_aligned = returns.loc[common_dates]

        self.logger.info(f"Aligned {len(common_dates)} dates for weight calculation")
        self.logger.info(f"Signal shape: {signal_aligned.shape}")
        self.logger.info(f"Returns shape: {returns_aligned.shape}")

        weights = pd.DataFrame(0, index=common_dates, columns=returns_aligned.columns)

        for date in common_dates:
            signals = signal_aligned.loc[date]
            if not signals.isna().all():
                ranked = signals.rank(ascending=True, method="first")
                quintiles = pd.cut(ranked, bins=5, labels=False, include_lowest=True)

                for i, asset in enumerate(returns_aligned.columns):
                    if i < len(quintiles):
                        quintile = quintiles.iloc[i]
                        if pd.notna(quintile):
                            if quintile == 4:
                                weight = 0.2
                            elif quintile == 3:
                                weight = 0.1
                            elif quintile == 1:
                                weight = -0.1
                            elif quintile == 0:
                                weight = -0.2
                            else:
                                weight = 0.0
                            weights.loc[date, asset] = weight

        self.logger.info(f"Weight distribution summary:")
        self.logger.info(f"Mean weight per asset: {weights.mean().describe()}")
        self.logger.info(f"Long positions: {(weights > 0).sum().sum()} instances")
        self.logger.info(f"Short positions: {(weights < 0).sum().sum()} instances")
        self.logger.info(f"Neutral positions: {(weights == 0).sum().sum()} instances")

        return weights

    def apply_risk_management(self, weights, returns, target_vol=0.12):
        self.logger.info("Applying risk management...")

        if weights is None or weights.empty:
            self.logger.warning("No weights provided for risk management")
            return weights

        if target_vol > 0:
            self.logger.info(f"Applying volatility targeting: {target_vol:.1%}")
            vol_scaled_weights = self.volatility_targeting(weights, returns, target_vol)
            weights = vol_scaled_weights

        max_position = self.config.get("max_position_size", 0.1)
        self.logger.info(f"Applying position size limits: max {max_position:.1%}")
        weights = weights.clip(-max_position, max_position)

        weights = self.apply_capacity_constraints(weights, returns)

        turnover_threshold = self.config.get("turnover_threshold", 0.1)
        if hasattr(self, "prev_weights") and self.prev_weights is not None:
            turnover = (weights - self.prev_weights).abs().sum(axis=1)
            high_turnover = turnover > turnover_threshold
            if high_turnover.any():
                self.logger.warning(
                    f"High turnover detected on {high_turnover.sum()} dates"
                )
                weights.loc[high_turnover] *= 0.5

        self.prev_weights = weights.copy()

        drawdown_limit = self.config.get("drawdown_limit", 0.20)
        if hasattr(self, "prev_weights") and self.prev_weights is not None:
            portfolio_returns = (weights * returns).sum(axis=1)
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.cummax()
            current_drawdown = (cumulative_returns - running_max) / running_max

            high_drawdown = current_drawdown < -drawdown_limit
            if high_drawdown.any():
                self.logger.warning(
                    f"Drawdown limit exceeded on {high_drawdown.sum()} dates"
                )
                weights.loc[high_drawdown] *= 0.5

        net_exposure = weights.sum(axis=1).abs()
        high_exposure = net_exposure > 0.5
        if high_exposure.any():
            self.logger.warning(
                f"High net exposure detected on {high_exposure.sum()} dates"
            )
            for date in weights.index[high_exposure]:
                scaling_factor = 0.5 / net_exposure.loc[date]
                weights.loc[date] *= scaling_factor

        self.logger.info("Risk management applied")
        self.logger.info(
            f"Final weight range: {weights.min().min():.3f} to {weights.max().max():.3f}"
        )
        self.logger.info(f"Mean weight per asset: {weights.mean().mean():.3f}")
        self.logger.info(
            f"Average net exposure: {weights.sum(axis=1).abs().mean():.3f}"
        )

        return weights

    def apply_capacity_constraints(self, weights, returns):
        self.logger.info("Applying capacity and liquidity constraints...")

        adv_proxy = returns.rolling(252, min_periods=30).std() * np.sqrt(252)

        capacity_limits = 1 / (adv_proxy + 1e-8)
        capacity_limits = capacity_limits.clip(0.01, 0.5)

        for date in weights.index:
            for asset in weights.columns:
                if asset in capacity_limits.columns:
                    max_weight = capacity_limits.loc[date, asset]
                    current_weight = weights.loc[date, asset]
                    if abs(current_weight) > max_weight:
                        sign = 1 if current_weight > 0 else -1
                        weights.loc[date, asset] = sign * max_weight

        total_capacity = capacity_limits.mean().mean()
        avg_weight = weights.abs().mean().mean()
        capacity_utilization = avg_weight / total_capacity if total_capacity > 0 else 0

        self.logger.info(f"Capacity utilization: {capacity_utilization:.1%}")
        self.logger.info(f"Average capacity limit: {total_capacity:.1%}")
        self.logger.info(f"Average position size: {avg_weight:.1%}")

        return weights

    def volatility_targeting(self, weights, returns, target_vol):
        portfolio_returns = (weights * returns).sum(axis=1)
        rolling_vol = portfolio_returns.rolling(252, min_periods=30).std() * np.sqrt(
            252
        )
        scaling = target_vol / (rolling_vol + 1e-8)
        scaling = scaling.clip(0.5, 3.0)
        weights_scaled = weights.multiply(scaling, axis=0)
        return weights_scaled

    def calculate_alpha_and_tstat(self, strategy_returns, benchmark_returns):
        common_dates = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_aligned = strategy_returns.loc[common_dates]
        benchmark_aligned = benchmark_returns.loc[common_dates]

        if len(strategy_aligned) == 0 or strategy_aligned.isna().all():
            return 0.0, 0.0

        excess_returns = strategy_aligned - benchmark_aligned
        excess_returns = excess_returns.dropna()

        if len(excess_returns) == 0:
            return 0.0, 0.0

        X = np.ones((len(excess_returns), 1))
        y = excess_returns.values

        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
            alpha = beta

            residuals = y - alpha
            n = len(residuals)
            mse = np.sum(residuals**2) / (n - 1)
            se_alpha = np.sqrt(mse / n)
            t_stat = alpha / se_alpha if se_alpha > 0 else 0

            return alpha, t_stat
        except:
            return 0.0, 0.0

    def run_backtest(self, alpha_signal, returns, factors=None, method="quintile"):
        self.logger.info("Running backtest...")

        self.returns = returns

        common_dates = alpha_signal.index.intersection(returns.index)
        if len(common_dates) == 0:
            self.logger.warning("No common dates between signals and returns")
            return None

        alpha_aligned = alpha_signal.loc[common_dates]
        returns_aligned = returns.loc[common_dates]

        self.logger.info(f"Alpha signal shape: {alpha_aligned.shape}")
        self.logger.info(f"Returns shape: {returns_aligned.shape}")
        self.logger.info(
            f"Alpha signal date range: {alpha_aligned.index.min()} to {alpha_aligned.index.max()}"
        )
        self.logger.info(
            f"Returns date range: {returns_aligned.index.min()} to {returns_aligned.index.max()}"
        )
        self.logger.info(
            f"Alpha signal NaN count: {alpha_aligned.isnull().sum().sum()}"
        )
        self.logger.info(f"Returns NaN count: {returns_aligned.isnull().sum().sum()}")
        self.logger.info(f"Common dates: {len(common_dates)}")

        weights = self.calculate_weights(alpha_aligned, returns_aligned, method)

        weights = self.apply_risk_management(
            weights,
            returns_aligned,
            target_vol=self.config.get("target_volatility", 0.12),
        )

        portfolio_returns = (weights * returns_aligned).sum(axis=1)

        total_return = (1 + portfolio_returns).prod() - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe = (
            portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
            if portfolio_returns.std() > 0
            else 0
        )
        drawdown_series = self.calculate_drawdown(portfolio_returns)
        max_drawdown = drawdown_series.min()

        self.results = {
            "returns": portfolio_returns,
            "weights": weights,
            "metrics": {
                "total_return": total_return,
                "volatility": volatility,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
            },
        }

        if "SPY" in returns_aligned.columns:
            spy_returns = returns_aligned["SPY"]
            alpha, t_stat = self.calculate_alpha_and_tstat(
                portfolio_returns, spy_returns
            )
            self.results["alpha_info"] = {
                "alpha": alpha,
                "t_statistic": t_stat,
                "benchmark": "SPY",
            }
            self.logger.info(f"Alpha vs SPY: {alpha:.2%} (t-stat: {t_stat:.3f})")
        else:
            self.results["alpha_info"] = {
                "alpha": 0,
                "t_statistic": 0,
                "benchmark": "None",
            }

        self.logger.info("Backtest Results:")
        self.logger.info(f"Sharpe Ratio: {sharpe:.3f}")
        self.logger.info(f"Total Return: {total_return:.2%}")
        self.logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        self.logger.info(f"Annual Volatility: {volatility:.2%}")

        return self.results

    def calculate_drawdown(self, returns=None):
        if returns is None:
            if hasattr(self, "results") and "returns" in self.results:
                returns = self.results["returns"]
            else:
                raise ValueError("No returns provided or available in self.results.")
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown

    def plot_results(self, filepath):
        if not self.results:
            self.logger.warning("No results to plot")
            return

        returns = self.results["returns"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Backtest Results", fontsize=16, fontweight="bold")

        cumulative_returns = (1 + returns).cumprod()
        axes[0, 0].plot(
            cumulative_returns.index, cumulative_returns.values, linewidth=2
        )
        axes[0, 0].set_title("Cumulative Returns")
        axes[0, 0].set_ylabel("Cumulative Return")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        drawdown = self.calculate_drawdown()
        axes[0, 1].fill_between(
            drawdown.index, drawdown.values, 0, alpha=0.3, color="red"
        )
        axes[0, 1].set_title("Drawdown")
        axes[0, 1].set_ylabel("Drawdown (%)")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        rolling_vol = returns.rolling(252).std() * np.sqrt(252)
        axes[1, 0].plot(rolling_vol.index, rolling_vol.values, linewidth=2)
        axes[1, 0].set_title("Rolling Volatility (252-day)")
        axes[1, 0].set_ylabel("Annualized Volatility")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        rolling_sharpe = (
            returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
        )
        axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
        axes[1, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[1, 1].set_title("Rolling Sharpe Ratio (252-day)")
        axes[1, 1].set_ylabel("Sharpe Ratio")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Backtest results plot saved to {filepath}")

    def export_results(self, filepath_prefix):
        if not self.results:
            self.logger.warning("No results to export")
            return

        returns_file = f"{filepath_prefix}_returns.csv"
        self.results["returns"].to_csv(returns_file)

        weights_file = f"{filepath_prefix}_weights.csv"
        self.results["weights"].to_csv(weights_file)

        metrics_file = f"{filepath_prefix}_metrics.txt"
        with open(metrics_file, "w") as f:
            f.write("Backtest Results:\n")
            f.write("=" * 50 + "\n")
            for metric, value in self.results["metrics"].items():
                f.write(f"{metric}: {value}\n")

        self.logger.info(f"Results exported to {filepath_prefix}_*")
