import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from pathlib import Path
import statsmodels.api as sm
from utils.logger import create_logger

warnings.filterwarnings("ignore")

logger = create_logger("logs/risk_analyzer.log")


class RiskAnalyzer:
    def __init__(self, config):
        self.config = config
        self.analysis_results = {}

    def create_benchmark_comparison_plot(self, strategy_returns, output_path):
        logger.info("Creating benchmark comparison plot...")

        benchmark_file_path = Path("../new_data/benchmarks.csv")
        if not benchmark_file_path.exists():
            logger.warning("Benchmark file not found")
            return

        benchmark_df = pd.read_csv(benchmark_file_path, index_col=0, parse_dates=True)

        spy_data = benchmark_df[benchmark_df["ticker"] == "SPY"].copy()
        qqq_data = benchmark_df[benchmark_df["ticker"] == "QQQ"].copy()

        if len(spy_data) == 0 or len(qqq_data) == 0:
            logger.warning("SPY or QQQ data not found")
            return

        spy_returns = spy_data["close"].pct_change().dropna()
        qqq_returns = qqq_data["close"].pct_change().dropna()

        common_dates = strategy_returns.index.intersection(
            spy_returns.index
        ).intersection(qqq_returns.index)
        if len(common_dates) == 0:
            logger.warning("No common dates for benchmark comparison")
            return

        strategy_aligned = strategy_returns.loc[common_dates]
        spy_aligned = spy_returns.loc[common_dates]
        qqq_aligned = qqq_returns.loc[common_dates]

        strategy_cumulative = (1 + strategy_aligned).cumprod()
        spy_cumulative = (1 + spy_aligned).cumprod()
        qqq_cumulative = (1 + qqq_aligned).cumprod()

        plt.figure(figsize=(12, 8))
        plt.plot(
            strategy_cumulative.index,
            strategy_cumulative.values,
            label="Strategy",
            linewidth=2,
            color="blue",
        )
        plt.plot(
            spy_cumulative.index,
            spy_cumulative.values,
            label="SPY",
            linewidth=2,
            color="red",
            alpha=0.8,
        )
        plt.plot(
            qqq_cumulative.index,
            qqq_cumulative.values,
            label="QQQ",
            linewidth=2,
            color="green",
            alpha=0.8,
        )

        plt.title("Strategy vs Benchmarks", fontsize=14, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Benchmark comparison plot saved to {output_path}")

    def create_macro_sensitivity_plot(self, strategy_returns, output_path):
        logger.info("Creating macro sensitivity plot...")

        try:
            cpi_df = pd.read_csv(
                "../new_data/cpi_fed.csv", index_col=0, parse_dates=True
            )
            usd_df = pd.read_csv(
                "../new_data/usd_values.csv", index_col=0, parse_dates=True
            )
            fedfunds_df = pd.read_csv(
                "../new_data/fedfunds.csv", index_col=0, parse_dates=True
            )

            cpi_data = cpi_df.iloc[:, 0]
            usd_data = usd_df.iloc[:, 0]
            fedfunds_data = fedfunds_df.iloc[:, 0]

            cpi_data = cpi_data.resample("D").ffill()
            fedfunds_data = fedfunds_data.resample("D").ffill()

        except Exception as e:
            logger.error(f"Error loading macro data: {e}")
            return

        common_dates = (
            strategy_returns.index.intersection(cpi_data.index)
            .intersection(usd_data.index)
            .intersection(fedfunds_data.index)
        )
        if len(common_dates) == 0:
            logger.warning("No common dates for macro comparison")
            return

        strategy_aligned = strategy_returns.loc[common_dates]
        cpi_aligned = cpi_data.loc[common_dates]
        usd_aligned = usd_data.loc[common_dates]
        fedfunds_aligned = fedfunds_data.loc[common_dates]

        strategy_normalized = (1 + strategy_aligned).cumprod()
        cpi_normalized = cpi_aligned / cpi_aligned.iloc[0]
        usd_normalized = usd_aligned / usd_aligned.iloc[0]
        fedfunds_normalized = fedfunds_aligned / fedfunds_aligned.iloc[0]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

        ax1.plot(
            strategy_normalized.index,
            strategy_normalized.values,
            label="Strategy",
            linewidth=2,
            color="blue",
        )
        ax1.plot(
            cpi_normalized.index,
            cpi_normalized.values,
            label="CPI",
            linewidth=2,
            color="red",
            alpha=0.8,
        )
        ax1.set_title("Strategy vs CPI (Normalized)", fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel("Normalized Value")

        ax2.plot(
            strategy_normalized.index,
            strategy_normalized.values,
            label="Strategy",
            linewidth=2,
            color="blue",
        )
        ax2.plot(
            usd_normalized.index,
            usd_normalized.values,
            label="USD Index",
            linewidth=2,
            color="green",
            alpha=0.8,
        )
        ax2.set_title("Strategy vs USD Index (Normalized)", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylabel("Normalized Value")

        ax3.plot(
            strategy_normalized.index,
            strategy_normalized.values,
            label="Strategy",
            linewidth=2,
            color="blue",
        )
        ax3.plot(
            fedfunds_normalized.index,
            fedfunds_normalized.values,
            label="FEDFUNDS",
            linewidth=2,
            color="orange",
            alpha=0.8,
        )
        ax3.set_title("Strategy vs FEDFUNDS (Normalized)", fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylabel("Normalized Value")
        ax3.set_xlabel("Date")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Macro sensitivity plot saved to {output_path}")

    def factor_attribution(self, returns, factors, weights):
        logger.info("Performing factor attribution (static OLS)...")

        common_dates = returns.index.intersection(factors.index)
        if len(common_dates) == 0:
            logger.warning("No common dates for factor attribution")
            return {}

        portfolio_returns = returns.loc[common_dates]
        factor_data = factors.loc[common_dates]

        factor_data = (
            factor_data.dropna(axis=1, how="all").fillna(method="ffill").fillna(0)
        )
        portfolio_returns = portfolio_returns.fillna(0)

        import statsmodels.api as sm

        X = sm.add_constant(factor_data)
        y = portfolio_returns
        model = sm.OLS(y, X).fit()
        exposures = model.params.drop("const")

        marginal_contributions = {}
        factor_contributions = {}
        factor_information_ratios = {}
        for factor in exposures.index:
            exposure = exposures[factor]
            mean_factor = factor_data[factor].mean()
            contrib = exposure * factor_data[factor]
            marginal_contributions[factor] = exposure * mean_factor
            factor_contributions[factor] = contrib
            if contrib.std() > 0:
                factor_information_ratios[factor] = contrib.mean() / contrib.std()
            else:
                factor_information_ratios[factor] = 0.0

        self.analysis_results["factor_attribution"] = {
            "exposures": exposures,
            "marginal_contributions": marginal_contributions,
            "factor_contributions": factor_contributions,
            "information_ratios": factor_information_ratios,
            "model_summary": model.summary().as_text(),
        }

        logger.info(
            f"Factor attribution completed for {len(exposures)} factors (static OLS)"
        )
        logger.info(f"Exposures: {exposures.to_dict()}")
        logger.info(f"Marginal contributions: {marginal_contributions}")
        logger.info(f"Information ratios: {factor_information_ratios}")

        return self.analysis_results["factor_attribution"]

    def calculate_factor_alpha(self, strategy_returns, factor_df):
        logger.info("Calculating true factor alpha (multi-factor regression)...")

        logger.info(f"Strategy returns shape: {strategy_returns.shape}")
        logger.info(f"Factor data shape: {factor_df.shape}")
        logger.info(f"Factor columns: {list(factor_df.columns)}")
        logger.info(
            f"Strategy returns range: {strategy_returns.index.min()} to {strategy_returns.index.max()}"
        )
        logger.info(
            f"Factor data range: {factor_df.index.min()} to {factor_df.index.max()}"
        )

        common_dates = strategy_returns.index.intersection(factor_df.index)
        logger.info(f"Common dates: {len(common_dates)}")

        if len(common_dates) < 100:
            logger.warning("Very few common dates for factor regression")
            return 0.0, 0.0, 0.0, 0.0, None

        y = strategy_returns.loc[common_dates]
        X = factor_df.loc[common_dates]

        X = X.dropna(axis=1, how="all")

        X = X.fillna(method="ffill").fillna(0)
        y = y.fillna(0)

        logger.info(f"Final X shape: {X.shape}")
        logger.info(f"Final y shape: {y.shape}")
        logger.info(f"X NaN count: {X.isnull().sum().sum()}")
        logger.info(f"y NaN count: {y.isnull().sum()}")

        if X.empty or y.empty:
            logger.error("No valid data for factor regression")
            return 0.0, 0.0, 0.0, 0.0, None

        X = sm.add_constant(X)

        try:
            model = sm.OLS(y, X).fit()
            alpha_daily = model.params["const"]
            tstat_daily = model.tvalues["const"]
            alpha_annual = alpha_daily * 252
            tstat_annual = tstat_daily

            logger.info(f"Daily alpha: {alpha_daily:.6f} (t-stat: {tstat_daily:.2f})")
            logger.info(
                f"Annualized alpha: {alpha_annual:.2%} (t-stat: {tstat_annual:.2f})"
            )
            logger.info(f"R-squared: {model.rsquared:.3f}")

            self.analysis_results["factor_alpha"] = {
                "alpha_daily": alpha_daily,
                "alpha_annual": alpha_annual,
                "tstat_daily": tstat_daily,
                "tstat_annual": tstat_annual,
                "model_summary": model.summary().as_text(),
            }
            return alpha_daily, alpha_annual, tstat_daily, tstat_annual, model

        except Exception as e:
            logger.error(f"Error in factor regression: {e}")
            return 0.0, 0.0, 0.0, 0.0, None

    def stress_testing(self, returns, stress_periods=None):
        logger.info("Running stress tests...")

        if stress_periods is None:
            stress_periods = {
                "COVID_Crash": ("2020-02-01", "2020-04-30"),
                "COVID_Recovery": ("2020-05-01", "2021-12-31"),
                "Inflation_Shock": ("2022-01-01", "2022-12-31"),
                "2008_Financial_Crisis": ("2008-09-01", "2009-03-31"),
            }

        stress_results = {}

        for period_name, (start_date, end_date) in stress_periods.items():
            try:
                period_returns = returns[start_date:end_date]
                if len(period_returns) > 10:
                    total_return = (1 + period_returns).prod() - 1
                    max_loss = period_returns.min()
                    volatility = period_returns.std() * np.sqrt(252)
                    sharpe = (
                        period_returns.mean() / period_returns.std() * np.sqrt(252)
                        if period_returns.std() > 0
                        else 0
                    )

                    stress_results[period_name] = {
                        "total_return": total_return,
                        "max_loss": max_loss,
                        "volatility": volatility,
                        "sharpe": sharpe,
                        "period": f"{start_date} to {end_date}",
                        "days": len(period_returns),
                    }

                    logger.info(f"{period_name}:")
                    logger.info(f"Total Return: {total_return:.2%}")
                    logger.info(f"Max Loss: {max_loss:.2%}")
                    logger.info(f"Volatility: {volatility:.2%}")
                    logger.info(f"Sharpe: {sharpe:.2f}")
                    logger.info(f"Days: {len(period_returns)}")
                else:
                    logger.warning(
                        f"Insufficient data for {period_name}: {len(period_returns)} days"
                    )
                    stress_results[period_name] = {
                        "total_return": 0.0,
                        "max_loss": 0.0,
                        "volatility": 0.0,
                        "sharpe": 0.0,
                        "period": "N/A",
                        "days": 0,
                    }
            except Exception as e:
                logger.error(f"Error in stress test {period_name}: {e}")
                stress_results[period_name] = {
                    "total_return": 0.0,
                    "max_loss": 0.0,
                    "volatility": 0.0,
                    "sharpe": 0.0,
                    "period": "N/A",
                    "days": 0,
                }

        self.analysis_results["stress_tests"] = stress_results
        return stress_results

    def plot_stress_tests(self, stress_results, output_path):
        if not stress_results:
            logger.warning("No stress test results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Stress Test Results", fontsize=16, fontweight="bold")

        periods = list(stress_results.keys())
        returns = [stress_results[p]["total_return"] for p in periods]
        volatilities = [stress_results[p]["volatility"] for p in periods]
        sharpes = [stress_results[p]["sharpe"] for p in periods]
        max_losses = [stress_results[p]["max_loss"] for p in periods]

        axes[0, 0].bar(
            periods, returns, color=["green" if r > 0 else "red" for r in returns]
        )
        axes[0, 0].set_title("Total Returns by Stress Period")
        axes[0, 0].set_ylabel("Total Return")
        axes[0, 0].tick_params(axis="x", rotation=45)

        axes[0, 1].bar(periods, volatilities, color="skyblue")
        axes[0, 1].set_title("Volatility by Stress Period")
        axes[0, 1].set_ylabel("Annual Volatility")
        axes[0, 1].tick_params(axis="x", rotation=45)

        axes[1, 0].bar(
            periods, sharpes, color=["green" if s > 0 else "red" for s in sharpes]
        )
        axes[1, 0].set_title("Sharpe Ratios by Stress Period")
        axes[1, 0].set_ylabel("Sharpe Ratio")
        axes[1, 0].tick_params(axis="x", rotation=45)

        axes[1, 1].bar(periods, max_losses, color="red")
        axes[1, 1].set_title("Maximum Losses by Stress Period")
        axes[1, 1].set_ylabel("Maximum Loss")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Stress test plot saved to {output_path}")

    def export_analysis(self, filepath_prefix):
        if "factor_attribution" in self.analysis_results:
            fa = self.analysis_results["factor_attribution"]
            with open(f"{filepath_prefix}_factor_attribution.txt", "w") as f:
                f.write("Factor Attribution Analysis:\n")
                f.write("=" * 50 + "\n\n")
                f.write("Factor Exposures:\n")
                for k, v in fa["exposures"].items():
                    f.write(f"{k}: {v:.4f}\n")
                f.write("\nFactor Information Ratios:\n")
                for k, v in fa["information_ratios"].items():
                    f.write(f"{k}: {v:.3f}\n")

        if "stress_tests" in self.analysis_results:
            stress_file = f"{filepath_prefix}_stress_tests.txt"
            with open(stress_file, "w") as f:
                f.write("Stress Test Results:\n")
                f.write("=" * 50 + "\n")

                for event, metrics in self.analysis_results["stress_tests"].items():
                    f.write(f"\n{event}:\n")
                    for metric, value in metrics.items():
                        f.write(f"  {metric}: {value}\n")

        logger.info(f"Analysis exported to {filepath_prefix}_*")

    def macro_sensitivity_stats(self, strategy_returns):
        import pandas as pd

        try:
            cpi_df = pd.read_csv(
                "../new_data/cpi_fed.csv", index_col=0, parse_dates=True
            )
            usd_df = pd.read_csv(
                "../new_data/usd_values.csv", index_col=0, parse_dates=True
            )
            ffd_df = pd.read_csv(
                "../new_data/fedfunds.csv", index_col=0, parse_dates=True
            )

            cpi = cpi_df.iloc[:, 0]
            usd = usd_df.iloc[:, 0]
            ffd = ffd_df.iloc[:, 0]

            cpi = cpi.resample("D").ffill()
            ffd = ffd.resample("D").ffill()

        except Exception as e:
            return {"error": f"Could not load macro data: {e}"}

        common = (
            strategy_returns.index.intersection(cpi.index)
            .intersection(usd.index)
            .intersection(ffd.index)
        )
        if len(common) == 0:
            return {"error": "No common dates for macro sensitivity"}

        strat = strategy_returns.loc[common]
        cpi = cpi.loc[common]
        usd = usd.loc[common]
        ffd = ffd.loc[common]

        results = {
            "correlation_to_cpi": strat.corr(cpi),
            "correlation_to_usd": strat.corr(usd),
            "correlation_to_ffd": strat.corr(ffd),
        }
        return results

    def classify_market_regimes(self, strategy_returns):
        logger.info("Classifying market regimes using SPY...")

        benchmark_file_path = Path("../new_data/benchmarks.csv")
        if not benchmark_file_path.exists():
            logger.warning("Benchmark file not found for regime classification")
            return None

        benchmark_df = pd.read_csv(benchmark_file_path, index_col=0, parse_dates=True)
        spy_data = benchmark_df[benchmark_df["ticker"] == "SPY"].copy()

        if len(spy_data) == 0:
            logger.warning("SPY data not found for regime classification")
            return None

        spy_returns = spy_data["close"].pct_change().dropna()

        common_dates = strategy_returns.index.intersection(spy_returns.index)
        if len(common_dates) == 0:
            logger.warning("No common dates for regime classification")
            return None

        strategy_aligned = strategy_returns.loc[common_dates]
        spy_aligned = spy_returns.loc[common_dates]

        rolling_return = spy_aligned.rolling(252, min_periods=126).mean() * 252
        rolling_vol = spy_aligned.rolling(252, min_periods=126).std() * np.sqrt(252)
        rolling_sharpe = rolling_return / (rolling_vol + 1e-8)

        regimes = pd.Series(index=common_dates, dtype=str)

        bull_condition = (rolling_return > 0.10) & (rolling_vol < 0.20)
        regimes[bull_condition] = "Bull"

        bear_condition = (rolling_return < -0.05) | (rolling_vol > 0.30)
        regimes[bear_condition] = "Bear"

        regimes[regimes.isna()] = "Sideways"

        regime_performance = {}
        for regime in ["Bull", "Bear", "Sideways"]:
            regime_mask = regimes == regime
            if regime_mask.sum() > 0:
                regime_returns = strategy_aligned[regime_mask]
                regime_spy = spy_aligned[regime_mask]

                total_return = (1 + regime_returns).prod() - 1
                volatility = regime_returns.std() * np.sqrt(252)
                sharpe = (
                    regime_returns.mean() / (regime_returns.std() + 1e-8) * np.sqrt(252)
                )
                max_drawdown = self.calculate_drawdown(regime_returns).min()

                excess_returns = regime_returns - regime_spy
                alpha = excess_returns.mean() * 252
                alpha_tstat = alpha / (
                    excess_returns.std() / np.sqrt(len(excess_returns)) + 1e-8
                )

                regime_performance[regime] = {
                    "total_return": total_return,
                    "volatility": volatility,
                    "sharpe": sharpe,
                    "max_drawdown": max_drawdown,
                    "alpha": alpha,
                    "alpha_tstat": alpha_tstat,
                    "days": len(regime_returns),
                    "percentage": len(regime_returns) / len(strategy_aligned) * 100,
                }

        self.analysis_results["market_regimes"] = {
            "regimes": regimes,
            "performance": regime_performance,
            "spy_returns": spy_aligned,
            "strategy_returns": strategy_aligned,
        }

        logger.info("Market regime classification completed")
        for regime, perf in regime_performance.items():
            logger.info(f"{regime} Market:")
            logger.info(f"Total Return: {perf['total_return']:.2%}")
            logger.info(f"Volatility: {perf['volatility']:.2%}")
            logger.info(f"Sharpe: {perf['sharpe']:.2f}")
            logger.info(f"Max Drawdown: {perf['max_drawdown']:.2%}")
            logger.info(
                f"Alpha: {perf['alpha']:.2%} (t-stat: {perf['alpha_tstat']:.2f})"
            )
            logger.info(f"Days: {perf['days']} ({perf['percentage']:.1f}%)")

        return self.analysis_results["market_regimes"]

    def calculate_drawdown(self, returns):
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown

    def plot_market_regimes(self, output_path):
        if "market_regimes" not in self.analysis_results:
            logger.warning("No market regime data available for plotting")
            return

        regimes_data = self.analysis_results["market_regimes"]
        regimes = regimes_data["regimes"]
        spy_returns = regimes_data["spy_returns"]
        strategy_returns = regimes_data["strategy_returns"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Market Regime Analysis", fontsize=16, fontweight="bold")

        colors = {"Bull": "green", "Bear": "red", "Sideways": "orange"}

        cumulative_spy = (1 + spy_returns).cumprod()
        for regime in ["Bull", "Bear", "Sideways"]:
            regime_mask = regimes == regime
            if regime_mask.sum() > 0:
                axes[0, 0].plot(
                    cumulative_spy[regime_mask].index,
                    cumulative_spy[regime_mask].values,
                    color=colors[regime],
                    label=regime,
                    linewidth=2,
                )
        axes[0, 0].set_title("SPY Returns by Market Regime")
        axes[0, 0].set_ylabel("Cumulative Return")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        cumulative_strategy = (1 + strategy_returns).cumprod()
        for regime in ["Bull", "Bear", "Sideways"]:
            regime_mask = regimes == regime
            if regime_mask.sum() > 0:
                axes[0, 1].plot(
                    cumulative_strategy[regime_mask].index,
                    cumulative_strategy[regime_mask].values,
                    color=colors[regime],
                    label=regime,
                    linewidth=2,
                )
        axes[0, 1].set_title("Strategy Returns by Market Regime")
        axes[0, 1].set_ylabel("Cumulative Return")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        regime_counts = regimes.value_counts()
        axes[1, 0].pie(
            regime_counts.values,
            labels=regime_counts.index,
            colors=[colors[r] for r in regime_counts.index],
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[1, 0].set_title("Market Regime Distribution")

        performance = regimes_data["performance"]
        regimes_list = list(performance.keys())
        returns_list = [performance[r]["total_return"] for r in regimes_list]
        sharpe_list = [performance[r]["sharpe"] for r in regimes_list]

        x = np.arange(len(regimes_list))
        width = 0.35

        ax2 = axes[1, 1]
        bars1 = ax2.bar(
            x - width / 2, returns_list, width, label="Total Return", color="skyblue"
        )
        ax2.set_ylabel("Total Return")
        ax2.set_title("Performance by Market Regime")
        ax2.set_xticks(x)
        ax2.set_xticklabels(regimes_list)
        ax2.legend()

        ax3 = ax2.twinx()
        bars2 = ax3.bar(
            x + width / 2, sharpe_list, width, label="Sharpe Ratio", color="lightcoral"
        )
        ax3.set_ylabel("Sharpe Ratio")
        ax3.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Market regime plot saved to {output_path}")
