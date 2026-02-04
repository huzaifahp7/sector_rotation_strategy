import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from alpha_signals import AlphaSignals
from backtester import Backtester
from data_processor import DataProcessor
from factor_model import FactorModel
from risk_analyzer import RiskAnalyzer
from utils.logger import create_logger

warnings.filterwarnings("ignore")

DATA = Path("../new_data")
OUT = Path("../results")

CONFIG = {
    "data_path": DATA,
    "output_path": OUT,
    "target_volatility": 0.18,
    "max_position_size": 0.10,
    "transaction_cost_bps": 5,
    "risk_free_rate": 0.02,
    "drawdown_limit": 0.20,
    "include_spy": True,
    "include_qqq": True,
    "lookback_periods": {
        "momentum": 252,
        "volatility": 63,
        "macro": 126,
        "sentiment": 21,
    },
}

PLOTS_DIR = OUT / "plots"
SIGNALS_DIR = OUT / "signals"
OPTIMIZATION_DIR = OUT / "optimization"
REPORTS_DIR = OUT / "reports"

for dir_path in [PLOTS_DIR, SIGNALS_DIR, OPTIMIZATION_DIR, REPORTS_DIR]:
    dir_path.mkdir(exist_ok=True)


def main():
    logger = create_logger("logs/main.log")

    logger.info("=" * 60)
    logger.info("ALPHA SIGNAL PIPELINE")
    logger.info("=" * 60)

    logger.info("1. Initializing components...")
    data_processor = DataProcessor(CONFIG["data_path"])
    alpha_signals = AlphaSignals(CONFIG)
    backtester = Backtester(CONFIG)
    factor_model = FactorModel(CONFIG)
    risk_analyzer = RiskAnalyzer(CONFIG)

    logger.info("2. Loading and processing data...")
    data_processor.process_all_data()
    aligned_data = data_processor.get_aligned_data()

    if aligned_data is None:
        logger.error("Failed to load data")
        return

    logger.info("Data loaded successfully")
    logger.info(f"Industries: {aligned_data['industries'].shape}")
    logger.info(f"ETFs: {aligned_data['etfs'].shape}")
    logger.info(
        f"Factors: {aligned_data['ff5'].shape if aligned_data.get('ff5') is not None else 'None'}"
    )

    if (
        "SPY" not in aligned_data["etfs"].columns
        or "QQQ" not in aligned_data["etfs"].columns
    ):
        logger.info("Adding SPY and QQQ to ETF returns...")
        benchmark_file_path = Path("../new_data/benchmarks.csv")
        if benchmark_file_path.exists():
            benchmark_df = pd.read_csv(
                benchmark_file_path, index_col=0, parse_dates=True
            )

            if CONFIG.get("include_spy", True):
                spy_data = benchmark_df[benchmark_df["ticker"] == "SPY"].copy()
                if len(spy_data) > 0:
                    spy_returns = spy_data["close"].pct_change().dropna()
                    common_dates = aligned_data["etfs"].index.intersection(
                        spy_returns.index
                    )
                    if len(common_dates) > 0:
                        aligned_data["etfs"] = aligned_data["etfs"].reindex(
                            common_dates
                        )
                        aligned_data["etfs"]["SPY"] = spy_returns.loc[common_dates]
                        logger.info(f"Added SPY with {len(common_dates)} common dates")
                else:
                    logger.warning("SPY data not found")
            else:
                logger.warning("SPY disabled in configuration")

            if CONFIG.get("include_qqq", True):
                qqq_data = benchmark_df[benchmark_df["ticker"] == "QQQ"].copy()
                if len(qqq_data) > 0:
                    qqq_returns = qqq_data["close"].pct_change().dropna()
                    common_dates = aligned_data["etfs"].index.intersection(
                        qqq_returns.index
                    )
                    if len(common_dates) > 0:
                        aligned_data["etfs"] = aligned_data["etfs"].reindex(
                            common_dates
                        )
                        aligned_data["etfs"]["QQQ"] = qqq_returns.loc[common_dates]
                        logger.info(f"Added QQQ with {len(common_dates)} common dates")
                else:
                    logger.warning("QQQ data not found")
            else:
                logger.warning("QQQ disabled in configuration")

    logger.info("3. Generating alpha signals...")
    industry_signals = alpha_signals.generate_optimized_signals(
        aligned_data, optimize_weights=True, optimize_lookbacks=True
    )

    if industry_signals is None or industry_signals.empty:
        logger.error("Failed to generate alpha signals")
        return

    logger.info(f"Generated industry signals: {industry_signals.shape}")

    logger.info("4. Mapping signals to ETFs...")
    etf_signals = alpha_signals.map_industry_signals_to_etfs(
        industry_signals, aligned_data["etfs"]
    )

    if etf_signals is None or etf_signals.empty:
        logger.error("Failed to map signals to ETFs")
        return

    logger.info(f"Mapped signals to ETFs: {etf_signals.shape}")

    logger.info("5. Running backtest...")
    backtest_results = backtester.run_backtest(etf_signals, aligned_data["etfs"])

    if backtest_results is None:
        logger.error("Backtest failed")
        return

    logger.info("Backtest completed")
    logger.info(f"Sharpe Ratio: {backtest_results['metrics']['sharpe']:.3f}")
    logger.info(f"Total Return: {backtest_results['metrics']['total_return']:.2%}")
    logger.info(f"Max Drawdown: {backtest_results['metrics']['max_drawdown']:.2%}")
    logger.info(f"Alpha: {backtest_results.get('alpha_info', {}).get('alpha', 0):.2%}")
    logger.info(
        f"T-statistic: {backtest_results.get('alpha_info', {}).get('t_statistic', 0):.3f}"
    )

    logger.info("6. Calculating true factor alpha (multi-factor regression)...")

    factor_data = None
    if aligned_data.get("ff5") is not None and not aligned_data["ff5"].empty:
        factor_data = aligned_data["ff5"].copy()
        logger.info(f"Using FF5 factors: {list(factor_data.columns)}")

    if (
        aligned_data.get("pca_factors") is not None
        and not aligned_data["pca_factors"].empty
    ):
        if factor_data is not None:
            common_dates = factor_data.index.intersection(
                aligned_data["pca_factors"].index
            )
            factor_data = factor_data.loc[common_dates]
            pca_factors = aligned_data["pca_factors"].loc[common_dates]
            factor_data = pd.concat([factor_data, pca_factors], axis=1)
            logger.info(f"Added PCA factors: {list(pca_factors.columns)}")
        else:
            factor_data = aligned_data["pca_factors"].copy()
            logger.info(f"Using PCA factors only: {list(factor_data.columns)}")

    if factor_data is not None and not factor_data.empty:
        alpha_daily, alpha_annual, tstat_daily, tstat_annual, factor_model = (
            risk_analyzer.calculate_factor_alpha(
                backtest_results["returns"], factor_data
            )
        )
        logger.info(
            f"Factor alpha (daily): {alpha_daily:.6f} (t-stat: {tstat_daily:.2f})"
        )
        logger.info(
            f"Factor alpha (annualized): {alpha_annual:.2%} (t-stat: {tstat_annual:.2f})"
        )
    else:
        logger.warning("No factors available for true alpha calculation")

    logger.info("7. Generating plots...")
    plot_file = PLOTS_DIR / "backtest_results.png"
    backtester.plot_results(str(plot_file))
    logger.info(f"Plot saved to {plot_file}")

    logger.info("8. Creating benchmark comparison...")
    benchmark_plot = PLOTS_DIR / "benchmark_comparison.png"
    risk_analyzer.create_benchmark_comparison_plot(
        backtest_results["returns"], str(benchmark_plot)
    )

    logger.info("9. Creating macro sensitivity analysis...")
    macro_plot = PLOTS_DIR / "macro_sensitivity.png"
    risk_analyzer.create_macro_sensitivity_plot(
        backtest_results["returns"], str(macro_plot)
    )

    macro_stats = risk_analyzer.macro_sensitivity_stats(backtest_results["returns"])
    logger.info(f"Macro Sensitivity (correlation): {macro_stats}")
    report_file = REPORTS_DIR / "comprehensive_report.txt"
    with open(report_file, "a") as f:
        f.write("\nMACRO SENSITIVITY (correlation)\n")
        for k, v in macro_stats.items():
            f.write(f"{k}: {v}\n")

    logger.info("10. Running market regime analysis...")
    market_regimes = risk_analyzer.classify_market_regimes(backtest_results["returns"])

    regime_plot = PLOTS_DIR / "market_regimes.png"
    risk_analyzer.plot_market_regimes(str(regime_plot))

    logger.info("11. Running stress tests...")
    stress_results = risk_analyzer.stress_testing(backtest_results["returns"])

    stress_plot = PLOTS_DIR / "stress_tests.png"
    risk_analyzer.plot_stress_tests(stress_results, str(stress_plot))

    logger.info("12. Running factor attribution...")

    if factor_data is not None and not factor_data.empty:
        factor_attribution = risk_analyzer.factor_attribution(
            backtest_results["returns"],
            factor_data,
            backtest_results["weights"],
        )
    else:
        logger.warning("No factors available for attribution analysis")

    logger.info("13. Exporting results...")

    backtest_prefix = str(REPORTS_DIR / "backtest")
    backtester.export_results(backtest_prefix)

    risk_prefix = str(REPORTS_DIR / "risk_analysis")
    risk_analyzer.export_analysis(risk_prefix)

    signals_file = SIGNALS_DIR / "etf_signals.csv"
    etf_signals.to_csv(signals_file)
    logger.info(f"Signals exported to {signals_file}")

    if hasattr(alpha_signals, "optimized_weights"):
        opt_file = OPTIMIZATION_DIR / "signal_weights.json"
        import json

        with open(opt_file, "w") as f:
            json.dump(alpha_signals.optimized_weights, f, indent=2)
        logger.info(f"Optimization results exported to {opt_file}")

    logger.info("Results exported to organized directories")

    logger.info("14. Generating comprehensive report...")
    report_file = REPORTS_DIR / "comprehensive_report.txt"
    generate_comprehensive_report(backtest_results, report_file, risk_analyzer)
    logger.info(f"Report saved to {report_file}")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


def generate_comprehensive_report(backtest_results, report_file, risk_analyzer=None):
    with open(report_file, "w") as f:
        f.write("ALPHA SIGNAL PIPELINE - COMPREHENSIVE REPORT\n")
        f.write("=" * 60 + "\n\n")

        metrics = backtest_results["metrics"]
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Sharpe Ratio: {metrics.get('sharpe', 0):.3f}\n")
        f.write(f"Total Return: {metrics.get('total_return', 0):.2%}\n")
        f.write(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n")
        f.write(f"Annual Volatility: {metrics.get('volatility', 0):.2%}\n")

        if risk_analyzer and "factor_alpha" in getattr(
            risk_analyzer, "analysis_results", {}
        ):
            fa = risk_analyzer.analysis_results["factor_alpha"]
            f.write("\nFACTOR ALPHA (MULTI-FACTOR REGRESSION)\n")
            f.write("-" * 35 + "\n")
            f.write(
                f"Daily Alpha: {fa['alpha_daily']:.6f} (t-stat: {fa['tstat_daily']:.2f})\n"
            )
            f.write(
                f"Annualized Alpha: {fa['alpha_annual']:.2%} (t-stat: {fa['tstat_annual']:.2f})\n"
            )
            f.write("\nRegression Summary:\n")
            f.write(fa["model_summary"])

        f.write("\nALPHA SIGNAL INFORMATION\n")
        f.write("-" * 25 + "\n")
        f.write("Core Alpha Signals:\n")
        f.write("  - Momentum (short vs long term)\n")
        f.write("  - Mean Reversion\n")
        f.write("  - Volatility\n")
        f.write("  - Quality\n")
        f.write("  - Macro Regime\n")
        f.write("  - Regime Adaptive\n")

        f.write("\nRISK MANAGEMENT\n")
        f.write("-" * 18 + "\n")
        f.write(f"Target Volatility: {CONFIG['target_volatility']:.1%}\n")
        f.write(f"Max Position Size: {CONFIG['max_position_size']:.1%}\n")
        f.write(f"Drawdown Limit: {CONFIG['drawdown_limit']:.1%}\n")
        f.write(f"Transaction Costs: {CONFIG['transaction_cost_bps']} bps\n")

        f.write("\nCONFIGURATION\n")
        f.write("-" * 13 + "\n")
        for key, value in CONFIG.items():
            if key != "data_path" and key != "output_path":
                f.write(f"{key}: {value}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("END OF REPORT\n")


if __name__ == "__main__":
    main()
