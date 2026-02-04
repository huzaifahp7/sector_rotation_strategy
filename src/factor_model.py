import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import warnings
from utils.logger import create_logger

warnings.filterwarnings("ignore")

logger = create_logger("logs/factor_model.log")


class FactorModel:
    def __init__(self, config):
        self.config = config
        self.factors = None
        self.factor_metadata = {}

    def create_ff5_factors(self, ff5_data):
        logger.info("Creating FF5 factors...")

        factors = ff5_data.copy()

        self.factor_metadata["ff5"] = {
            "MKT_RF": "Market excess return",
            "SMB": "Small minus Big (size factor)",
            "HML": "High minus Low (value factor)",
            "RMW": "Robust minus Weak (profitability factor)",
            "CMA": "Conservative minus Aggressive (investment factor)",
            "RF": "Risk-free rate",
        }

        return factors

    def create_momentum_factor(self, returns, lookback=252):
        logger.info("Creating momentum factor...")

        momentum = returns.rolling(lookback).mean()
        momentum_factor = momentum.mean(axis=1)

        self.factor_metadata["MOMENTUM"] = f"Momentum factor ({lookback} days)"

        return momentum_factor

    def create_volatility_factor(self, returns, lookback=63):
        logger.info("Creating volatility factor...")

        volatility = returns.rolling(lookback).std()
        volatility_factor = volatility.mean(axis=1)

        self.factor_metadata["VOLATILITY"] = f"Volatility factor ({lookback} days)"

        return volatility_factor

    def create_macro_factors(self, macro_data, dates):
        logger.info("Creating macro factors...")

        macro_factors = pd.DataFrame(index=dates)

        fedfunds = macro_data["fedfunds"].reindex(dates, method="ffill")
        macro_factors["FEDFUNDS"] = fedfunds["FEDFUNDS"]

        output_gap = macro_data["output_gap"].reindex(dates, method="ffill")
        macro_factors["OUTPUT_GAP"] = output_gap["GDPC1_GDPPOT"]

        nber = macro_data["nber"].reindex(dates, method="ffill")
        macro_factors["REGIME"] = nber["USREC"]

        self.factor_metadata.update(
            {
                "FEDFUNDS": "Federal Funds Rate",
                "OUTPUT_GAP": "Output gap (GDP vs potential)",
                "REGIME": "NBER recession indicator",
            }
        )

        return macro_factors

    def create_pca_factor(self, returns):
        logger.info("Creating PCA factor...")

        pca_factor = pd.Series(0.0, index=returns.index, dtype=float)

        window_size = 252

        for i in range(window_size, len(returns)):
            window_returns = returns.iloc[i - window_size : i]
            window_returns_clean = window_returns.dropna()

            if len(window_returns_clean) > 10:
                try:
                    pca = PCA(n_components=1)
                    pca_result = pca.fit_transform(window_returns_clean)
                    pca_factor.iloc[i] = pca_result[-1, 0]
                except:
                    pca_factor.iloc[i] = 0.0
            else:
                pca_factor.iloc[i] = 0.0

        np.random.seed(42)
        noise = np.random.normal(0, 0.001, len(pca_factor))
        pca_factor = pca_factor + noise

        self.factor_metadata["PCA"] = "Dynamic PCA factor (rolling 252-day window)"

        return pca_factor

    def create_custom_factors(self, returns, macro_data, dates):
        logger.info("Creating custom factors...")

        custom_factors = pd.DataFrame(index=dates)

        fedfunds = macro_data["fedfunds"].reindex(dates, method="ffill")["FEDFUNDS"]
        custom_factors["RATE_REGIME"] = (fedfunds > 5).astype(int) - (
            fedfunds < 2
        ).astype(int)

        output_gap = macro_data["output_gap"].reindex(dates, method="ffill")[
            "GDPC1_GDPPOT"
        ]
        custom_factors["GROWTH_REGIME"] = (output_gap > 0).astype(int)

        market_vol = returns.std(axis=1).rolling(252).mean()
        custom_factors["VOL_REGIME"] = (market_vol > market_vol.quantile(0.8)).astype(
            int
        )

        self.factor_metadata.update(
            {
                "RATE_REGIME": "Interest rate regime (high/low)",
                "GROWTH_REGIME": "Growth regime (expansion/contraction)",
                "VOL_REGIME": "Volatility regime (high/low)",
            }
        )

        return custom_factors

    def build_comprehensive_model(self, aligned_data):
        logger.info("Building comprehensive factor model...")

        if "etfs" in aligned_data:
            etf_dates = aligned_data["etfs"].index
        elif "industries" in aligned_data:
            etf_dates = aligned_data["industries"].index
        else:
            etf_dates = aligned_data["returns"].index

        factors = pd.DataFrame(index=etf_dates)

        if "ff5" in aligned_data and aligned_data["ff5"] is not None:
            ff5_data = aligned_data["ff5"]
            ff5_aligned = ff5_data.reindex(etf_dates, method="ffill")
            factors = pd.concat([factors, ff5_aligned], axis=1)
            logger.info(
                f"Loaded real FF5 factors with {len(ff5_aligned.columns)} factors"
            )
        else:
            ff5_factors = self.create_dynamic_ff5_factors(etf_dates)
            factors = pd.concat([factors, ff5_factors], axis=1)
            logger.warning("Using synthetic FF5 factors (real data not available)")

        if "industries" in aligned_data and aligned_data["industries"] is not None:
            industry_returns = aligned_data["industries"]

            industry_returns = industry_returns.dropna(axis=1, how="all")

            if len(industry_returns.columns) > 1:
                industry_returns = industry_returns.fillna(method="ffill").fillna(
                    method="bfill"
                )

                industry_returns_std = (
                    industry_returns - industry_returns.mean()
                ) / industry_returns.std()

                try:
                    from sklearn.decomposition import PCA
                    from sklearn.preprocessing import StandardScaler

                    scaler = StandardScaler()
                    industry_returns_scaled = scaler.fit_transform(
                        industry_returns_std.fillna(0)
                    )

                    pca = PCA(n_components=1)
                    pca_factor = pca.fit_transform(industry_returns_scaled)

                    pca_factor_df = pd.DataFrame(
                        pca_factor, index=industry_returns.index, columns=["PCA"]
                    )

                    target_vol = 0.15
                    current_vol = pca_factor_df["PCA"].std() * np.sqrt(252)
                    if current_vol > 0:
                        scaling_factor = target_vol / current_vol
                        pca_factor_df["PCA"] = pca_factor_df["PCA"] * scaling_factor

                    factors["PCA"] = pca_factor_df["PCA"]
                    logger.info(
                        f"Created PCA factor with {target_vol:.1%} annual volatility"
                    )

                except Exception as e:
                    logger.error(f"PCA factor creation failed: {e}")
                    factors["PCA"] = industry_returns.mean(axis=1).rolling(20).mean()
                    factors["PCA"] = factors["PCA"].fillna(0)
            else:
                logger.warning("Insufficient industry data for PCA factor")
                factors["PCA"] = pd.Series(0, index=etf_dates)
        else:
            logger.warning("No industry data available for PCA factor")
            factors["PCA"] = pd.Series(0, index=etf_dates)

        factors = factors.fillna(method="ffill").fillna(method="bfill")

        logger.info(f"Factor model created with {len(factors.columns)} factors")

        for factor in factors.columns:
            if factor != "RF":
                vol = factors[factor].std() * np.sqrt(252)
                logger.info(f"{factor}: {vol:.2%} annual volatility")

        return factors, aligned_data.get("industries", aligned_data.get("etfs"))

    def create_dynamic_ff5_factors(self, dates):
        np.random.seed(42)

        n_days = len(dates)

        market_trend = np.linspace(0, 0.001, n_days)
        market_cycle = 0.0005 * np.sin(2 * np.pi * np.arange(n_days) / 252)
        market_noise = np.random.normal(0, 0.01, n_days)
        market_factor = market_trend + market_cycle + market_noise

        smb_trend = np.linspace(0, 0.0002, n_days)
        smb_cycle = 0.0003 * np.sin(2 * np.pi * np.arange(n_days) / 126)
        smb_noise = np.random.normal(0, 0.008, n_days)
        smb_factor = smb_trend + smb_cycle + smb_noise

        hml_trend = np.linspace(0, 0.0003, n_days)
        hml_cycle = 0.0004 * np.sin(2 * np.pi * np.arange(n_days) / 504)
        hml_noise = np.random.normal(0, 0.009, n_days)
        hml_factor = hml_trend + hml_cycle + hml_noise

        rmw_trend = np.linspace(0, 0.0001, n_days)
        rmw_cycle = 0.0002 * np.sin(2 * np.pi * np.arange(n_days) / 252)
        rmw_noise = np.random.normal(0, 0.007, n_days)
        rmw_factor = rmw_trend + rmw_cycle + rmw_noise

        cma_trend = np.linspace(0, 0.0002, n_days)
        cma_cycle = 0.0003 * np.sin(2 * np.pi * np.arange(n_days) / 378)
        cma_noise = np.random.normal(0, 0.008, n_days)
        cma_factor = cma_trend + cma_cycle + cma_noise

        rf_factor = np.random.normal(0.0001, 0.001, n_days)

        ff5_factors = pd.DataFrame(
            {
                "MKT_RF": market_factor,
                "SMB": smb_factor,
                "HML": hml_factor,
                "RMW": rmw_factor,
                "CMA": cma_factor,
                "RF": rf_factor,
            },
            index=dates,
        )

        return ff5_factors

    def get_factor_summary(self):
        if self.factors is None:
            return None

        summary = {}

        for factor in self.factors.columns:
            if factor != "RF":
                factor_data = self.factors[factor]

                summary[factor] = {
                    "mean": factor_data.mean(),
                    "std": factor_data.std(),
                    "sharpe": (
                        factor_data.mean() / factor_data.std()
                        if factor_data.std() > 0
                        else 0
                    ),
                    "description": self.factor_metadata.get(factor, "No description"),
                }

        return summary

    def get_factor_correlations(self):
        if self.factors is None:
            return None

        return self.factors.corr()

    def export_factors(self, filepath):
        if self.factors is not None:
            self.factors.to_csv(filepath)
            logger.info(f"Factors exported to {filepath}")
        else:
            logger.warning("No factors to export")
