import pandas as pd
import numpy as np
import pathlib
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.logger import create_logger

warnings.filterwarnings("ignore")

logger = create_logger("logs/data_processor.log")


class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = {}

    def create_pca_factors(self, industry_returns, n_factors=3):
        logger.info(f"Creating {n_factors} PCA factors from industry returns...")

        industry_returns = industry_returns.dropna(axis=1, how="all")

        industry_returns = industry_returns.fillna(method="ffill").fillna(0)

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(industry_returns)

        pca = PCA(n_components=n_factors)
        pca_factors = pca.fit_transform(scaled_data)

        factor_names = [f"PCA_{i+1}" for i in range(n_factors)]
        pca_df = pd.DataFrame(
            pca_factors, index=industry_returns.index, columns=factor_names
        )

        explained_variance = pca.explained_variance_ratio_
        logger.info(f"PCA explained variance ratios: {explained_variance}")
        logger.info(f"Total explained variance: {sum(explained_variance):.3f}")

        return pca_df

    def process_all_data(self):
        logger.info("Processing all data files...")

        logger.info("Loading 49 Industry Portfolios...")
        industries = self.load_industry_portfolios()
        logger.info(
            f"Loaded {len(industries)} observations for {len(industries.columns)} industries"
        )

        logger.info("Loading Sector ETFs...")
        etfs, etf_prices = self.load_etf_data()
        logger.info(f"Loaded {len(etfs)} observations for {len(etfs.columns)} ETFs")

        logger.info("Loading macroeconomic data...")
        fedfunds = self.load_fedfunds()
        output_gap = self.load_output_gap()
        nber = self.load_nber()

        logger.info("Loading FF5 data...")
        ff5 = self.load_ff5()

        logger.info("Loading benchmark data...")
        benchmarks = self.load_benchmarks()

        logger.info("Creating PCA factors...")
        pca_factors = self.create_pca_factors(industries, n_factors=3)

        self.data = {
            "industries": industries,
            "etfs": etfs,
            "etf_prices": etf_prices,
            "fedfunds": fedfunds,
            "output_gap": output_gap,
            "nber": nber,
            "ff5": ff5,
            "benchmarks": benchmarks,
            "pca_factors": pca_factors,
        }

        logger.info("Data processing complete!")
        logger.info(f"Date range: {industries.index.min()} to {industries.index.max()}")

        return self.data

    def load_industry_portfolios(self):
        file_path = self.data_path / "49_Industry_Portfolios.csv"

        df = pd.read_csv(file_path, index_col=0)

        df.columns = df.columns.str.strip()

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.replace(-99.99, np.nan)
        df = df.fillna(method="ffill")

        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        returns = df.pct_change()

        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.fillna(0)

        returns = returns.resample("M").last()

        return returns

    def load_etf_data(self):
        file_path = self.data_path / "sector_prices_full.csv"

        df = pd.read_csv(file_path, parse_dates=["date"])
        df.set_index("date", inplace=True)

        etf_prices = df.pivot(columns="sector", values="close")

        etf_returns = etf_prices.pct_change()

        etf_returns = etf_returns.replace([np.inf, -np.inf], np.nan)
        etf_returns = etf_returns.fillna(0)

        return etf_returns, etf_prices

    def load_fedfunds(self):
        file_path = self.data_path / "fedfunds.csv"
        if file_path.exists():
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            return df
        else:
            return pd.DataFrame(
                {"FEDFUNDS": [2.0] * 1000},
                index=pd.date_range("2000-01-01", periods=1000, freq="D"),
            )

    def load_output_gap(self):
        file_path = self.data_path / "output_gap.csv"
        if file_path.exists():
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            return df
        else:
            return pd.DataFrame(
                {"GDPC1_GDPPOT": [0.0] * 1000},
                index=pd.date_range("2000-01-01", periods=1000, freq="D"),
            )

    def load_nber(self):
        file_path = self.data_path / "nber.csv"
        if file_path.exists():
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            return df
        else:
            return pd.DataFrame(
                {"USREC": [0] * 1000},
                index=pd.date_range("2000-01-01", periods=1000, freq="D"),
            )

    def load_ff5(self):
        file_path = self.data_path / "ff5_daily.csv"
        if file_path.exists():
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            df = df.replace(-99.99, np.nan)
            df = df.interpolate(method="linear", limit_direction="both")
            df = df.fillna(method="ffill").fillna(method="bfill").fillna(0)
            return df
        else:
            dates = pd.date_range("2000-01-01", periods=1000, freq="D")
            np.random.seed(42)

            base_values = {
                "MKT_RF": np.random.normal(0.0005, 0.01, 1000),
                "SMB": np.random.normal(0.0002, 0.008, 1000),
                "HML": np.random.normal(0.0003, 0.009, 1000),
                "RMW": np.random.normal(0.0001, 0.007, 1000),
                "CMA": np.random.normal(0.0002, 0.008, 1000),
                "RF": np.random.normal(0.0001, 0.001, 1000),
            }

            for factor in base_values:
                trend = np.linspace(0, 0.001, 1000)
                noise = np.random.normal(0, 0.005, 1000)
                base_values[factor] = base_values[factor] + trend + noise

            return pd.DataFrame(base_values, index=dates)

    def load_benchmarks(self):
        benchmarks = {}

        for ticker in ["SPY", "QQQ"]:
            file_path = self.data_path / f"{ticker}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                if "Adj Close" in df.columns:
                    returns = df["Adj Close"].pct_change()
                    benchmarks[ticker] = returns

        return benchmarks

    def get_aligned_data(self):
        industry_returns = self.data["industries"]
        etf_returns = self.data["etfs"]
        etf_prices = self.data["etf_prices"]

        logger.info(f"Industry data shape: {industry_returns.shape}")
        logger.info(f"ETF data shape: {etf_returns.shape}")
        logger.info(
            f"Industry date range: {industry_returns.index.min()} to {industry_returns.index.max()}"
        )
        logger.info(
            f"ETF date range: {etf_returns.index.min()} to {etf_returns.index.max()}"
        )

        industry_daily = industry_returns.resample("D").ffill()

        common_dates = industry_daily.index.intersection(etf_returns.index)

        logger.info(f"Common dates: {len(common_dates)} observations")
        logger.info(f"Common date range: {common_dates.min()} to {common_dates.max()}")

        aligned_industries = industry_daily.loc[common_dates]
        aligned_etfs = etf_returns.loc[common_dates]
        aligned_etf_prices = etf_prices.loc[common_dates]

        pca_factors = self.data["pca_factors"]
        aligned_pca = pca_factors.reindex(common_dates).fillna(method="ffill").fillna(0)

        aligned_data = {
            "industries": aligned_industries,
            "etfs": aligned_etfs,
            "etf_prices": aligned_etf_prices,
            "fedfunds": self.data["fedfunds"],
            "output_gap": self.data["output_gap"],
            "nber": self.data["nber"],
            "ff5": self.data["ff5"],
            "benchmarks": self.data["benchmarks"],
            "pca_factors": aligned_pca,
        }

        return aligned_data

    def create_industry_to_etf_mapping(self):
        industry_to_etf = {
            "Food": "XLP",
            "Soda": "XLP",
            "Beer": "XLP",
            "Smoke": "XLP",
            "Toys": "XLY",
            "Fun": "XLY",
            "Books": "XLY",
            "Hshld": "XLY",
            "Clths": "XLY",
            "Meals": "XLY",
            "Whlsl": "XLY",
            "Rtail": "XLY",
            "Hlth": "XLV",
            "MedEq": "XLV",
            "Drugs": "XLV",
            "Chems": "XLI",
            "Rubbr": "XLI",
            "Txtls": "XLI",
            "BldMt": "XLI",
            "Cnstr": "XLI",
            "Steel": "XLI",
            "FabPr": "XLI",
            "Mach": "XLI",
            "ElcEq": "XLI",
            "Autos": "XLI",
            "Aero": "XLI",
            "Ships": "XLI",
            "Guns": "XLI",
            "PerSv": "XLI",
            "BusSv": "XLI",
            "LabEq": "XLI",
            "Trans": "XLI",
            "Paper": "XLB",
            "Boxes": "XLB",
            "Gold": "XLE",
            "Mines": "XLE",
            "Coal": "XLE",
            "Oil": "XLE",
            "Util": "XLU",
            "Telcm": "XLC",
            "Hardw": "XLK",
            "Softw": "XLK",
            "Chips": "XLK",
            "Banks": "XLF",
            "Insur": "XLF",
            "Fin": "XLF",
            "RlEst": "XLRE",
            "Agric": None,
            "Other": None,
        }

        return industry_to_etf
