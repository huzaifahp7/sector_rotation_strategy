import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import GradientBoostingRegressor
import warnings
import matplotlib.pyplot as plt
from utils.logger import create_logger

warnings.filterwarnings("ignore")

logger = create_logger("logs/optimizer.log")


class Optimizer:
    def __init__(self, config):
        self.config = config
        self.optimization_results = {}

    def optimize_signal_weights(self, signals, returns, factors, n_trials=50):
        logger.info("Optimizing signal weights with Optuna...")

        def objective(trial):
            weights = {}
            for signal_name in signals.keys():
                weights[signal_name] = trial.suggest_float(
                    f"weight_{signal_name}", 0, 1
                )

            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}

            combined_signal = self.combine_signals(signals, weights)
            backtest_result = self.run_quick_backtest(combined_signal, returns)

            return -backtest_result["sharpe"]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Best trial: {study.best_trial.value}")
        logger.info(f"Best parameters: {study.best_trial.params}")

        self.optimization_results["signal_weights"] = {
            "best_params": study.best_trial.params,
            "best_value": study.best_trial.value,
            "study": study,
        }

        return study.best_trial.params

    def optimize_factor_model(self, factors, returns, n_trials=30):
        logger.info("Optimizing factor model parameters...")

        def objective(trial):
            selected_factors = []
            factor_weights = {}

            for factor in factors.columns:
                if factor != "RF":
                    if trial.suggest_categorical(f"use_{factor}", [True, False]):
                        selected_factors.append(factor)
                        factor_weights[factor] = trial.suggest_float(
                            f"weight_{factor}", 0, 1
                        )

            if not selected_factors:
                return 0

            total_weight = sum(factor_weights.values())
            if total_weight > 0:
                factor_weights = {
                    k: v / total_weight for k, v in factor_weights.items()
                }

            factor_portfolio = pd.Series(0, index=factors.index)
            for factor, weight in factor_weights.items():
                factor_portfolio += weight * factors[factor]

            performance = self.calculate_performance_metrics(factor_portfolio)

            return -performance["sharpe"]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Best factor model trial: {study.best_trial.value}")
        logger.info(f"Best parameters: {study.best_trial.params}")

        self.optimization_results["factor_model"] = {
            "best_params": study.best_trial.params,
            "best_value": study.best_trial.value,
            "study": study,
        }

        return study.best_trial.params

    def optimize_ml_model(self, factors, returns, n_trials=20):
        logger.info("Optimizing ML model hyperparameters...")

        X = factors.dropna()
        y = returns.loc[X.index]

        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
            max_depth = trial.suggest_int("max_depth", 2, 8)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
            )

            n_splits = 3
            split_size = len(X) // n_splits

            scores = []
            for i in range(n_splits):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < n_splits - 1 else len(X)

                X_train = X.iloc[:start_idx]
                y_train = y.iloc[:start_idx]
                X_val = X.iloc[start_idx:end_idx]
                y_val = y.iloc[start_idx:end_idx]

                if len(X_train) > 0 and len(X_val) > 0:
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_val)

                    ss_res = np.sum((y_val - predictions) ** 2)
                    ss_tot = np.sum((y_val - y_val.mean()) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    scores.append(r2)

            return -np.mean(scores) if scores else 0

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Best ML model trial: {study.best_trial.value}")
        logger.info(f"Best parameters: {study.best_trial.params}")

        self.optimization_results["ml_model"] = {
            "best_params": study.best_trial.params,
            "best_value": study.best_trial.value,
            "study": study,
        }

        return study.best_trial.params

    def optimize_risk_parameters(self, alpha_signal, returns, n_trials=20):
        logger.info("Optimizing risk management parameters...")

        def objective(trial):
            target_vol = trial.suggest_float("target_vol", 0.10, 0.25)
            vol_lookback = trial.suggest_int("vol_lookback", 126, 504)
            max_leverage = trial.suggest_float("max_leverage", 1.5, 3.0)
            min_leverage = trial.suggest_float("min_leverage", 0.3, 0.8)

            weights = alpha_signal.rank(axis=1, pct=True) - 0.5

            portfolio_returns = (weights * returns).sum(axis=1)
            rolling_vol = portfolio_returns.rolling(vol_lookback).std() * np.sqrt(252)

            vol_scalar = target_vol / rolling_vol
            vol_scalar = vol_scalar.clip(min_leverage, max_leverage)

            weights_scaled = weights.multiply(vol_scalar, axis=0)

            strategy_returns = (weights_scaled.shift(1) * returns).sum(axis=1)
            performance = self.calculate_performance_metrics(strategy_returns)

            return -performance["sharpe"]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Best risk parameters trial: {study.best_trial.value}")
        logger.info(f"Best parameters: {study.best_trial.params}")

        self.optimization_results["risk_parameters"] = {
            "best_params": study.best_trial.params,
            "best_value": study.best_trial.value,
            "study": study,
        }

        return study.best_trial.params

    def combine_signals(self, signals, weights=None):
        if isinstance(signals, pd.DataFrame):
            return signals

        if signals is None or (hasattr(signals, "empty") and signals.empty):
            logger.warning("No signals to combine")
            return None

        if not isinstance(signals, dict):
            logger.warning(f"Unexpected signals type: {type(signals)}")
            return None

        if not signals:
            logger.warning("No signals to combine")
            return None

        if weights is None:
            weights = {
                "valuation_spread": 0.25,
                "momentum": 0.25,
                "volatility": 0.20,
                "macro_regime": 0.15,
                "mean_reversion": 0.10,
                "quality": 0.05,
            }

        ref_signal = list(signals.values())[0]
        combined = pd.DataFrame(0, index=ref_signal.index, columns=ref_signal.columns)

        for signal_name, signal_data in signals.items():
            weight = weights.get(signal_name, 0)

            signal_normalized = signal_data.fillna(0)

            for col in signal_normalized.columns:
                q_low = signal_normalized[col].quantile(0.01)
                q_high = signal_normalized[col].quantile(0.99)
                signal_normalized[col] = signal_normalized[col].clip(q_low, q_high)

            combined += weight * signal_normalized

        return combined

    def run_quick_backtest(self, alpha_signal, returns):
        weights = alpha_signal.rank(axis=1, pct=True) - 0.5
        strategy_returns = (weights.shift(1) * returns).sum(axis=1)
        metrics = self.calculate_performance_metrics(strategy_returns)

        return metrics

    def calculate_performance_metrics(self, returns):
        if len(returns) == 0:
            return {"sharpe": 0, "total_return": 0, "volatility": 0}

        total_return = (1 + returns).prod() - 1
        annual_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0

        return {
            "sharpe": sharpe,
            "total_return": total_return,
            "volatility": volatility,
            "annual_return": annual_return,
        }

    def get_optimization_summary(self):
        summary = {}

        for optimization_type, results in self.optimization_results.items():
            summary[optimization_type] = {
                "best_value": results["best_value"],
                "best_params": results["best_params"],
            }

        return summary

    def plot_optimization_history(self, save_path=None):
        if not self.optimization_results:
            logger.warning("No optimization results to plot")
            return

        n_optimizations = len(self.optimization_results)
        fig, axes = plt.subplots(1, n_optimizations, figsize=(5 * n_optimizations, 4))

        if n_optimizations == 1:
            axes = [axes]

        for i, (optimization_type, results) in enumerate(
            self.optimization_results.items()
        ):
            study = results["study"]

            axes[i].plot(study.trials_dataframe()["value"])
            axes[i].set_title(f'{optimization_type.replace("_", " ").title()}')
            axes[i].set_xlabel("Trial")
            axes[i].set_ylabel("Objective Value")
            axes[i].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Optimization history plot saved to {save_path}")

        plt.show()

    def export_optimization_results(self, filepath_prefix):
        for optimization_type, results in self.optimization_results.items():
            params_file = f"{filepath_prefix}_{optimization_type}_params.txt"
            with open(params_file, "w") as f:
                f.write(
                    f"{optimization_type.replace('_', ' ').title()} Optimization Results:\n"
                )
                f.write("=" * 50 + "\n")
                f.write(f"Best value: {results['best_value']:.6f}\n")
                f.write(f"Best parameters:\n")
                for param, value in results["best_params"].items():
                    f.write(f"  {param}: {value}\n")

            study_file = f"{filepath_prefix}_{optimization_type}_study.pkl"
            import pickle

            with open(study_file, "wb") as f:
                pickle.dump(results["study"], f)

        logger.info(f"Optimization results exported to {filepath_prefix}_*")
