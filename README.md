# Overview
This is a sector rotation strategy implementation. The main entry point is the modular Python script `src/main.py`.

## Prerequisites
The reader should ensure that the data files are stored in the directory called `data`.

## Components
There are 8 main components:

### 1. Logger

**Log Files Created**:
- `logs/main.log` - Main pipeline execution
- `logs/data_processor.log` - Data loading and processing
- `logs/alpha_signals.log` - Signal generation
- `logs/backtester.log` - Backtesting operations
- `logs/risk_analyzer.log` - Risk analysis
- `logs/optimizer.log` - Optimization processes
- `logs/factor_model.log` - Factor model operations

### 2. Data Processor

**Key Functions**:
- `process_all_data()` - Loads and processes all data files
- `load_industry_portfolios()` - Loads 49 Industry Portfolios
- `load_etf_data()` - Loads sector ETF data
- `create_pca_factors()` - Creates PCA factors from industry returns
- `get_aligned_data()` - Aligns all data to common dates

**Data Sources**:
- 49 Industry Portfolios (for signal generation)
- Sector ETFs (for actual trading)
- Macroeconomic data (Fed Funds, Output Gap, NBER)
- FF5 factors
- Benchmark data (SPY, QQQ)

### 3. Optuna (optimizer)
- `optimize_signal_weights()` - Optimizes weights for 7 core signals
- `optimize_factor_model()` - Optimizes factor selection and weights
- `optimize_ml_model()` - Optimizes ML model hyperparameters
- `optimize_risk_parameters()` - Optimizes risk management parameters

**Optimization Features**:
- Sharpe ratio as objective function
- 50 trials by default
- Caching to prevent redundant calculations
- Early stopping for efficiency
- Study export for analysis

### 4. Signal Generation
**Purpose**: Contains 7 core alpha signals (hypotheses).

**Core Signals**:
1. **Momentum Signal** - Cross-sectional momentum ranking (252-day lookback)
2. **Mean Reversion Signal** - Mean reversion with volatility adjustment (63-day lookback)
3. **Volatility Signal** - Volatility-based ranking (21-day lookback)
4. **Quality Signal** - Quality factor proxy using returns (126-day lookback)
5. **Macro Regime Signal** - Macroeconomic regime detection
6. **Regime Adaptive Signal** - Dynamic signal adjustment based on market conditions
7. **Defensive Signal** - Contrarian positioning during stress periods

### 5. Backtesting Framework
**Key Functions**:
- `run_backtest()` - Main backtesting function
- `calculate_weights()` - Converts signals to portfolio weights
- `apply_risk_management()` - Applies risk management rules
- `calculate_performance_metrics()` - Calculates comprehensive metrics

**Risk**:
- Volatility targeting (18% target)
- Position size limits (10% maximum)
- Drawdown protection (20% limit)
- Transaction costs (5 bps)
- Leverage constraints

**Performance Metrics**:
- Sharpe ratio, Sortino ratio, Calmar ratio
- Total return, max drawdown, volatility
- Turnover analysis, capacity utilization
- Factor alpha and t-statistics

### 6. Factor Modeling

**Key Functions**:
- `build_comprehensive_model()` - Combines FF5 and PCA factors
- `create_ff5_factors()` - Loads or creates FF5 factors
- `create_pca_factor()` - Creates PCA factors from industry returns
- `create_dynamic_ff5_factors()` - Creates synthetic factors as fallback

**Factor Types**:
- **FF5 Factors**: Market (MKT_RF), Size (SMB), Value (HML), Profitability (RMW), Investment (CMA), Risk-free (RF)
- **PCA Factors**: 3 dynamic factors from industry returns
- **Macro Factors**: Fed Funds Rate, Output Gap, NBER recession indicator


### 7. Risk analysis

**Key Functions**:
- `calculate_factor_alpha()` - True alpha calculation using multi-factor regression
- `factor_attribution()` - Factor exposure and contribution analysis
- `stress_testing()` - Historical crisis period analysis
- `classify_market_regimes()` - Market condition classification
- `create_benchmark_comparison_plot()` - Strategy vs benchmark comparison
- `create_macro_sensitivity_plot()` - Economic indicator sensitivity

**Stress Test Periods**:
- COVID Crash (2020-02-01 to 2020-04-30)
- COVID Recovery (2020-05-01 to 2021-12-31)
- Inflation Shock (2022-01-01 to 2022-12-31)
- 2008 Financial Crisis (2008-09-01 to 2009-03-31)

**Market Regimes**:
- Bull Market: High return, low volatility
- Bear Market: Negative return or high volatility
- Sideways Market: Everything else

### 8. Main
The orchestrator and calls all other functions and runs the strategy. It has a config defined for bps, whether to include SPY/QQQ in the portfolio or not.

**Configuration**:
```python
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
```

**Pipeline Flow**:
1. **Initialization** - Create directories and initialize components
2. **Data Processing** - Load and align all data
3. **Signal Generation** - Generate and optimize alpha signals
4. **Signal Mapping** - Map industry signals to ETFs
5. **Backtesting** - Run comprehensive backtest
6. **Factor Analysis** - Calculate factor alpha and attribution
7. **Risk Analysis** - Stress tests and regime analysis
8. **Reporting** - Export results and generate comprehensive report

**Output Structure**:
```
results/
├── plots/
│   ├── backtest_results.png
│   ├── benchmark_comparison.png
│   ├── macro_sensitivity.png
│   ├── market_regimes.png
│   └── stress_tests.png
├── signals/
│   └── etf_signals.csv
├── optimization/
│   └── signal_weights.json
└── reports/
    ├── comprehensive_report.txt
    ├── backtest_results.txt
    └── risk_analysis.txt
```

## Scripts in src

| Script | Purpose | Key Functions |
|--------|---------|---------------|
| `main.py` | Pipeline orchestration | `main()`, `generate_comprehensive_report()` |
| `data_processor.py` | Data loading and processing | `process_all_data()`, `get_aligned_data()` |
| `alpha_signals.py` | Signal generation | `generate_optimized_signals()`, `map_industry_signals_to_etfs()` |
| `backtester.py` | Backtesting engine | `run_backtest()`, `calculate_weights()` |
| `factor_model.py` | Factor model creation | `build_comprehensive_model()`, `create_pca_factor()` |
| `risk_analyzer.py` | Risk analysis | `calculate_factor_alpha()`, `stress_testing()` |
| `optimizer.py` | Hyperparameter optimization | `optimize_signal_weights()`, `optimize_factor_model()` |
| `utils/logger.py` | Logging system | `create_logger()`, `Logger` class |

