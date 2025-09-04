# Portfolio Optimizer SDK

This SDK provides financial data handling, portfolio optimization, and results visualization tools for Python projects.

## Installation

Copy the `portfolio_optimizer` folder into your project, or package it with `setup.py`/`pyproject.toml` for pip installation.

## Usage Example

```python
from portfolio_optimizer import DataHandler, PortfolioOptimizer, ResultsHandler

# Data
data_handler = DataHandler()
nifty50_df = data_handler.fetch_nifty50_composition()
prices = data_handler.fetch_historical_prices([...], '2023-01-01', '2023-12-31')
mu, cov = data_handler.calculate_statistics(prices)

# Optimization
optimizer = PortfolioOptimizer(mu, cov, w_benchmark, w_old)
optimizer.define_problem(objective_str, constraints_list)
status, w_opt = optimizer.solve()

# Results
results = ResultsHandler([...], w_opt, w_old, w_benchmark, mu, cov, prices)
metrics = results.calculate_metrics()
trade_signals = results.generate_trade_signals()
plot_data = results.get_plot_data()
```

## Contents
- `DataHandler`: Load benchmark, fetch prices, calculate statistics
- `PortfolioOptimizer`: Define and solve optimization problems
- `ResultsHandler`: Metrics, trade signals, and plot data

## License
MIT
