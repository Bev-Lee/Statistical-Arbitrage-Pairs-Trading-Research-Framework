# Statistical Arbitrage & Pairs Trading Research Framework

A reproducible Python research project asking a deliberately difficult question:

> Can temporary deviations between historically related equities generate market-neutral opportunities **after** transaction costs and out-of-sample validation?

This is an educational research framework, not investment advice or a live-trading system. It is designed to make every modelling choice inspectable: the asset universe is defined before screening, hedge ratios are estimated using information available at the time, signals execute on the next bar, and costs are applied on turnover.

## Research design

1. Download and clean daily adjusted-close data for a pre-defined liquid-equity universe.
2. Screen all pairs using Engle–Granger cointegration on an in-sample period.
3. Estimate an OLS hedge ratio, construct a log-price spread, and calculate a rolling z-score.
4. Trade only the selected pair in the next, unseen period: enter at `|z| >= 2.0`, exit at `|z| <= 0.5`.
5. Apply next-bar execution, transaction costs, slippage, volatility scaling, and position limits.
6. Evaluate return, volatility, Sharpe ratio, drawdown, turnover, and trade statistics; then stress-test costs and thresholds.

The default demo is deterministic synthetic data so the repository runs offline and never claims a fabricated market result. Use `--source yfinance` to run the same pipeline on data you download yourself.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_research --source synthetic
pytest -q
```

Outputs are written to `reports/figures/` and `reports/results/`. The runner creates a correlation heatmap, spread/z-score chart, equity/drawdown chart, cost sensitivity chart, and a metrics table.

## Repository layout

```
src/          reusable data, statistics, signal, backtest, metric, and plotting modules
tests/        unit tests including a no-look-ahead execution test
config.yaml   explicit research assumptions and universe
reports/      generated results and a concise research report
data/         ignored raw/processed market data (only .gitkeep files are tracked)
```

## Key safeguards and limitations

- **No same-bar P&L:** a signal observed at close `t` affects returns only from `t+1`.
- **Costs:** all position changes incur configurable commissions and slippage.
- **Validation:** pairs are selected and parameterised in training, then tested on unseen data.
- **Research, not proof:** cointegration can break, liquidity and borrow costs are simplified, and synthetic data are only a reproducible smoke test.

## CV-ready project description

**Statistical Arbitrage & Pairs Trading Research Framework** | Python, Statistics, Financial Modelling

- Developed a market-neutral research framework using Engle–Granger cointegration, OLS hedge ratios, and rolling z-score signals to study mean-reverting equity relationships.
- Built a transaction-cost- and slippage-aware backtester with next-period execution, volatility-scaled exposure, and portfolio limits to mitigate look-ahead bias.
- Implemented out-of-sample validation and sensitivity analysis across signal thresholds and trading costs; documented metrics, assumptions, and failure modes in a reproducible report.

Do not add performance figures to a CV until they come from a completed, documented run on a declared data sample.
