# Research report: Statistical Arbitrage & Pairs Trading

## Question

Can deviations between historically related liquid equities support a market-neutral mean-reversion strategy after execution costs and out-of-sample evaluation?

## Method

The project pre-defines a small liquid-equity universe, splits prices into a training and an unseen segment, screens the training data with the Engle–Granger cointegration test, estimates an OLS log-price hedge ratio, and converts the resulting spread to a rolling z-score. Positions open at an absolute z-score of 2.0 and close at 0.5. A trade decision at close is lagged before it contributes to P&L.

## Evaluation

The runner records annualised return and volatility, Sharpe ratio, maximum drawdown, turnover, exposure days, and position changes. It applies configurable commission and slippage costs to every position change. Its offline default uses synthetic data solely to demonstrate reproducibility; it is not evidence of a market edge.

## What did not work / limitations

Correlation alone is not sufficient evidence of a stable tradable relationship. Cointegration may fail out of sample, and this baseline does not model borrow availability, corporate-action data quality, intraday execution, dynamic spreads, or capacity. These omissions are intentional limitations to discuss before interpreting results.

## Next experiments

1. Run rolling walk-forward pair selection rather than one fixed split.
2. Compare static OLS, rolling OLS, and a Kalman-filter hedge ratio.
3. Run a threshold/cost grid only on validation data, then lock assumptions for the final test.
4. Add a realistic benchmark and a market-regime analysis.
