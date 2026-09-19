from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from .backtest import run_backtest
from .data import download_prices, make_synthetic_prices, save_prices
from .metrics import performance_metrics
from .signals import positions_from_zscore
from .statistics import estimate_hedge_ratio, rolling_zscore, screen_cointegrated_pairs, spread


def plot_outputs(prices: pd.DataFrame, test: pd.DataFrame, pair: tuple[str, str], figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5)); sns.heatmap(prices.pct_change().corr(), cmap="vlag", center=0, annot=True, fmt=".2f")
    plt.title("Daily-return correlation"); plt.tight_layout(); plt.savefig(figures / "correlation_heatmap.png", dpi=160); plt.close()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(test.index, test["spread"], label="Log-price spread"); axes[0].legend(); axes[0].set_title(f"{pair[0]}/{pair[1]} out-of-sample spread")
    z = rolling_zscore(test["spread"], 20); axes[1].plot(z.index, z, label="z-score"); axes[1].axhline(2, color="r", ls="--"); axes[1].axhline(-2, color="r", ls="--"); axes[1].axhline(0.5, color="grey", ls=":"); axes[1].axhline(-0.5, color="grey", ls=":"); axes[1].legend()
    plt.tight_layout(); plt.savefig(figures / "spread_and_zscore.png", dpi=160); plt.close()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(test.index, test["equity"], label="Net equity"); axes[0].legend(); axes[0].set_title("Out-of-sample equity curve")
    axes[1].fill_between(test.index, test["drawdown"], 0, color="firebrick", alpha=.55); axes[1].set_title("Drawdown")
    plt.tight_layout(); plt.savefig(figures / "equity_and_drawdown.png", dpi=160); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["synthetic", "yfinance"], default="synthetic")
    parser.add_argument("--start", default="2018-01-01"); parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()
    config = yaml.safe_load(Path("config.yaml").read_text()); cfg = config["research"]
    prices = make_synthetic_prices() if args.source == "synthetic" else download_prices(config["universe"], args.start, args.end)
    save_prices(prices, "data/processed/prices.csv")
    split = int(len(prices) * cfg["train_fraction"]); train, unseen = prices.iloc[:split], prices.iloc[split:]
    candidates = screen_cointegrated_pairs(train)
    best = candidates.iloc[0]; left, right = best.left, best.right
    beta = estimate_hedge_ratio(train[left], train[right])
    unseen_spread = spread(unseen[left], unseen[right], beta)
    z = rolling_zscore(unseen_spread, cfg["z_window"])
    signal = positions_from_zscore(z, cfg["entry_z"], cfg["exit_z"])
    test = run_backtest(unseen_spread, signal, cfg["commission_bps"], cfg["slippage_bps"], cfg["target_daily_vol"], cfg["max_gross_exposure"])
    metrics = performance_metrics(test, cfg["annualization_factor"])
    results = Path("reports/results"); results.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(results / "pair_screen.csv", index=False); pd.Series(metrics).to_csv(results / "metrics.csv", header=["value"])
    pd.DataFrame([{"source": args.source, "pair": f"{left}/{right}", "hedge_ratio": beta, "pair_pvalue": best.pvalue, **metrics}]).to_csv(results / "summary.csv", index=False)
    plot_outputs(prices, test, (left, right), Path("reports/figures"))
    print(f"Selected {left}/{right}; OOS Sharpe={metrics['sharpe']:.2f}; outputs in reports/")


if __name__ == "__main__":
    main()
