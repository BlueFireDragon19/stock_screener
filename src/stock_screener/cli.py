from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from stock_screener.config import ScreenerConfig
from stock_screener.screener import run_screener


def _load_dotenv(path: Path | None = None) -> None:
    env_path = path or Path.cwd() / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Long-only screener: support, catalyst, fundamentals, politicians; "
            "VIX regime adjusts gates"
        ),
    )
    p.add_argument("--tickers", type=str, default="", help="Comma-separated tickers")
    p.add_argument(
        "--universe",
        choices=("us", "sp500"),
        default="us",
        help="Free universe source",
    )
    p.add_argument(
        "--mode",
        choices=("support", "catalyst", "fundamentals", "politicians", "both", "all"),
        default="all",
        help="Screen mode (default: all)",
    )
    p.add_argument("--limit", type=int, default=None, help="Cap universe size")
    p.add_argument("--top", type=int, default=25, help="Rows to print per mode")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("output/screen.csv"),
        help="CSV base path (suffixes per mode)",
    )
    p.add_argument("--no-news", action="store_true")
    p.add_argument("--no-reddit", action="store_true")
    p.add_argument("--no-earnings", action="store_true")
    p.add_argument("--no-fundamentals", action="store_true")
    p.add_argument("--no-politicians", action="store_true")
    p.add_argument("--no-capitoltrades", action="store_true")
    p.add_argument("--no-house", action="store_true")
    p.add_argument("--no-senate", action="store_true")
    p.add_argument("--no-grok", action="store_true")
    p.add_argument("--grok-top", type=int, default=15)
    p.add_argument("--politician-pages", type=int, default=5)
    p.add_argument("--house-filings", type=int, default=25)
    p.add_argument(
        "--allow-below-200",
        action="store_true",
        help="Support baseline: allow below SMA200 (risk-off still forces above)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _print_table(df: pd.DataFrame, cols: list[str], top: int) -> None:
    show = df.head(top)
    use = [c for c in cols if c in show.columns]
    try:
        from tabulate import tabulate

        print(tabulate(show[use], headers="keys", tablefmt="simple", showindex=False))
    except Exception:  # noqa: BLE001
        print(show[use].to_string(index=False))


def _save(df: pd.DataFrame, base: Path, mode_name: str, single: bool) -> Path:
    path = base if single else base.with_name(f"{base.stem}_{mode_name}.csv")
    df.to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    logging.getLogger("yfinance").setLevel(logging.INFO)
    logging.getLogger("peewee").setLevel(logging.INFO)

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] or None
    config = ScreenerConfig(
        universe=args.universe,
        mode=args.mode,
        max_tickers=args.limit,
        top_n=args.top,
        require_above_sma_slow=not args.allow_below_200,
        enable_reddit=not args.no_reddit,
        enable_earnings=not args.no_earnings,
        enable_fundamentals=not args.no_fundamentals,
        enable_politicians=not args.no_politicians,
        politicians_enable_capitol=not args.no_capitoltrades,
        politicians_enable_house=not args.no_house,
        politicians_enable_senate=not args.no_senate,
        politicians_capitol_pages=args.politician_pages,
        politicians_house_filings=args.house_filings,
        enable_grok=not args.no_grok,
        grok_top_n=args.grok_top,
    )

    support_df, catalyst_df, fund_df, pol_df, market = run_screener(
        config=config,
        tickers=tickers,
        fetch_news=not args.no_news,
    )

    print(
        f"\nMarket: {market.label} | VIX={market.vix if market.vix is not None else 'n/a'}"
    )
    print(f"Regime: {market.reason}")
    try:
        from stock_screener.data.market_cycle import assess_market_cycle

        cycle = assess_market_cycle()
        cape_s = f"{cycle.cape:.1f}" if cycle.cape is not None else "n/a"
        buff_s = f"{cycle.buffett_pct:.0f}%" if cycle.buffett_pct is not None else "n/a"
        print(
            f"Market cycle: {cycle.label} | CAPE={cape_s} | Buffett={buff_s} ({cycle.note})"
        )
    except Exception:  # noqa: BLE001
        pass

    args.output.parent.mkdir(parents=True, exist_ok=True)
    any_rows = False
    single = args.mode in {
        "support",
        "catalyst",
        "fundamentals",
        "politicians",
    }

    if args.mode in {"support", "both", "all"}:
        print("\n=== SUPPORT (near support / mean-reversion) ===")
        if support_df.empty:
            print("No names passed support filters.")
        else:
            any_rows = True
            path = _save(
                support_df, args.output, "support", single and args.mode == "support"
            )
            _print_table(
                support_df,
                [
                    "ticker",
                    "price",
                    "rsi14",
                    "rsi_bias",
                    "proximity",
                    "pct_from_52w_high",
                    "pct_from_52w_low",
                    "pct_from_1w_high",
                    "sma_regime",
                    "news_score",
                    "fund_score",
                    "peg",
                    "fcf_yield",
                    "roe",
                    "composite",
                    "why",
                ],
                args.top,
            )
            print(f"Saved {len(support_df)} rows → {path}")

    if args.mode in {"catalyst", "both", "all"}:
        print("\n=== CATALYST (news / earnings / momentum pop) ===")
        if catalyst_df.empty:
            print("No names passed catalyst filters.")
        else:
            any_rows = True
            path = _save(
                catalyst_df,
                args.output,
                "catalyst",
                single and args.mode == "catalyst",
            )
            _print_table(
                catalyst_df,
                [
                    "ticker",
                    "price",
                    "rsi14",
                    "rsi_bias",
                    "day_return_pct",
                    "volume_ratio",
                    "pct_from_52w_high",
                    "pct_from_1w_high",
                    "news_score",
                    "momentum_score",
                    "fund_score",
                    "peg",
                    "fcf_yield",
                    "roe",
                    "composite",
                    "why",
                ],
                args.top,
            )
            print(f"Saved {len(catalyst_df)} rows → {path}")

    if args.mode in {"fundamentals", "all"} or (
        not args.no_fundamentals and args.mode in {"both", "support", "catalyst"}
    ):
        print("\n=== FUNDAMENTALS (growth / profit / BS / ROIC / valuation) ===")
        if fund_df.empty:
            print("No names passed fundamentals filters.")
        else:
            any_rows = True
            path = _save(
                fund_df,
                args.output,
                "fundamentals",
                single and args.mode == "fundamentals",
            )
            _print_table(
                fund_df,
                [
                    "ticker",
                    "sector",
                    "trailing_pe",
                    "peg",
                    "fcf_yield",
                    "roe",
                    "gross_margin",
                    "profit_margin",
                    "growth_score",
                    "profitability_score",
                    "balance_score",
                    "capital_score",
                    "valuation_score",
                    "composite",
                    "why",
                ],
                args.top,
            )
            print(f"Saved {len(fund_df)} rows → {path}")

    if args.mode in {"politicians", "all"} or (
        not args.no_politicians and args.mode in {"both", "support", "catalyst"}
    ):
        print(
            "\n=== POLITICIANS (Capitol Trades + House/Senate · freshness-weighted) ==="
        )
        if pol_df.empty:
            print("No politician-trade tickers scored.")
        else:
            any_rows = True
            path = _save(
                pol_df,
                args.output,
                "politicians",
                single and args.mode == "politicians",
            )
            _print_table(
                pol_df,
                [
                    "ticker",
                    "score",
                    "buy_count",
                    "sell_count",
                    "avg_filed_after",
                    "sources",
                    "politicians",
                    "why",
                ],
                args.top,
            )
            print(f"Saved {len(pol_df)} rows → {path}")

    return 0 if any_rows else 1


if __name__ == "__main__":
    sys.exit(main())
