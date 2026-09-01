# Stock screener (long-only)

Python screener with four long-only modes + **VIX regime**:

1. **Support** — near support / SMA pullback (stricter in risk-off)  
2. **Catalyst** — news / earnings / volume pops (needs vol+green day in risk-off)  
3. **Fundamentals** — Feroldi-style growth, profitability, balance sheet, ROIC, valuation  
4. **Politicians** — Congress trades from **Capitol Trades scrape** + **official House/Senate** disclosures, weighted so **shorter `filed_after` days score higher**

Shared free inputs: Yahoo OHLCV + news + earnings + fundamentals, `^VIX`, STOCK Act public filings, optional Reddit + Grok X.

## Free data sources

| Source | Use |
|--------|-----|
| NASDAQ Trader symbol dirs | Full US listed universe (`--universe us`) |
| Wikipedia | S&P 500 (`--universe sp500`) |
| Yahoo Finance / yfinance | OHLCV, VIX, news, earnings |
| Reddit public JSON / optional OAuth | Retail mention / polarity |
| xAI Grok API | Optional X sentiment on top names |

Reddit often blocks anonymous `.json` calls (403). Create a free Reddit “script” app and set `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET` (see `.env.example`) for reliable access. Without that, the screener continues with neutral Reddit scores.

Robinhood / SoFi / Public are not used (no free scan APIs). Webull OpenAPI is paid/approval-based and not wired in.

## Setup

```bash
cd ~/Projects/stock-screener
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
```

Optional Grok X enrichment:

```bash
export XAI_API_KEY=...
# optional: export XAI_MODEL=grok-4-1-fast-non-reasoning
```

## Run

```bash
# Both modes on a custom basket
stock-screener --mode both --tickers AAPL,MSFT,NVDA,META,AMD,GOOGL,AVGO --top 10 --no-reddit --no-grok

# Catalyst pops only
stock-screener --mode catalyst --universe sp500 --limit 80 --top 20

# Support dips only
stock-screener --mode support --tickers AAPL,MSFT,NVDA,JPM --allow-below-200

# Politicians (Capitol Trades + House PTRs; Senate best-effort)
stock-screener --mode politicians --politician-pages 3 --house-filings 20 --top 25
```

Politician freshness weight: `freshness = max(0, 1 - filed_after_days/45)` then  
`trade_w = ±log1p(size) * (0.25 + 0.75 * freshness)` — late filings barely count.

Results print to the terminal and write `output/screen.csv`.

## Scoring defaults

| Pillar | Weight |
|--------|--------|
| S/R proximity | 30% |
| SMA regime | 20% |
| Market (VIX) | 15% |
| News | 15% |
| Earnings proximity | 10% |
| Social (Reddit ± Grok) | 10% |

Hard filters still drop illiquid names, tight ranges, resistance-hugging prices, below-SMA200 (unless `--allow-below-200`), death crosses, and news vetoes.

## Notes

- Full US scans should use `--limit` or a nightly job; Yahoo rate-limits.  
- Reddit matching is noisy — treat as a soft signal.  
- Grok runs only on `--grok-top` survivors (default 15).  
- Not investment advice.
