# Stock screener (long-only)

Python screener with six long-only modes + **VIX regime**:

1. **Support** — near support / SMA pullback (stricter in risk-off)  
2. **Catalyst** — news / earnings / volume pops (needs vol+green day in risk-off)  
3. **Fundamentals** — Feroldi-style growth, profitability, balance sheet, ROIC, valuation  
4. **Politicians** — Congress trades from **Capitol Trades scrape** + **official House/Senate** disclosures, weighted so **shorter `filed_after` days score higher**  
5. **Trump** — Truth Social (trumpstruth.org RSS) + Google News RSS + White House RSS + optional Grok X; maps cashtags / company names / policy themes → tickers  
6. **Athdip** — uptrend near multi-year ATH with **4h RSI ~30** (pass if ≤35)

Shared free inputs: Yahoo OHLCV + news (+ X-quoting Google News RSS) + Polymarket odds + earnings + fundamentals, `^VIX`, STOCK Act public filings, optional Reddit + Grok X.

## Free data sources

| Source | Use |
|--------|-----|
| NASDAQ Trader symbol dirs | Full US listed universe (`--universe us`) |
| Wikipedia | S&P 500 (`--universe sp500`) |
| Yahoo Finance / yfinance | OHLCV, VIX, news, earnings |
| Google News RSS | Per-ticker X/Twitter-quoted headlines (blended into support/catalyst news); also Trump / tariff / Fed / trade |
| Polymarket Gamma API | Per-ticker prediction-market odds (blended into support/catalyst social; sparse outside mega-caps) |
| trumpstruth.org RSS | Truth Social mirror for @realDonaldTrump |
| whitehouse.gov RSS | Official news, presidential actions, briefings |
| Reddit public JSON / optional OAuth | Retail mention / polarity |
| xAI Grok API | Optional X sentiment + Trump X posts |

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

# Trump tracker (Truth + news + WH; add Grok for X if credits available)
stock-screener --mode trump --no-trump-x --top 25

# Near ATH + 4h RSI ~30 (dip in uptrend)
stock-screener --mode athdip --universe sp500 --limit 80 --top 25
```

Politician freshness weight: `freshness = max(0, 1 - filed_after_days/45)` then  
`trade_w = ±log1p(size) * (0.25 + 0.75 * freshness)` — late filings barely count.

Trump tracker weights Truth Social > White House > X > news, with ~36h freshness half-life, then maps themes (tariffs, China, energy, defense, banks, etc.) to liquid tickers/ETFs.

Athdip: **setup** = multi-year ATH break (21d fresh high at ATH time); **trigger** = **min 4h RSI ≤ `--athdip-rsi-max`** over `--athdip-rsi-lookback` days (defaults **31** / **5**) while ATH is within `--athdip-max-ath-age` sessions (default **63**). Universe `--limit` keeps liquid mega-caps (NVDA, AMD, …) before alphabetical fill.

Results print to the terminal and write `output/screen.csv`.

### Daily dashboard (Today's focus)

After a screen run, rebuild the Cursor canvas + `output/today_focus.csv`:

```bash
python3 scripts/build_screen_dashboard.py
```

Project skill (for agents / teammates): `.cursor/skills/daily-screen/SKILL.md` — covers the full reproduce path (screen → focus → canvas).

## Scoring defaults

| Pillar | Weight |
|--------|--------|
| S/R proximity | 30% |
| SMA regime | 20% |
| Market (VIX) | 15% |
| News (Yahoo ± X-quote RSS) | 15% |
| Earnings proximity | 10% |
| Social (Reddit ± Grok ± Polymarket) | 10% |

Hard filters still drop illiquid names, tight ranges, resistance-hugging prices, below-SMA200 (unless `--allow-below-200`), death crosses, and news vetoes. X-quoting headlines tilt news score (~35% when both Yahoo and X-RSS fire); disable with `--no-x-quote-news`. Polymarket tilts social (~35% when open markets exist) and `why` notes source plus news/reddit agree/diverge; disable with `--no-polymarket`.

## Notes

- Full US scans should use `--limit` or a nightly job; Yahoo rate-limits.  
- Reddit matching is noisy — treat as a soft signal.  
- Grok runs only on `--grok-top` survivors (default 15).  
- Not investment advice.
