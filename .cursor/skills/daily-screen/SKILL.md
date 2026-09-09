---
name: daily-screen
description: >-
  Run the stock-screener all-mode daily refresh, build Today's focus
  (intersections/conflicts + single-mode leaders with fundamentals), and
  regenerate the Cursor screen dashboard canvas. Use when the user says
  update for today, refresh the screen, rebuild dashboard, today's focus,
  or reproduce the screener dashboard.
---

# Daily screen + dashboard

## Goal

Anyone with this repo can reproduce today's multi-mode screen and the **Today's focus** dashboard section.

## Reproduce (canonical)

```bash
cd /path/to/stock-screener
source .venv/bin/activate

# 1) Live screen (writes output/screen_*.csv)
stock-screener --mode all --universe sp500 --limit 80 --top 25 \
  --allow-below-200 --no-reddit --no-grok --no-trump-x \
  --output output/screen.csv 2>&1 | tee output/run_today.log

# 2) Build focus CSV + Cursor canvas + wiki section
python3 scripts/build_screen_dashboard.py
```

Optional YTD backtest (slow):

```bash
python3 scripts/backtest_ytd.py --modes support,catalyst,value,trump,athdip --limit 80 --top 10 \
  2>&1 | tee output/backtest_ytd_all.log
# Optionally save summary JSON for the canvas YTD chart:
# output/backtest_ytd_summary.json
```

## Outputs

| Path | Purpose |
|------|---------|
| `output/screen_*.csv` | Per-mode screener results |
| `output/today_focus.csv` | Today's focus rows (intersections + singles + fundamentals) |
| `output/run_today.log` | VIX / Trump feed counts for the canvas header |
| `docs/screen-dashboard.canvas.tsx` | Repo copy of the dashboard (git-friendly) |
| Cursor `canvases/screen-dashboard.canvas.tsx` | Live IDE canvas (managed folder) |
| `wiki/Home.md` | Markdown dashboard including Today's focus |

## Today's focus rules

The builder (`scripts/build_screen_dashboard.py`) picks:

1. **Athdip hard passes** — only if **trigger day is today** OR **current price < trigger price**
2. **Athdip ∩ politicians** — `conflict` if sells > buys, else `confirm` / `intersection`
3. **Support ≥ 75 and fundamentals ≥ 70** — intersection
4. **Top single-mode leaders** — support, catalyst, trump, politicians (fundamentals if room)

Each row includes mode scores when present plus **sector / PE / PEG / FCF% / ROE%** (from screener CSVs, else Yahoo fill-in).

## Agent checklist

When the user asks to "update for today":

1. Run the screener command above (network required).
2. Run `python3 scripts/build_screen_dashboard.py`.
3. Open the canvas: `canvases/screen-dashboard.canvas.tsx` (or the path printed by the script).
4. Summarize Today's focus briefly in chat (conflicts first).

Do **not** hand-edit the canvas; regenerate it from CSVs so results stay reproducible.

## Notes

- `output/` is gitignored — commit the **scripts + skill + docs canvas**, not daily CSVs.
- Label results as a **watchlist**, not investment advice.
- Athdip setup = multi-year ATH break; trigger = min 4h RSI ≤ 31 over 5 calendar days while ATH age ≤ 63 sessions.
