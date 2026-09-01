#!/usr/bin/env python3
"""Generate wiki/Home.md from output CSVs and push to GitHub wiki."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
WIKI_MD = ROOT / "wiki" / "Home.md"
WIKI_REMOTE = "https://github.com/BlueFireDragon19/stock_screener.wiki.git"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt_num(value: str | None, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return value or "—"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_home_md() -> str:
    support_path = OUTPUT / "screen_support.csv"
    catalyst_path = OUTPUT / "screen_catalyst.csv"
    fund_path = OUTPUT / "screen_fundamentals.csv"
    for path in (support_path, catalyst_path, fund_path):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run the screener first.")

    support = read_csv(support_path)
    catalyst = read_csv(catalyst_path)
    fund = read_csv(fund_path)
    as_of = datetime.fromtimestamp(support_path.stat().st_mtime).strftime("%Y-%m-%d")
    regime = support[0].get("market_label", "—") if support else "—"

    support_rows = [
        [
            r["ticker"],
            fmt_num(r["price"]),
            fmt_num(r["rsi14"], 1),
            r.get("rsi_bias", "—"),
            fmt_num(r["proximity"], 3),
            fmt_num(r["high_52w"]),
            fmt_num(r["low_52w"]),
            f"{fmt_num(r['pct_from_52w_high'], 1)}%",
            f"{fmt_num(r['pct_from_52w_low'], 1)}%",
            fmt_num(r["high_1w"]),
            fmt_num(r["low_1w"]),
            f"{fmt_num(r['pct_from_1w_high'], 1)}%",
            fmt_num(r["composite"]),
        ]
        for r in support
    ]

    catalyst_rows = [
        [
            r["ticker"],
            fmt_num(r["price"]),
            fmt_num(r["rsi14"], 1),
            r.get("rsi_bias", "—"),
            f"{fmt_num(r['day_return_pct'], 2)}%",
            fmt_num(r["volume_ratio"], 2),
            fmt_num(r["high_52w"]),
            fmt_num(r["low_52w"]),
            f"{fmt_num(r['pct_from_52w_high'], 1)}%",
            fmt_num(r["high_1w"]),
            fmt_num(r["low_1w"]),
            f"{fmt_num(r['pct_from_1w_high'], 1)}%",
            fmt_num(r["composite"]),
        ]
        for r in catalyst[:10]
    ]

    fund_rows = [
        [
            r["ticker"],
            (r.get("sector") or "—")[:24],
            fmt_num(r["composite"]),
            fmt_num(r["growth_score"], 1),
            fmt_num(r["profitability_score"], 1),
            fmt_num(r["valuation_score"], 1),
        ]
        for r in fund[:10]
    ]

    return f"""# Screener dashboard

**As of:** {as_of} · **Regime:** {regime}

Long-only stock screener results (auto-generated from latest CSV output).

## Support

{md_table([
    "Ticker", "Price", "RSI", "RSI bias", "Prox", "52w H", "52w L", "% vs 52wH", "% vs 52wL", "1w H", "1w L", "% vs 1wH", "Score"
], support_rows)}

## Catalyst

{md_table([
    "Ticker", "Price", "RSI", "RSI bias", "Day %", "Vol×", "52w H", "52w L", "% vs 52wH", "1w H", "1w L", "% vs 1wH", "Score"
], catalyst_rows)}

## Fundamentals (top 10)

{md_table([
    "Ticker", "Sector", "Score", "Growth", "Profit", "Value"
], fund_rows)}

---

### Refresh

```bash
stock-screener --mode both --no-politicians
python3 scripts/publish_wiki.py --push
```

Source: `output/screen_support.csv`, `output/screen_catalyst.csv`, `output/screen_fundamentals.csv`
"""


def write_home() -> Path:
    WIKI_MD.parent.mkdir(parents=True, exist_ok=True)
    WIKI_MD.write_text(build_home_md(), encoding="utf-8")
    return WIKI_MD


def push_wiki(home_md: Path) -> None:
    probe = subprocess.run(
        ["git", "ls-remote", WIKI_REMOTE],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise SystemExit(
            "GitHub wiki is not initialized yet. Open "
            "https://github.com/BlueFireDragon19/stock_screener/wiki/_new "
            "and save any first page, then re-run with --push."
        )

    with tempfile.TemporaryDirectory() as tmp:
        wiki_dir = Path(tmp) / "wiki"
        subprocess.run(["git", "clone", WIKI_REMOTE, str(wiki_dir)], check=True)
        (wiki_dir / "Home.md").write_text(home_md.read_text(encoding="utf-8"), encoding="utf-8")
        subprocess.run(["git", "-C", str(wiki_dir), "add", "Home.md"], check=True)
        status = subprocess.run(["git", "-C", str(wiki_dir), "status", "--porcelain"], capture_output=True, text=True, check=True)
        if not status.stdout.strip():
            print("Wiki already up to date.")
            return
        subprocess.run(
            [
                "git",
                "-C",
                str(wiki_dir),
                "-c",
                "user.name=BlueFireDragon19",
                "-c",
                "user.email=BlueFireDragon19@users.noreply.github.com",
                "commit",
                "-m",
                "Update screener dashboard from CSV output.",
            ],
            check=True,
        )
        for branch in ("master", "main"):
            push = subprocess.run(["git", "-C", str(wiki_dir), "push", "-u", "origin", f"HEAD:{branch}"])
            if push.returncode == 0:
                print(f"Pushed wiki to {branch}.")
                return
        raise SystemExit("Wiki push failed.")


def main() -> None:
    home = write_home()
    print(f"Wrote {home}")
    if "--push" in sys.argv:
        push_wiki(home)
        print("https://github.com/BlueFireDragon19/stock_screener/wiki")


if __name__ == "__main__":
    main()
