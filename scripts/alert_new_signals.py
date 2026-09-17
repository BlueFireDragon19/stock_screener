#!/usr/bin/env python3
"""Detect newly signaled tickers vs the previous screen snapshot and alert.

Compares current ``output/screen_*.csv`` (+ optional ``today_focus.csv``) to
``output/signal_snapshot.json``. First run only seeds the baseline (no flood).

Alerts (any that are configured):
  - stdout (always)
  - macOS Notification Center (default on Darwin; ``--no-desktop`` to skip)
  - Slack incoming webhook if ``SCREENER_SLACK_WEBHOOK`` or ``SLACK_WEBHOOK_URL`` is set

Near-real-time: re-run the daily screen on an interval (e.g. Cursor ``/loop 30m``
or cron), then this script — the screener itself is batch, not a live stream.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
SNAPSHOT = OUTPUT / "signal_snapshot.json"
MODES = (
    "support",
    "catalyst",
    "fundamentals",
    "politicians",
    "trump",
    "athdip",
)


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _tickers_from_csv(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return []
    if "ticker" not in df.columns:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for t in df["ticker"].astype(str):
        t = t.strip().upper()
        if not t or t == "NAN" or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def collect_current(output: Path = OUTPUT) -> dict:
    by_mode: dict[str, list[str]] = {}
    for mode in MODES:
        # Prefer screen_{mode}.csv; also accept screen.csv for single-mode runs
        path = output / f"screen_{mode}.csv"
        if not path.exists() and mode == "support":
            alt = output / "screen.csv"
            if alt.exists():
                path = alt
        by_mode[mode] = _tickers_from_csv(path)

    focus_rows: list[dict] = []
    focus_path = output / "today_focus.csv"
    if focus_path.exists():
        try:
            df = pd.read_csv(focus_path)
            for _, row in df.iterrows():
                t = str(row.get("ticker") or "").strip().upper()
                if not t:
                    continue
                focus_rows.append(
                    {
                        "ticker": t,
                        "kind": str(row.get("kind") or ""),
                        "why": str(row.get("why") or "")[:160],
                        "tags": str(row.get("tags") or ""),
                    }
                )
        except Exception:  # noqa: BLE001
            focus_rows = []

    union: list[str] = []
    seen: set[str] = set()
    for mode in MODES:
        for t in by_mode.get(mode) or []:
            if t not in seen:
                seen.add(t)
                union.append(t)

    return {
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "by_mode": by_mode,
        "focus": [r["ticker"] for r in focus_rows],
        "focus_detail": focus_rows,
        "all": union,
    }


def load_snapshot(path: Path = SNAPSHOT) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def save_snapshot(data: dict, path: Path = SNAPSHOT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def diff_new(prev: dict, cur: dict) -> dict[str, list[str]]:
    """Return new tickers overall and per mode / focus."""
    prev_all = set(prev.get("all") or [])
    prev_focus = set(prev.get("focus") or [])
    prev_modes = prev.get("by_mode") or {}

    new_all = [t for t in cur.get("all") or [] if t not in prev_all]
    new_focus = [t for t in cur.get("focus") or [] if t not in prev_focus]
    new_by_mode: dict[str, list[str]] = {}
    for mode, tickers in (cur.get("by_mode") or {}).items():
        before = set(prev_modes.get(mode) or [])
        added = [t for t in tickers if t not in before]
        if added:
            new_by_mode[mode] = added
    return {"all": new_all, "focus": new_focus, "by_mode": new_by_mode}


def _modes_for(ticker: str, cur: dict) -> list[str]:
    return [m for m, ts in (cur.get("by_mode") or {}).items() if ticker in (ts or [])]


def format_alert(new: dict, cur: dict) -> str:
    lines: list[str] = []
    lines.append(f"Screener new signals · {cur.get('as_of')}")
    if new.get("focus"):
        lines.append("")
        lines.append("New in Today's focus:")
        detail = {r["ticker"]: r for r in (cur.get("focus_detail") or [])}
        for t in new["focus"]:
            d = detail.get(t) or {}
            why = d.get("why") or ""
            kind = d.get("kind") or ""
            modes = ", ".join(_modes_for(t, cur)) or (d.get("tags") or "")
            bit = f"  • {t}"
            if kind:
                bit += f" [{kind}]"
            if modes:
                bit += f" · {modes}"
            if why:
                bit += f" — {why}"
            lines.append(bit)
    if new.get("all"):
        lines.append("")
        lines.append("New in any mode:")
        for t in new["all"]:
            modes = ", ".join(_modes_for(t, cur)) or "?"
            lines.append(f"  • {t} · {modes}")
    if new.get("by_mode"):
        lines.append("")
        lines.append("By mode:")
        for mode, tickers in new["by_mode"].items():
            lines.append(f"  {mode}: {', '.join(tickers)}")
    if not new.get("all") and not new.get("focus"):
        lines.append("No new symbols vs last snapshot.")
    return "\n".join(lines)


def notify_desktop(title: str, body: str) -> None:
    if platform.system() != "Darwin":
        return
    # Keep notification short
    short = body.replace("\n", " · ")[:180]
    script = (
        f'display notification {json.dumps(short)} with title {json.dumps(title)}'
    )
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=5)
    except Exception:  # noqa: BLE001
        pass


def notify_slack(text: str) -> bool:
    url = (
        os.getenv("SCREENER_SLACK_WEBHOOK")
        or os.getenv("SLACK_WEBHOOK_URL")
        or ""
    ).strip()
    if not url:
        return False
    try:
        resp = requests.post(url, json={"text": text}, timeout=15)
        return resp.status_code < 300
    except Exception as exc:  # noqa: BLE001
        print(f"Slack webhook failed: {exc}", file=sys.stderr)
        return False


def run(*, desktop: bool = True, slack: bool = True, dry_run: bool = False) -> int:
    _load_dotenv()
    cur = collect_current()
    if not cur.get("all") and not cur.get("focus"):
        print("No screen CSVs / today_focus found under output/. Run the screener first.")
        return 1

    prev = load_snapshot()
    if prev is None:
        if not dry_run:
            save_snapshot(cur)
        print(
            f"Baseline saved ({len(cur['all'])} mode tickers, "
            f"{len(cur['focus'])} focus). No alert on first run."
        )
        print(f"Snapshot → {SNAPSHOT}")
        return 0

    new = diff_new(prev, cur)
    msg = format_alert(new, cur)
    print(msg)

    has_new = bool(new.get("all") or new.get("focus"))
    if has_new and not dry_run:
        if desktop:
            n = len(set(new.get("all") or []) | set(new.get("focus") or []))
            notify_desktop(
                "Screener · new signals",
                f"{n} new symbol(s): {', '.join((new.get('focus') or new.get('all') or [])[:8])}",
            )
        if slack:
            ok = notify_slack(f"```{msg}```")
            if ok:
                print("Slack webhook: sent")
            elif (
                os.getenv("SCREENER_SLACK_WEBHOOK") or os.getenv("SLACK_WEBHOOK_URL")
            ):
                print("Slack webhook: failed", file=sys.stderr)
        save_snapshot(cur)
        print(f"Updated snapshot → {SNAPSHOT}")
    elif not dry_run:
        # Still refresh snapshot so drops don't re-alert as new later incorrectly
        # Actually: if we update when no new, tickers that left and return later WILL alert — good.
        save_snapshot(cur)
        print(f"Snapshot refreshed (no new symbols) → {SNAPSHOT}")

    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-desktop", action="store_true", help="Skip macOS notification")
    p.add_argument("--no-slack", action="store_true", help="Skip Slack webhook")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Diff only; do not write snapshot or send alerts",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Delete snapshot and re-seed baseline from current CSVs",
    )
    args = p.parse_args(argv)
    if args.reset and SNAPSHOT.exists():
        SNAPSHOT.unlink()
        print(f"Removed {SNAPSHOT}")
    return run(
        desktop=not args.no_desktop,
        slack=not args.no_slack,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
