"""Bootstrap the initial 510300 benchmark CSV when none exists yet.

The refresh script ``update_benchmark_510300.py`` appends from an existing
benchmark's latest date, so the very first file must be built separately.
This bootstraps the full history (2022-12-05..asof) from **Sohu + Yahoo**
with per-date 2-source agreement (same quorum rules as the refresh script),
then the refresh script can take over incrementally with the 3-source quorum.

Output schema (exact, matches update_benchmark_510300.COLUMNS):
    date,open,high,low,close,volume,amount

Usage:
    python bootstrap_benchmark_510300.py [--asof-date 2026-08-05] [--output external_data/benchmarks/510300.csv]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from update_benchmark_510300 import (  # noqa: E402
    BenchmarkRow,
    _validate_agreement,
    fetch_sohu_history,
    fetch_yahoo_history,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap initial 510300 benchmark from Sohu+Yahoo.")
    parser.add_argument("--asof-date", default="2026-08-05")
    parser.add_argument("--start-date", default="2022-12-05")
    parser.add_argument("--output", default="external_data/benchmarks/510300.csv")
    parser.add_argument("--symbol", default="510300")
    args = parser.parse_args()

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.asof_date)
    symbol = args.symbol

    print(f"[bootstrap] fetching {symbol} history {start}..{end} from Sohu+Yahoo ...")
    sohu = fetch_sohu_history(symbol, start, end)
    yahoo = fetch_yahoo_history(symbol, start, end)
    print(f"[bootstrap] sohu days={len(sohu)}  yahoo days={len(yahoo)}")

    common = sorted(set(sohu) & set(yahoo))
    rows: list[BenchmarkRow] = []
    skipped: list[str] = []
    for d in common:
        try:
            _validate_agreement(sohu[d], yahoo[d], "Sohu/Yahoo")
        except ValueError as exc:
            skipped.append(f"{d.isoformat()}: {exc}")
            continue
        rows.append(
            BenchmarkRow(
                date=d,
                open=yahoo[d].open,
                high=yahoo[d].high,
                low=yahoo[d].low,
                close=yahoo[d].close,
                volume=yahoo[d].volume,
                amount=sohu[d].amount,
            )
        )

    if not rows:
        raise SystemExit("no rows survived the 2-source agreement check; aborting bootstrap")
    only_sohu = sorted(set(sohu) - set(yahoo))
    only_yahoo = sorted(set(yahoo) - set(sohu))
    print(f"[bootstrap] agreed rows={len(rows)}  skipped_mismatch={len(skipped)} "
          f"only_sohu={len(only_sohu)} only_yahoo={len(only_yahoo)}")
    for item in skipped[:10]:
        print("  mismatch:", item)

    frame = pd.DataFrame([row.as_record() for row in rows])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8", date_format="%Y-%m-%d")
    print(f"[bootstrap] wrote {output} rows={len(frame)} days={frame['date'].nunique()} "
          f"range={frame['date'].min()}..{frame['date'].max()}")


if __name__ == "__main__":
    main()
