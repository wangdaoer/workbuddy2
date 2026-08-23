"""Read-only trading-calendar audit of our merged panel (P0-2, network-free).

Builds the expected session calendar from TWO authoritative sources and runs the
ported model4 ``trading_calendar.trading_session_audit``:
  * Tdx daily files ``ths_hs_a_share_YYYY-MM-DD.csv`` -> the historical backbone
    sessions (880 days, 2022-12-05..2026-07-23);
  * the panel's own weekday dates -> the recent Tencent tail extension.

This catches:
  * invalid_panel_dates : panel rows on a non-session day (weekend/holiday)
  * missing_panel_dates : a Tdx/panel session absent from the merged panel
                          (real ingestion-gap risk)

DIAGNOSTIC ONLY -- never rewrites the panel. The repair/quarantine step
(repair_panel_trading_calendar.py) is kept separate and requires explicit
confirmation because it mutates the 4.89M-row panel.

Usage:
    python model3_data_pipeline/audit_panel_calendar.py
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from trading_calendar import trading_session_audit  # noqa: E402

PANEL = Path("external_data/daily-market-data/data_panel.csv")
TDX_DIR = Path("external_data/daily-market-data-tdx/ths_exports/normalized")
BENCHMARK_CSV = Path("external_data/daily-market-data/benchmark_sessions_from_tdx.csv")


def tdx_session_dates() -> list[pd.Timestamp]:
    pat = re.compile(r"ths_hs_a_share_(\d{4}-\d{2}-\d{2})\.csv$")
    dates = []
    for p in glob.glob(str(TDX_DIR / "ths_hs_a_share_*.csv")):
        m = pat.search(Path(p).name)
        if m:
            dates.append(pd.Timestamp(m.group(1)))
    return sorted(set(dates))


def main() -> None:
    print(f"[audit] reading panel dates from {PANEL} ...")
    panel_dates = pd.to_datetime(
        pd.read_csv(PANEL, usecols=["date"], low_memory=False)["date"]
    )
    print(f"[audit] panel rows={len(panel_dates):,}  unique_dates={panel_dates.nunique()}")

    tdx = pd.DatetimeIndex(tdx_session_dates())
    print(f"[audit] Tdx backbone sessions: n={len(tdx)} ({tdx.min().date()}..{tdx.max().date()})")

    # Expected sessions = Tdx backbone (authoritative) + panel's own weekday dates
    # (the Tencent tail extension). All are weekdays by construction.
    panel_weekdays = pd.DatetimeIndex(panel_dates.dropna().unique()).normalize()
    panel_weekdays = panel_weekdays[panel_weekdays.dayofweek < 5]
    expected = pd.DatetimeIndex(sorted(set(tdx) | set(panel_weekdays))).sort_values()
    BENCHMARK_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": expected.strftime("%Y-%m-%d")}).to_csv(BENCHMARK_CSV, index=False)
    print(f"[audit] expected sessions (Tdx ∪ panel-weekdays): n={len(expected)} -> {BENCHMARK_CSV}")

    result = trading_session_audit(panel_dates, BENCHMARK_CSV)
    print("\n=== trading-session audit ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # Extra diagnostics
    tdx_missing = sorted(set(tdx) - set(panel_weekdays))
    tail = sorted(set(panel_weekdays) - set(tdx))
    print(f"\n[info] Tdx sessions MISSING from panel: {len(tdx_missing)} "
          f"{[d.strftime('%Y-%m-%d') for d in tdx_missing[:10]]}")
    print(f"[info] panel dates BEYOND Tdx range (Tencent tail): {len(tail)} "
          f"{[d.strftime('%Y-%m-%d') for d in tail[:12]]}")

    if result["passed"]:
        print("\n[OK] panel trading calendar is consistent with the Tdx+tail session reference "
              "(no weekend/holiday rows, no Tdx ingestion gaps).")
    else:
        print(f"\n[WARN] inconsistencies: "
              f"invalid={len(result['invalid_panel_dates'])} missing={len(result['missing_panel_dates'])}")


if __name__ == "__main__":
    main()
