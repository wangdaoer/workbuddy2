#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""公平 A/B 对照：读取两个 --monitor 产出的 regime_monitor_<token>.json，
打印 PIT（全市场诚实宇宙）vs WATCHLIST（自选股诚实宇宙）在 [start_date, 末日] 同一窗口的可比指标。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKEN = "20260805"
PIT = HERE / "outputs" / "ab_pit" / f"regime_monitor_{TOKEN}.json"
WL = HERE / "outputs" / "ab_watchlist" / f"regime_monitor_{TOKEN}.json"


def load(p: Path):
    if not p.exists():
        raise SystemExit(f"缺失: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def row(label, rep):
    b = rep.get("backtest") or {}
    bf = rep.get("backtest_full") or {}
    return {
        "label": label,
        "universe": rep.get("universe"),
        "window": rep.get("window"),
        "asof": rep.get("asof"),
        "status": rep.get("status"),
        "win_total": b.get("total_return"),
        "win_sharpe": b.get("sharpe_like"),
        "win_dd": b.get("max_drawdown"),
        "win_days": b.get("trade_days"),
        "full_total": bf.get("total_return"),
        "full_sharpe": bf.get("sharpe_like"),
        "full_dd": bf.get("max_drawdown"),
        "trailing_ic": rep.get("trailing_ic_current"),
        "rec_scale": rep.get("recommended_gross_scale"),
        "alert": (rep.get("alert") if "alert" in rep else None),
    }


def main():
    pit = load(PIT)
    wl = load(WL)
    rp = row("全市场诚实 PIT 宇宙", pit)
    rw = row("自选股诚实 PIT 宇宙", wl)

    def f(v, fmt="{:.4f}"):
        return "—" if v is None else fmt.format(v)

    print("=" * 72)
    print(f"公平 A/B 对照  |  窗口 = [{rp['window']} → {rp['asof']}]  (合并面板 →08-05)")
    print("=" * 72)
    print(f"{'指标':<22}{'PIT(全市场)':>18}{'WATCHLIST(自选)':>20}{'差额':>12}")
    print("-" * 72)
    print(f"{'窗口总收益':<20}{f(rp['win_total'],'{:+.2%}'):>18}{f(rw['win_total'],'{:+.2%}'):>20}{f(rw['win_total']-rp['win_total'],'{:+.2%}'):>12}")
    print(f"{'窗口 Sharpe':<20}{f(rp['win_sharpe']):>18}{f(rw['win_sharpe']):>20}{f(rw['win_sharpe']-rp['win_sharpe']):>12}")
    print(f"{'窗口最大回撤':<20}{f(rp['win_dd'],'{:+.2%}'):>18}{f(rw['win_dd'],'{:+.2%}'):>20}{f(rw['win_dd']-rp['win_dd'],'{:+.2%}'):>12}")
    print(f"{'窗口交易日':<20}{str(rp['win_days']):>18}{str(rw['win_days']):>20}")
    print("-" * 72)
    print(f"{'全程总收益(对照)':<18}{f(rp['full_total'],'{:+.2%}'):>18}{f(rw['full_total'],'{:+.2%}'):>20}")
    print(f"{'全程 Sharpe(对照)':<18}{f(rp['full_sharpe']):>18}{f(rw['full_sharpe']):>20}")
    print(f"{'全程最大回撤(对照)':<18}{f(rp['full_dd'],'{:+.2%}'):>18}{f(rw['full_dd'],'{:+.2%}'):>20}")
    print("-" * 72)
    print(f"regime: PIT status={rp['status']} ic={f(rp['trailing_ic'])} scale={f(rp['rec_scale'])}")
    print(f"regime: WL  status={rw['status']} ic={f(rw['trailing_ic'])} scale={f(rw['rec_scale'])}")
    print("=" * 72)
    # 结论速写
    d_sh = (rw['win_sharpe'] or 0) - (rp['win_sharpe'] or 0)
    d_tot = (rw['win_total'] or 0) - (rp['win_total'] or 0)
    verdict = "自选股宇宙在该窗口跑赢全市场" if d_sh > 0 else "全市场宇宙在该窗口更优（自选股未带来超额）"
    print(f"结论：窗口 Sharpe 差 = {d_sh:+.3f}，总收益差 = {d_tot:+.2%} → {verdict}")
    print("=" * 72)


if __name__ == "__main__":
    main()
