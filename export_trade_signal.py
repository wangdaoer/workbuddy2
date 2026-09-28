"""实盘信号导出：把 overlay 输出的入选股转成手动跟单交易单。

设计原则（与用户 2026-09-28 约定）：
- 用户选择「直接目标仓位」= 满仓部署，每只取 overlay 的 personal_adjusted_target_weight。
  注意：权威 overlay（full_overlay_calibrated_blended.csv）已含市场宽度降仓，故「满仓列」
  即模型当日实际推荐权重（弱势市可能 <100%）。
- 同时并排给出「regime 缩放参考权重」：regime_scale × 满仓权重。regime_scale 取值优先级：
  CLI --regime-scale > trade_config.json > 当日 production 的 recommended_gross_scale
  （outputs/ab_pit/regime_monitor_*.json）> 缺省 0.676（decaying 状态经验值）。
  最终仓位由用户拍板。
- 不自动下单、不强制降仓——只导出可读交易单。
- 账户资金从 trade_config.json 读取（CLI --capital 可覆盖）。

用法：
  python export_trade_signal.py                      # 用 trade_config.json 的 capital
  python export_trade_signal.py --capital 500000      # 显式指定资金
  python export_trade_signal.py --overlay <path>      # 指定 overlay（自动化里指向 blended 版）
产物：
  outputs/trade_sheet_<面板末日>.csv   机器可读
  outputs/trade_sheet_<面板末日>.md    人工可读交易单
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OVERLAY = HERE / "outputs" / "watchlist_audit" / "full_overlay_calibrated.csv"
PANEL = HERE / "external_data" / "daily-market-data" / "data_panel.csv"
OUT = HERE / "outputs"
CONFIG = HERE / "trade_config.json"
DEFAULT_REGIME = 0.676  # decaying 状态经验值；若当日 production 给出 recommended_gross_scale 则覆盖


def _is_selected(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "1.0")


def _load_config(path: Path) -> dict:
    if path.exists():
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _resolve_regime_scale(cli_val, cfg: dict) -> tuple[float, str]:
    """返回 (scale, 来源说明)。"""
    if cli_val is not None:
        return cli_val, f"CLI 指定 {cli_val}"
    if cfg.get("regime_scale") is not None:
        return float(cfg["regime_scale"]), "trade_config.json"
    # 当日 production 的 recommended_gross_scale
    fs = sorted(glob.glob(str(HERE / "outputs" / "ab_pit" / "regime_monitor_*.json")))
    if fs:
        try:
            d = json.load(open(fs[-1], encoding="utf-8"))
            if d.get("recommended_gross_scale") is not None:
                return float(d["recommended_gross_scale"]), \
                    f"当日 production recommended_gross_scale={d['recommended_gross_scale']}"
        except Exception:
            pass
    return DEFAULT_REGIME, f"缺省 {DEFAULT_REGIME}（decaying 经验值）"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", default=str(OVERLAY), help="overlay 结构化输出 CSV")
    ap.add_argument("--capital", type=float, default=None, help="账户总资金(元)；缺省读 trade_config.json")
    ap.add_argument("--regime-scale", type=float, default=None, help="regime 缩放；缺省自动取/0.676")
    ap.add_argument("--config", default=str(CONFIG), help="交易配置文件")
    args = ap.parse_args()

    cfg = _load_config(Path(args.config))
    capital = args.capital if args.capital is not None else float(cfg.get("capital", 100000.0))
    regime, regime_src = _resolve_regime_scale(args.regime_scale, cfg)

    df = pd.read_csv(args.overlay)
    if "personal_selected" not in df.columns:
        raise SystemExit("overlay CSV 缺少 personal_selected 列")
    sel = df[df["personal_selected"].map(_is_selected)].copy()
    if sel.empty:
        raise SystemExit("没有 personal_selected=True 的入选股，overlay 可能未跑")

    sort_col = "personal_rank" if "personal_rank" in sel.columns else \
        ("final_score" if "final_score" in sel.columns else None)
    if sort_col == "personal_rank":
        sel = sel.sort_values("personal_rank")
    elif sort_col == "final_score":
        sel = sel.sort_values("final_score", ascending=False)

    p = pd.read_csv(PANEL, usecols=["date"])
    data_date = str(p["date"].iloc[-1])

    rows = []
    for _, r in sel.iterrows():
        sym = str(r["symbol"]).zfill(6)
        name = str(r.get("stock_name", ""))
        score = r.get("final_score", r.get("personal_adjusted_score", float("nan")))
        try:
            score = round(float(score), 4)
        except Exception:
            score = float("nan")
        w_full = float(r.get("personal_adjusted_target_weight", 0.05))
        w_reg = w_full * regime
        rows.append({
            "代码": sym, "名称": name, "评分": score,
            "满仓权重": w_full, "满仓买入额": capital * w_full,
            "保守权重(regime)": w_reg, "保守买入额": capital * w_reg,
        })
    out_df = pd.DataFrame(rows)
    total_full = out_df["满仓权重"].sum()
    total_reg = out_df["保守权重(regime)"].sum()

    csv_path = OUT / f"trade_sheet_{data_date}.csv"
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = []
    md.append(f"# 实盘信号单（面板末日 {data_date} / 生成 {today}）\n")
    md.append("> ⚠️ **风险告知**\n")
    md.append(f"> - 数据基于面板末日 **{data_date}**，可能滞后于今日行情，**下单前请核对实时价与停牌/涨跌停**")
    md.append(f"> - 当前 regime 缩放参考 = **{regime}**（来源：{regime_src}）；「保守列」即按此缩放")
    md.append(f"> - 回测为 **gross（未扣成本）**，实盘净收益会更低；佣金/印花税/滑点未计入")
    md.append(f"> - 「满仓」= 模型当日实际推荐权重（已含市场宽度降仓）；最终仓位你自己拍板\n")
    md.append(f"**账户资金**：¥{capital:,.0f}（来源：{'CLI' if args.capital else 'trade_config.json'}）\n")
    md.append("| # | 代码 | 名称 | 评分 | 满仓权重 | 满仓买入额(元) | 保守权重 | 保守买入额(元) |")
    md.append("|---:|------|------|------:|------:|------:|------:|------:|")
    for i, r in out_df.iterrows():
        md.append(
            f"| {i+1} | {r['代码']} | {r['名称']} | {r['评分']} | "
            f"{r['满仓权重']*100:.1f}% | {r['满仓买入额']:,.0f} | "
            f"{r['保守权重(regime)']*100:.1f}% | {r['保守买入额']:,.0f} |"
        )
    md.append("")
    md.append(f"**合计**：满仓 {total_full*100:.1f}%（¥{capital*total_full:,.0f}）"
              f" ／ 保守 {total_reg*100:.1f}%（¥{capital*total_reg:,.0f}）")
    md.append("")
    md.append("> 来源：overlay 入选（strongD 默认 + 个人行为叠加层）。本单仅作信号，不构成投资建议。")
    md_path = OUT / f"trade_sheet_{data_date}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"入选 {len(out_df)} 只 | 满仓 {total_full*100:.1f}% / 保守 {total_reg*100:.1f}%  (regime={regime}, {regime_src})")
    print(f"CSV  -> {csv_path}")
    print(f"MD   -> {md_path}")


if __name__ == "__main__":
    main()
