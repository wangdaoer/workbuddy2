"""P10g-production：p10e_soft regime 门控 MLP 的生产化打分 + regime 监控模块。

把 P10e 验证过的「因果连续混合 regime 门控 MLP」封装成一个可每日调用的生产单元：

  produce_book_score(panel)
    -> (soft_score, regime_report, book_equity)
       - soft_score     : (date x symbol) walk-forward 因果门控 MLP 元分数（每日 alpha feed）
       - regime_report  : trailing IC 时序 + 当前 status(healthy/decaying/dead) + 建议降权
       - book_equity    : walk-forward 回测权益（p8b 兼容格式，供 P9-7 直接消费）

regime 监控与 monitor_factor_decay.classify_overall_status 同构（status + 阈值 + 建议动作）：
  trailing IC (60d 滚动, 因果 shift(1)) > THR_HI(0.03) -> healthy
  THR_LO(0.0) < trailing <= THR_HI        -> decaying
  trailing <= THR_LO                      -> dead   -> 建议把 book gross 降至 EXPOSURE_FLOOR

用法（日常生产循环）：
  from production_soft_score import produce_book_score, regime_alert
  soft, rep, eq = produce_book_score(PANEL_CSV)
  alert = regime_alert(rep)   # -> {"alert": bool, "status": str, "recommended_gross_scale": float, ...}

可直接替换 P9-7 load_streams 的 book 源（见 p9_7_production_portfolio.py 中 load_book_soft_production）。
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from execution_rules import next_open_return_label
from run_backtest import load_prices, pivot_prices, sharpe_like, max_drawdown
from train_next_open_rank_model import (
    MIN_LIVE_SYMBOLS, build_features, calculate_walk_forward_metrics, clean_matrix,
    daily_ic, load_market_exposure, run_walk_forward,
)
from p10c_ensemble import (
    select_positive, build_linear_mlp_scores, MLP, MIN_LIVE_SYMBOLS as _M,
    TRAIN_DAYS, MTH, LIQ_LOOKBACK, THR as LIQ_THR, RETRAIN,
)
import pit_universe as pit
from wq_alpha_factors import build_wq_alpha_factors  # [并入] WQ101 因子吸收（2026-08-26）
from score_cache import load_or_build  # 内容寻址缓存守卫（根治换面板形状错）

HERE = Path(__file__).resolve().parent
PANEL = HERE / "external_data" / "daily-market-data-tdx" / "data_panel.csv"
P10E = HERE / "outputs" / "p10e_regime_gated"
SCORES_NPZ = P10E / "linear_mlp_scores.npz"
OUT = HERE / "outputs" / "production_soft_score"
OUT.mkdir(parents=True, exist_ok=True)

MAX_ABS = 0.22
COMMISSION_BPS = 3.0  # 万三佣金（A 股实盘标准，见 MOS 规则 13）
IMPACT_BPS = 0.7  # 市场冲击/滑点，独立于手续费的执行假设
STAMP_TAX_BPS = 5.0  # 法定印花税万分之五，仅卖出侧（见 MOS 规则 13）
IMPACT_REF = 0.01
TOP_N = 20                      # 仅用于 alpha 排序参考，实际持仓名数见下方 BOOK_* 容量配置
REBALANCE = 5  # Route C：降频甜区（freq=5）。原 1（日频）在真实成本模型下成本 47% 吃掉大部分毛 alpha；freq=5 砍 4× 换手、毛 alpha 基本不损，净 CAGR 8%→22%（见 模型提升_总收尾报告.md）
BOOK_AUM = 100_000_000.0

# 容量工程（P10h）推荐部署配置：在可接受夏普(>=0.8)与冲击(<0.5)下，容量最大的组合。
# top_n=40 拓宽广度（分散 + 聚合流动性），ADV=2% 放宽单票参与度上限（P10h 证明单票权重 w 几乎无效）。
BOOK_TOP_N = 40
BOOK_MAX_W = 0.04
BOOK_ADV_P = 0.02
# P10h 推荐配置容量拐点（top_n=40/ADV=2%），用于联合调度容量余量耦合
BOOK_KNEE_REF = 451_469_068.0
MIN_DEAD_TOP_N = 20
# regime 参数（因果 shift(1)；P10j walk-forward 寻优结论，已用干净净收益口径 + 阈值一致化复核）：
#  - THR_HI=0.03 为验证(OOS)窗口最优（val 1.562，高于 0.02 的 1.471 与 0.04 的 1.328）；0.04 及以上反而
#    伤 OOS（错配脏跑曾误报 0.04 更优，实为 soft 混合用 0.03、闸门用 0.04 的不一致所致）。P10e 原部署值被证实最优。
#  - FLOOR 0.40->0.25 在各 THR_HI 下单调有益（已是最低测试档），dead 期降权更彻底。
#  - THR_LO=0 为平衡档（0.01 时 val 略高但 dead 亏损恶化至 -0.033），故取 0。
#  - 自适应(滚动)阈值全面最差（val 0.804），不采用，保持固定阈值。
# 最终部署 THR_HI=0.03/THR_LO=0/FLOOR=0.25（与 P10e 一致）。
IC_WIN = 60
IC_MIN = 40
THR_HI = 0.03
THR_LO = 0.0
EXPOSURE_FLOOR = 0.25


def build_panel(panel_csv: Path, use_wq: bool = False):
    raw = load_prices(panel_csv, None, None)
    close = clean_matrix(pivot_prices(raw, "close"), MAX_ABS)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), MAX_ABS)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), MAX_ABS)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), MAX_ABS)
    amount = pivot_prices(raw, "amount").reindex_like(close)
    symbols = list(close.columns)
    features = build_features(close, open_px, high, low, amount)
    if use_wq:
        # [并入] WQ101 因子吸收（2026-08-26，非破坏式）：追加 19 个 PIT 安全 wq_* 因子。
        # 生产走 build_linear_mlp_scores 的 select_positive(IC>0) 自适应入选，故仅正向 wq 因子参与。
        # 注：宽松 walk-forward(feature_selection=None) 曾显示 +0.135 sharpe 假阳性；真实生产配置
        # (select_positive) 实测 WQ 净增量 -0.329 sharpe(0.410 vs 0.739)，故默认关闭(--wq opt-in)。
        # build_features 本身未改动，其余 ~40 个调用方不受影响。
        wq_feats = build_wq_alpha_factors(close, open_px, high, low, amount)
        features = {**features, **wq_feats}
    label = next_open_return_label(open_px, max_abs_daily_return=MAX_ABS)
    market_exposure = load_market_exposure(None, close.index, ma_window=120, risk_off_drawdown_20d=-0.08, below_ma_exposure=0.60, crash_exposure=0.0)
    roll_med = amount.rolling(LIQ_LOOKBACK, min_periods=LIQ_LOOKBACK).median()
    liquid_mask = (roll_med >= LIQ_THR).fillna(False)
    liquid_mask_aligned = liquid_mask.reindex(index=close.index, columns=close.columns).fillna(False)
    feat_arrays = {k: fr.values for k, fr in features.items()}
    label_arr = label.values
    return dict(close=close, open_px=open_px, high=high, low=low, amount=amount,
                symbols=symbols, features=features, label=label,
                market_exposure=market_exposure, liquid_mask_aligned=liquid_mask_aligned,
                feat_arrays=feat_arrays, label_arr=label_arr)


def causal_soft_blend(linear_score: pd.DataFrame, mlp_score: pd.DataFrame,
                      label: pd.DataFrame, symbols: list[str],
                      thr_hi: float = THR_HI, thr_lo: float = THR_LO):
    """因果 regime 连续混合：trailing 用 shift(2) 避免前视。返回 soft 分与诊断序列。

    标签口径: label[t] = open[t+2]/open[t+1]-1 (next_open_return_label, horizon_days=1),
    **t 行的标签在 t+2 开盘才成熟**。因此 as-of d (d 日收盘后) 只能使用 IC <= d-2：
      - IC[t] = corr(mlp_score[t], label[t]) 完整可得于 t+2 开盘后；
      - 原 shift(1) 使 trailing[d] 用到 IC[d-1](需 label[d-1]=open[d+1]/open[d], open[d+1] 明日
        开盘在 d 收盘时未知) = 1 天未来信息泄漏 (P0-3 残留修复, 2026-08-23, 与
        build_leakfree_nor_feed.py 的 shift(2) 口径一致)。
    注意: 该修复会改变 soft_score_feed.csv / book_soft_equity.csv / regime 产物——是
    消除前视的预期行为, 生产消费方(blend_candidate_pools / p9_7)直接获得更严格因果的分数。

    thr_hi/thr_lo 默认取模块常量；阈值寻优(P10j)时由调用方显式传入，确保 soft 混合与
    gross 闸门使用【同一个】THR_HI/THR_LO（否则会出现混合阈值与闸门阈值错配的脏结果）。
    """
    mlp_ic = daily_ic({"mlp": mlp_score}, label)["mlp"]
    trailing = mlp_ic.shift(2).rolling(IC_WIN, min_periods=IC_MIN).mean()
    alpha_dead = (trailing < thr_lo).fillna(False)
    adv = ((thr_hi - trailing) / (thr_hi - thr_lo)).clip(0, 1).fillna(1.0)  # 1=全线性(防御)
    soft = adv.values[:, None] * linear_score.values + (1 - adv.values[:, None]) * mlp_score.values
    soft = pd.DataFrame(soft, index=label.index, columns=symbols)
    return soft, trailing, adv, alpha_dead, mlp_ic


def joint_schedule(trailing: pd.Series, aum: float, knee_ref: float = BOOK_KNEE_REF,
                   base_top_n: int = BOOK_TOP_N, min_dead_top_n: int = MIN_DEAD_TOP_N,
                   thr_hi: float = THR_HI, thr_lo: float = THR_LO, floor: float = EXPOSURE_FLOOR):
    """regime × 容量 联合调度（P10i）。

    返回 (gross_scale, top_n_schedule, dead_top_n, headroom)：
      - gross_scale : 逐日 gross 暴露上限。dead->floor(0.40)；decaying 在 (floor,1) 插值；healthy->1。
      - top_n_schedule : 逐日持仓名数。dead 时按容量余量收窄到 [min_dead, base]；healthy 恢复 base。
      - 容量余量耦合：headroom = clip(1 - AUM/knee_ref, 0, 1)。容量充裕时 dead 可砍到 min_dead
        （冲击最敏感），容量紧张时不砍（避免爆容量）。
    部署时 market_exposure 取 min(risk_off, gross_scale)；top_n 传 top_n_schedule。
    """
    t = trailing.fillna(0.0)
    gross = np.where(t <= thr_lo, floor,
             np.where(t < thr_hi, floor + (1 - floor) * (t - thr_lo) / (thr_hi - thr_lo), 1.0))
    gross = pd.Series(gross, index=trailing.index)
    headroom = float(np.clip(1.0 - aum / knee_ref, 0.0, 1.0))
    dead_tn = int(round(base_top_n - (base_top_n - min_dead_top_n) * headroom))
    dead_tn = int(np.clip(dead_tn, min_dead_top_n, base_top_n))
    tn = np.where(t <= thr_lo, dead_tn,
          np.where(t < thr_hi, dead_tn + (base_top_n - dead_tn) * (t - thr_lo) / (thr_hi - thr_lo),
                   base_top_n))
    tn = pd.Series(np.round(tn).astype(int), index=trailing.index)
    return gross, tn, dead_tn, headroom


def assert_regime_thresholds_consistent(soft_thr_hi, soft_thr_lo, gate_thr_hi, gate_thr_lo,
                                        context: str = ""):
    """P10j 纪律守卫：soft 混合与 gross 闸门必须使用【同一个】THR_HI/THR_LO。

    causal_soft_blend 用 THR_HI/THR_LO 决定 soft 分里线性/MLP 的配比(adv 权重)；joint_schedule
    用 THR_HI/THR_LO 决定 gross 闸门。若两者不一致（如 soft 用 0.03、闸门用 0.04），回测会被悄悄
    污染并误导阈值寻优——P10j 曾因此误报 THR_HI=0.04 更优(val 1.677)，实则为错配假象
    (一致化后一致值 val 仅 1.328)。任何不一致立即 fail-fast，绝不静默放行。
    """
    if not (float(soft_thr_hi) == float(gate_thr_hi) and float(soft_thr_lo) == float(gate_thr_lo)):
        raise ValueError(
            f"[P10j 阈值一致性守卫] soft 混合与 gross 闸门阈值错配"
            f"{(' @ ' + context) if context else ''}: "
            f"soft=(hi={soft_thr_hi}, lo={soft_thr_lo}) gate=(hi={gate_thr_hi}, lo={gate_thr_lo})。"
            f"二者必须用同一个 THR_HI/THR_LO，否则回测结果不可信。"
        )


def regime_classify(trailing: pd.Series) -> pd.Series:
    def _s(v):
        if pd.isna(v):
            return "insufficient_history"
        if v > THR_HI:
            return "healthy"
        if v > THR_LO:
            return "decaying"
        return "dead"
    return trailing.apply(_s)


def regime_report(trailing: pd.Series, adv: pd.Series, alpha_dead: pd.Series,
                  mlp_ic: pd.Series, close_index) -> dict:
    last = trailing.index[-1]
    recent = trailing.iloc[-IC_WIN:]
    rec_dead = float(alpha_dead.iloc[-IC_WIN:].mean())
    cur_trailing = float(trailing.iloc[-1])
    cur_adv = float(adv.iloc[-1])
    cur_dead = bool(alpha_dead.iloc[-1])
    status = regime_classify(trailing).iloc[-1]
    # 建议 gross 缩放：全 regime 用衰减 gross 带（dead 自然落到 floor，健康→1.0）。
    # 注意 cur_adv 是线性/MLP 混合权重(1=全线性防御)，并非 gross 暴露带，绝不能当 gross scale 返回——
    # 否则健康日会把簿暴露压到 0（cur_adv≈0）的灾难性 bug。衰减带公式与 joint_schedule 保持一致。
    if pd.isna(cur_trailing):
        rec_scale = 1.0  # 历史不足：保守不降权
    else:
        rec_scale = float(EXPOSURE_FLOOR) + (1.0 - float(EXPOSURE_FLOOR)) * (cur_trailing - THR_LO) / (THR_HI - THR_LO)
        rec_scale = float(min(1.0, max(float(EXPOSURE_FLOOR), rec_scale)))
    # 仅保留最近 120 日时序，控制体积
    tail = trailing.iloc[-120:]
    series = {str(d.date()): (None if pd.isna(v) else round(float(v), 5)) for d, v in tail.items()}
    return {
        "asof": str(last.date()),
        "trailing_ic_recent_60d": round(float(recent.mean()), 5),
        "trailing_ic_current": round(cur_trailing, 5),
        "status": status,
        "alpha_dead_today": cur_dead,
        "recent_dead_pct_60d": round(rec_dead, 4),
        "adv_today": round(cur_adv, 5),
        "recommended_gross_scale": round(rec_scale, 4),
        "thresholds": {"thr_hi": THR_HI, "thr_lo": THR_LO, "exposure_floor": EXPOSURE_FLOOR},
        "alert": cur_dead,
        "trailing_ic_series_last120d": series,
    }


def regime_alert(rep: dict) -> dict:
    """与 monitor_factor_decay 同构的告警输出。"""
    return {
        "model": "p10e_soft_regime_gated_mlp",
        "asof": rep["asof"],
        "status": rep["status"],
        "alert": rep["alert"],
        "recommended_action": (
            f"de-risk book gross to {rep['recommended_gross_scale']:.2f}"
            if rep["alert"] else "maintain"
        ),
        "recommended_gross_scale": rep["recommended_gross_scale"],
        "trailing_ic_current": rep["trailing_ic_current"],
        "recent_dead_pct_60d": rep["recent_dead_pct_60d"],
    }


def read_regime_monitor(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_regime_alert(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def effective_book_gross_scale(alert: dict | None, base_scale: float = 1.0) -> float:
    """执行层闸门（regime 监控“生产化”的落点）。

    全 regime 消费 recommended_gross_scale（dead→floor，衰减→插值，健康→1.0）：alert 携带该字段时
    始终返回它，让衰减 gross 带在健康/衰减日也真正生效；此前 non-dead 直接返回 base_scale，丢弃了带。
    告警不是报告，而是可被下单/簿构建层消费的降权信号——P9-7 / prelive 下单前应先调用本函数得到当日
    生效 gross 上限。alert 为 None/缺字段时回退 base_scale（向后兼容）。
    """
    if not alert:
        return float(base_scale)
    rec = alert.get("recommended_gross_scale", None)
    if rec is None:
        return float(base_scale)
    return float(rec)


def latest_regime_alert(project_root: Path = HERE, asof_date: str | None = None) -> dict | None:
    """在已知输出目录里找最新 regime 告警（canonical 或流水线日期令牌化产物）。

    优先 canonical outputs/production_soft_score/regime_alert.json，其次按 mtime 扫描
    outputs/*/regime_alert_*.json 与 outputs/production_soft_score/regime_alert_*.json。
    返回最新一条告警 dict（含 asof/alert/recommended_gross_scale），找不到返回 None。
    """
    root = Path(project_root)
    candidates: list[Path] = [OUT / "regime_alert.json"]
    patterns = (
        "outputs/production_soft_score/regime_alert_*.json",
        "outputs/*/regime_alert_*.json",
        "outputs/regime_alert_*.json",
    )
    for pat in patterns:
        candidates.extend(sorted(root.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True))
    for c in candidates:
        if c and c.exists():
            try:
                return read_regime_alert(c)
            except (OSError, json.JSONDecodeError, ValueError):
                continue
    return None


def produce_book_score(panel_csv: Path = PANEL, aum: float = BOOK_AUM,
                       top_n: int = BOOK_TOP_N, max_w: float = BOOK_MAX_W,
                       adv_p: float = BOOK_ADV_P, schedule: str = "joint",
                       use_cache: bool = True, skip_backtest: bool = False,
                       universe: str = "pit", start_date: str | None = None,
                       refresh_live_daily: bool = False,
                       scores_npz: Path | None = None,
                       watchlist_mask_dir: str | None = None,
                       use_wq: bool = False
                       ) -> tuple[pd.DataFrame, dict, pd.DataFrame | None]:
    """生产入口：返回 (soft_score, regime_report, book_equity)。

    universe 控制回测+训练候选宇宙（根治"筛选后自选股"前视，对应 option 2：point-in-time 框架）：
      - "pit"           : 默认。每个再平衡日 t 只用 ≤t 信息算资格（pit_universe.pit_eligible），
                          杜绝用未来名单/未来退市状态；训练与回测宇宙一致。
      - "full"          : 全面板（旧行为，含当前快照全部标的，带轻度幸存者偏差）。
      - "static-survivor": 诊断用反例——把末日存在的标的广播到所有历史日（=用未来名单回测），
                          用于量化"人工精选子集"会虚高多少。

    top_n / max_w / adv_p 为容量工程（P10h）可调参数，默认采用推荐部署配置
    (top_n=40, w=0.04, ADV=2%)：该配置在 1e8 簿 AUM 下夏普反而高于基线(20名/无ADV上限)，
    且容量拐点从 ~1e8 放大到 ~4.5e8（部署率>=0.6 的 AUM）。

    schedule 控制 regime × 容量 联合调度（P10i），决定回测里 market_exposure 与 top_n 是否逐日随
    regime 调整：
      - "static"      : 固定 top_n，market_exposure=风险偏好(risk_off) 仅（当前生产旧行为，dead 不降权）。
      - "regime_gross": 仅把 regime 的 gross 降权(0.40)写进回测，top_n 固定。
      - "joint"       : regime->gross 降权 + top_n 按容量余量动态下调（dead 收窄到 20~40）。默认。
    """
    P = build_panel(panel_csv, use_wq=use_wq)

    # --- 候选宇宙分支（根治"筛选后自选股"前视，对应 option 2：point-in-time 框架） ---
    # PIT 掩码在每个再平衡日 t 只用 ≤t 信息判合格（上市/未退市/未停牌/trailing 流动性/最小历史），
    # 杜绝用"未来名单/未来退市状态"回测；训练与回测共用同一掩码，避免训练被未来信息污染。
    if universe == "pit":
        universe_mask = pit.pit_eligible(
            P["close"], P["amount"], None,
            thr=LIQ_THR, lookback=LIQ_LOOKBACK, min_days=60)
        score_mask = universe_mask
        backtest_mask = universe_mask
        rep_universe = "pit(point-in-time)"
    elif universe == "static-survivor":
        universe_mask = pit.static_survivor_mask(P["close"])
        score_mask = universe_mask          # 反例：训练+回测都用末日名单（=未来信息）
        backtest_mask = universe_mask
        rep_universe = "static-survivor(反例/前视污染)"
    elif universe == "watchlist":
        # 用户"每日自选股名单"的诚实 PIT 宇宙：票在日 t 合格当且仅当
        # 出现在 ≤t 的名单里（watchlist_leak_audit.py 产出）。训练+回测共用，
        # 既享受用户的真实筛选，又从根上排除名单自身的前视。
        from watchlist_leak_audit import load_watchlist_mask
        wl = load_watchlist_mask(P["close"], out_dir=watchlist_mask_dir)
        score_mask = wl
        backtest_mask = wl
        rep_universe = "watchlist(用户自选股·诚实PIT)"
    else:  # full — 旧行为
        score_mask = P["liquid_mask_aligned"]
        backtest_mask = None
        rep_universe = "full(旧行为/轻度幸存者偏差)"

    # 内容寻址缓存守卫：npz 内嵌面板指纹，面板变更（如 880→889 行）自动失效重建，
    # 根治 "Shape of passed values is (880,5910), indices imply (889,5910)" 形状错。
    # 思想来自 model4 的 pipeline_cache.py（按 base_panel+基准+每日文件+代码/配置哈希寻址）。
    scores_npz = Path(scores_npz) if scores_npz is not None else P10E / f"linear_mlp_scores_{universe}.npz"
    def _build_scores():
        return build_linear_mlp_scores(
            P["features"], P["label"], P["symbols"], P["feat_arrays"], P["label_arr"], score_mask,
            refresh_live_daily=refresh_live_daily)
    linear_score, mlp_score = load_or_build(
        scores_npz, P["label"].index, P["symbols"], _build_scores,
        panel_csv=panel_csv, use_cache=use_cache)

    soft_thr_hi, soft_thr_lo = THR_HI, THR_LO
    gate_thr_hi, gate_thr_lo = THR_HI, THR_LO
    soft, trailing, adv, alpha_dead, mlp_ic = causal_soft_blend(
        linear_score, mlp_score, P["label"], P["symbols"], thr_hi=soft_thr_hi, thr_lo=soft_thr_lo)
    rep = regime_report(trailing, adv, alpha_dead, mlp_ic, P["label"].index)
    rep["universe"] = rep_universe

    # 联合调度（P10i）：构造逐日 market_exposure 与 top_n_schedule
    mkt_exp = P["market_exposure"]
    top_n_sched = None
    sched_info = {"mode": schedule}
    if schedule in ("regime_gross", "joint"):
        # P10j 阈值一致性守卫：soft 混合与 gross 闸门必须用同一 THR_HI/THR_LO（fail-fast）
        gross_scale, tn_sched, dead_tn, headroom = joint_schedule(
            trailing, aum, thr_hi=gate_thr_hi, thr_lo=gate_thr_lo, floor=EXPOSURE_FLOOR)
        assert_regime_thresholds_consistent(
            soft_thr_hi, soft_thr_lo, gate_thr_hi, gate_thr_lo, context="produce_book_score")
        joint_mkt = pd.Series(
            np.minimum(mkt_exp.reindex(gross_scale.index).fillna(1.0).values, gross_scale.values),
            index=gross_scale.index)
        mkt_exp = joint_mkt
        sched_info.update({"dead_top_n": dead_tn, "headroom": round(headroom, 4),
                           "avg_top_n": round(float(tn_sched.mean()), 1),
                           "top_n_min": int(tn_sched.min()), "top_n_max": int(tn_sched.max())})
        if schedule == "joint":
            top_n_sched = tn_sched
    rep["schedule"] = sched_info

    # walk-forward 回测得到 book 权益（p8b 兼容格式）
    # skip_backtest=True 时仅产出 regime 监控/告警（流水线每日闸门用），跳过昂贵回测
    if skip_backtest:
        rep["backtest"] = None
        eq = None
    else:
        ic_run = daily_ic({"soft": soft}, P["label"])
        eq, _, _ = run_walk_forward(
            close=P["close"], open_px=P["open_px"], features={"soft": soft}, label=P["label"], ic=ic_run,
            train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=top_n,
            rebalance_frequency=REBALANCE, max_position_weight=max_w, leverage=1.0,
            commission_bps=COMMISSION_BPS, impact_bps=IMPACT_BPS, stamp_tax_bps=STAMP_TAX_BPS,
            impact_model="sqrt", impact_ref_participation=IMPACT_REF,
            max_buy_open_gap=0.06, limit_buffer=0.995, market_exposure=mkt_exp,
            initial_capital=aum, max_training_horizon=MTH, feature_directions=None,
            amount=P["amount"], feature_selection=None, max_daily_amount_participation=adv_p,
            liquid_mask=backtest_mask, top_n_schedule=top_n_sched,
        )
        m = calculate_walk_forward_metrics(eq, aum)
        rep["backtest_full"] = {"total_return": round(float(m.get("total_return")), 4),
                                "sharpe_like": round(float(m.get("sharpe_like")), 4),
                                "max_drawdown": round(float(m.get("max_drawdown")), 4),
                                "aum": aum, "trade_days": int(m.get("trade_days", 0))}
        # 公平 A/B：当指定 start_date 时，只报告该窗口内的表现（训练仍用全历史，
        # 仅交易/指标切片到窗口），使不同宇宙在[窗口, 末日]可比。
        if start_date is not None:
            rep["backtest"] = _windowed_metrics(eq, start_date, aum)
            rep["window"] = start_date
        else:
            rep["backtest"] = rep["backtest_full"]
    return soft, rep, eq


def _windowed_metrics(eq: pd.DataFrame, start_date: str, aum: float) -> dict:
    """对 walk-forward 权益曲线按 start_date 切片，计算窗口内总收益/Sharpe/回撤。

    训练仍用全历史（特征/再训练不受影响），仅把"报告口径"限制到 [start_date, 末日]，
    从而让不同候选宇宙在同一交易窗口上可比（避免全市场宇宙用 3.5 年、自选宇宙只用 1 个月）。
    """
    e = eq.copy()
    dts = pd.to_datetime(e["date"])
    e = e[dts >= pd.to_datetime(start_date)].reset_index(drop=True)
    if e.empty:
        return {"total_return": 0.0, "sharpe_like": 0.0, "max_drawdown": 0.0,
                "aum": aum, "trade_days": 0, "window_start": start_date}
    nav = pd.to_numeric(e["equity"], errors="coerce")
    net = (pd.to_numeric(e["gross_return"], errors="coerce")
           - pd.to_numeric(e["cost"], errors="coerce"))
    start_eq = float(nav.iloc[0])
    total = float(nav.iloc[-1] / start_eq - 1.0)
    return {"total_return": round(total, 4),
            "sharpe_like": round(sharpe_like(net), 4),
            "max_drawdown": round(max_drawdown(nav), 4),
            "aum": aum, "trade_days": int(len(nav)), "window_start": start_date}


def _regime_markdown(rep: dict, alert: dict) -> str:
    sched = rep.get("schedule") or {}
    back = rep.get("backtest") or {}
    lines = [
        f"# Regime Monitor {rep.get('asof')}",
        "",
        f"- status: **{rep.get('status')}**  alert: **{alert.get('alert')}**",
        f"- trailing IC (current): {rep.get('trailing_ic_current')}  (recent 60d: {rep.get('trailing_ic_recent_60d')})",
        f"- alpha_dead_today: {rep.get('alpha_dead_today')}  recent_dead%_60d: {rep.get('recent_dead_pct_60d')}",
        f"- adv_today: {rep.get('adv_today')}  recommended_gross_scale: {rep.get('recommended_gross_scale')}",
        f"- thresholds: hi={rep.get('thresholds', {}).get('thr_hi')} lo={rep.get('thresholds', {}).get('thr_lo')} "
        f"floor={rep.get('thresholds', {}).get('exposure_floor')}",
        f"- schedule(mode={sched.get('mode')}): dead_top_n={sched.get('dead_top_n')} "
        f"avg_top_n={sched.get('avg_top_n')} headroom={sched.get('headroom')}",
        f"- recommended_action: **{alert.get('recommended_action')}**",
    ]
    if back:
        w = rep.get("window")
        if w:
            lines.append(
                f"- backtest(**窗口 {w}→末日**): total={back.get('total_return'):+.4f} "
                f"sharpe={back.get('sharpe_like'):.3f} dd={back.get('max_drawdown'):.3f} "
                f"days={back.get('trade_days')}"
            )
            bf = rep.get("backtest_full") or {}
            lines.append(
                f"  - 全程对照(2022→末日): total={bf.get('total_return'):+.4f} "
                f"sharpe={bf.get('sharpe_like'):.3f} dd={bf.get('max_drawdown'):.3f} "
                f"days={bf.get('trade_days')}"
            )
        else:
            lines.append(
                f"- backtest: total={back.get('total_return'):+.4f} sharpe={back.get('sharpe_like'):.3f} "
                f"dd={back.get('max_drawdown'):.3f} days={back.get('trade_days')}"
            )
    else:
        lines.append("- backtest: skipped (monitor mode)")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="生产化 p10e_soft regime 门控 MLP 打分 + regime 监控")
    parser.add_argument("--panel", default=str(PANEL), help="面板 CSV/parquet 路径")
    parser.add_argument("--asof-date", default=None, help="命名/陈旧性自检用的业务日期")
    parser.add_argument("--output-dir", default=str(OUT), help="监控产物输出目录（流水线用 outputs/high_return_v2）")
    parser.add_argument("--token", default=None, help="日期令牌（缺省由 --asof-date 推导）")
    parser.add_argument("--schedule", default="joint", choices=("joint", "regime_gross", "static"))
    parser.add_argument("--universe", default="pit",
                        choices=("pit", "full", "static-survivor", "watchlist"),
                        help="候选宇宙：pit=默认 point-in-time(根治自选股前视)；"
                             "full=旧行为(轻度幸存者偏差)；static-survivor=前视反例诊断；"
                             "watchlist=用户每日自选股名单的诚实PIT宇宙(需先跑 watchlist_leak_audit.py)")
    parser.add_argument("--start-date", default=None,
                        help="公平 A/B 窗口起点：指标只报告 [start-date, 末日] 区间"
                             "(训练仍用全历史)，使不同宇宙在同一交易窗口可比")
    parser.add_argument("--skip-backtest", action="store_true", help="跳过 walk-forward 回测（仅 regime 监控/告警）")
    parser.add_argument("--monitor", action="store_true",
                        help="监控模式：仅写日期令牌化 regime_monitor_/regime_alert_ 产物到 --output-dir")
    parser.add_argument("--block-on-alert", action="store_true",
                        help="ALERT=True 时以退出码 2 阻断（用于流水线硬闸门）")
    parser.add_argument("--refresh-live-daily", action="store_true",
                        help="实验变体：非 retrain 日按当日流动性刷新 live 集（解决覆盖错位；默认关=生产行为不变）")
    parser.add_argument("--scores-npz", default=None,
                        help="分数缓存路径覆盖（实验变体用独立缓存，避免与默认宇宙 npz 串味）")
    parser.add_argument("--wq", action="store_true",
                        help="启用 WQ101 因子并入（A/B 实验：在 base 因子之上叠加 19 个 wq_* 因子；默认关闭，验证显示其净增量为负）")
    parser.add_argument("--watchlist-mask-dir", default=None,
                        help="watchlist 宇宙掩码目录覆盖（默认 outputs/watchlist_audit；聚焦实验用独立掩码）")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    asof = args.asof_date
    token = args.token or (asof.replace("-", "") if asof else None)

    soft, rep, eq = produce_book_score(
        Path(args.panel), BOOK_AUM, schedule=args.schedule, universe=args.universe,
        use_cache=True, skip_backtest=args.skip_backtest, start_date=args.start_date,
        refresh_live_daily=args.refresh_live_daily,
        scores_npz=(Path(args.scores_npz) if args.scores_npz
                    else (P10E / f"linear_mlp_scores_{args.universe}_wq.npz") if args.wq
                    else None),
        watchlist_mask_dir=args.watchlist_mask_dir,
        use_wq=args.wq)
    alert = regime_alert(rep)

    if args.monitor:
        if not token:
            raise SystemExit("--monitor 需要 --token 或 --asof-date 以命名产物")
        mon_path = out_dir / f"regime_monitor_{token}.json"
        alert_path = out_dir / f"regime_alert_{token}.json"
        md_path = out_dir / f"regime_monitor_{token}.md"
        mon_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        alert_path.write_text(json.dumps(alert, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(_regime_markdown(rep, alert), encoding="utf-8")
        # 同步刷新 canonical 最新告警（供 P9-7 / prelive 闸门即时消费）
        (OUT / "regime_monitor.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        (OUT / "regime_alert.json").write_text(json.dumps(alert, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[monitor] asof={rep['asof']} status={rep['status']} alert={alert['alert']} "
              f"action={alert['recommended_action']} -> {mon_path.name}/{alert_path.name}")
        if alert["alert"] and args.block_on_alert:
            print("[monitor] ALERT=True 且 --block-on-alert：阻断退出（code=2）。")
            return 2
        return 0

    # 默认完整模式：写产物到 out_dir（= --output-dir；缺省即规范 OUT，供 P9-7 等消费）。
    # 修复：此前写死模块常量 OUT，使 --output-dir 在非 monitor 模式被静默忽略。
    print("=== P10g-production: 生产化 p10e_soft 打分 + regime 监控 ===")
    soft.to_csv(out_dir / "soft_score_feed.csv", encoding="utf-8")
    latest = soft.iloc[[-1]].copy()
    latest.to_csv(out_dir / "latest_soft_score.csv", encoding="utf-8")
    if eq is not None:
        eq.to_csv(out_dir / "book_soft_equity.csv", index=False, encoding="utf-8")
    (out_dir / "regime_monitor.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "regime_alert.json").write_text(json.dumps(alert, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  asof={rep['asof']}  status={rep['status']}  trailing_ic_now={rep['trailing_ic_current']:.4f}")
    print(f"  recent_dead%_60d={rep['recent_dead_pct_60d']:.3f}  adv_today={rep['adv_today']:.3f}  "
          f"recommended_gross_scale={rep['recommended_gross_scale']:.3f}")
    bt = rep.get("backtest")
    if bt:
        print(f"  backtest: total={bt['total_return']:+.4f} sharpe={bt['sharpe_like']:.3f} "
              f"dd={bt['max_drawdown']:.3f}")
    else:
        print("  backtest: skipped (skip-backtest / monitor 模式)")
    print(f"  ALERT={alert['alert']}  action={alert['recommended_action']}")
    print(f"  产物: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
