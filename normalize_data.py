"""Normalize heterogeneous strategy output files to a Backtest-ready panel.

Expected output columns:
  - date
  - symbol
  - open
  - high
  - low
  - close
  - volume
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pyyaml. Install with `pip install pyyaml`."
    ) from exc


FIELD_GUESSES = {
    "date": [
        "date",
        "trade_date",
        "datetime",
        "trade_datetime",
        "ts",
        "timestamp",
        "as_of_date",
        "as_of_datetime",
        "event_time",
        "event_datetime",
    ],
    "symbol": [
        "symbol",
        "code",
        "ts_code",
        "ticker",
        "stock_code",
        "instrument_code",
    ],
    "close": [
        "close",
        "close_price",
        "adj_close",
        "adjclose",
        "px_last",
        "last",
        "price",
    ],
    "open": ["open", "open_price", "px_open", "open_px"],
    "high": ["high", "high_price", "px_high", "high_px"],
    "low": ["low", "low_price", "px_low", "low_px"],
    "volume": ["volume", "vol", "trade_volume", "turnover_volume"],
}
WIDE_FIELDS = ["open", "high", "low", "close", "volume"]


def detect_column(columns: pd.Index, candidates: list[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def load_mapping(raw_mapping: Optional[str]) -> Dict[str, str]:
    if not raw_mapping:
        return {}
    text = str(raw_mapping).strip()
    if not text:
        return {}

    # support inline json/yaml mapping text, e.g. `--mapping "{...}"`
    if text.startswith("{") and text.endswith("}"):
        parsed = yaml.safe_load(text)
        if isinstance(parsed, dict):
            return {k: str(v) for k, v in parsed.items() if isinstance(v, str)}
        raise ValueError("内联 --mapping 不是合法 YAML/JSON 对象。")

    # fallback: treat as path
    path = Path(text)
    if not path.exists():
        raise FileNotFoundError(f"mapping 文件不存在: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text.strip():
        return {}
    if path.suffix.lower() in {".yaml", ".yml"}:
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("mapping 文件必须是 YAML/JSON 对象。")
    return {k: str(v) for k, v in raw.items() if isinstance(v, str)}


def infer_or_get(mapping: Dict[str, str], key: str, columns: pd.Index) -> str:
    if key in mapping and mapping[key] in columns:
        return mapping[key]
    guess = detect_column(columns, FIELD_GUESSES[key])
    if not guess:
        raise ValueError(
            f"未能识别字段 `{key}`，当前表头：{list(columns)}。"
            f"可用 --mapping 显式指定 {key} 对应字段。"
        )
    return guess


def pick_field(mapping: Dict[str, str], key: str, columns: pd.Index, required: bool = False) -> Optional[str]:
    if key in mapping and mapping[key] in columns:
        return mapping[key]
    guess = detect_column(columns, FIELD_GUESSES[key])
    if guess is not None:
        return guess
    return None if not required else (_ for _ in ()).throw(
        ValueError(f"未识别可选字段 `{key}`，当前表头：{list(columns)}。")
    )


def normalize_narrow(
    input_path: Path,
    output_path: Path,
    mapping: Dict[str, str],
) -> pd.DataFrame:
    df = pd.read_csv(input_path, dtype=str)
    if df.empty:
        raise ValueError("输入 CSV 为空。")

    cols = pd.Index(df.columns)
    date_col = infer_or_get(mapping, "date", cols)
    symbol_col = infer_or_get(mapping, "symbol", cols)
    close_col = infer_or_get(mapping, "close", cols)

    open_col = pick_field(mapping, "open", cols)
    high_col = pick_field(mapping, "high", cols)
    low_col = pick_field(mapping, "low", cols)
    volume_col = pick_field(mapping, "volume", cols)

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["symbol"] = df[symbol_col].astype(str).str.strip()
    out["close"] = pd.to_numeric(df[close_col], errors="coerce")

    if open_col is not None:
        out["open"] = pd.to_numeric(df[open_col], errors="coerce")
    if high_col is not None:
        out["high"] = pd.to_numeric(df[high_col], errors="coerce")
    if low_col is not None:
        out["low"] = pd.to_numeric(df[low_col], errors="coerce")
    if volume_col is not None:
        out["volume"] = pd.to_numeric(df[volume_col], errors="coerce")

    if "open" not in out:
        out["open"] = out["close"]
    else:
        out["open"] = out["open"].fillna(out["close"])
    if "high" not in out:
        out["high"] = out["close"]
    else:
        out["high"] = out["high"].fillna(out["close"])
    if "low" not in out:
        out["low"] = out["close"]
    else:
        out["low"] = out["low"].fillna(out["close"])
    if "volume" not in out:
        out["volume"] = pd.NA

    out = out.dropna(subset=["date", "symbol", "close"]).copy()
    if out.empty:
        raise ValueError(
            "标准化后无有效行。请确认 date/symbol/close 列存在并且可解析。"
        )

    out = out.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last")
    out["date"] = out["date"].dt.date.astype(str)
    out = out[["date", "symbol", "open", "high", "low", "close", "volume"]]
    return out


def parse_wide_field(col: str) -> Optional[tuple[str, str]]:
    col = col.strip()
    # close_000001 / close-000001 / 000001_close / 000001.close
    for field in WIDE_FIELDS:
        m = re.match(rf"^{re.escape(field)}[\._-]+(.+)$", col, flags=re.IGNORECASE)
        if m:
            symbol = m.group(1).strip()
            if symbol:
                return field.lower(), symbol
    for field in WIDE_FIELDS:
        m = re.match(rf"^(.+)[\._-]+{re.escape(field)}$", col, flags=re.IGNORECASE)
        if m:
            symbol = m.group(1).strip()
            if symbol:
                return field.lower(), symbol
    return None


def normalize_wide(
    input_path: Path,
    output_path: Path,
    mapping: Dict[str, str],
    value_field: str = "close",
) -> pd.DataFrame:
    raw = pd.read_csv(input_path, dtype=str)
    if raw.empty:
        raise ValueError("输入 CSV 为空。")

    cols = pd.Index(raw.columns)
    date_col = infer_or_get(mapping, "date", cols)

    date_df = pd.to_datetime(raw[date_col], errors="coerce")
    value_records: list[dict[str, Any]] = []

    for c in cols:
        if c == date_col:
            continue

        parsed = parse_wide_field(c)
        if parsed is None:
            # fallback: unlabelled wide panel column, try assign as value_field
            field = value_field.lower()
            symbol = c.strip()
            if not symbol:
                continue
        else:
            field, symbol = parsed

        field = field.lower()
        if field not in {"open", "high", "low", "close", "volume"}:
            continue

        values = pd.to_numeric(raw[c], errors="coerce")
        for dt, v in zip(date_df, values):
            if pd.isna(dt) or pd.isna(v):
                continue
            value_records.append(
                {
                    "date": dt,
                    "symbol": symbol,
                    "field": field,
                    "value": float(v),
                }
            )

    if not value_records:
        raise ValueError(
            "宽表标准化失败：未能从列名解析出 symbol/field 组合。"
            "建议改用 --wide 显式模式并提供 symbol-like 列名，或使用 narrow 映射文件。"
        )

    wide_df = pd.DataFrame(value_records)
    panel = (
        wide_df.pivot_table(
            index=["date", "symbol"], columns="field", values="value", aggfunc="last"
        )
        .reset_index()
    )
    panel.columns = [c if isinstance(c, str) else c for c in panel.columns.get_level_values(0)]
    panel = panel.rename_axis(None, axis=1)

    if "close" not in panel.columns:
        raise ValueError("宽表中缺少 close 数据（无法构建回测面板）。")

    if "open" not in panel.columns:
        panel["open"] = panel["close"]
    if "high" not in panel.columns:
        panel["high"] = panel["close"]
    if "low" not in panel.columns:
        panel["low"] = panel["close"]
    if "volume" not in panel.columns:
        panel["volume"] = pd.NA

    out = panel[["date", "symbol", "open", "high", "low", "close", "volume"]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out = out.dropna(subset=["date", "symbol", "close"]).sort_values(["date", "symbol"]).reset_index(
        drop=True
    )
    return out


def normalize(
    input_path: Path,
    output_path: Path,
    mapping: Dict[str, str],
    wide_mode: bool = False,
    wide_value_field: str = "close",
) -> pd.DataFrame:
    # 先尝试窄表路径；如果不满足再按宽表处理
    df = pd.read_csv(input_path, dtype=str)
    if df.empty:
        raise ValueError("输入 CSV 为空。")
    cols = pd.Index(df.columns)

    date_col = infer_or_get(mapping, "date", cols)
    symbol_col = pick_field(mapping, "symbol", cols)
    close_col = pick_field(mapping, "close", cols)

    if wide_mode or symbol_col is None or close_col is None:
        out = normalize_wide(input_path, output_path, mapping, value_field=wide_value_field)
    else:
        # reuse narrow path
        out = normalize_narrow(input_path, output_path, mapping)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize mixed-format CSV data to backtest-friendly OHLCV panel."
    )
    parser.add_argument("--input", required=True, help="输入 CSV 路径。")
    parser.add_argument("--output", required=True, help="输出 data_panel.csv 路径。")
    parser.add_argument(
        "--mapping",
        default=None,
        help="可选，字段映射文件（YAML/JSON），示例: {\"date\":\"price_date\",\"symbol\":\"code\",\"close\":\"close_price\"}。",
    )
    parser.add_argument(
        "--wide",
        action="store_true",
        help="启用宽表模式：将非 date 列按 symbol/field 解析为面板形式。",
    )
    parser.add_argument(
        "--wide-value-field",
        default="close",
        help="宽表列未显式编码 field 时，默认作为该字段处理（默认: close）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = load_mapping(args.mapping) if args.mapping else {}
    out = normalize(
        Path(args.input),
        Path(args.output),
        mapping,
        wide_mode=args.wide,
        wide_value_field=args.wide_value_field,
    )
    print(
        "normalize done\n"
        f"- input: {args.input}\n"
        f"- output: {args.output}\n"
        f"- rows: {len(out)}\n"
        f"- symbols: {out['symbol'].nunique()}"
    )


if __name__ == "__main__":
    main()
