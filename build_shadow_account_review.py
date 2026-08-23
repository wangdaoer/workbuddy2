"""Derive research-only shadow account rules from personal trade review outputs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


BUCKET_SPECS = [
    ("by_holding_bucket.csv", "holding_bucket", "holding"),
    ("by_entry_position_bucket.csv", "entry_position_bucket", "entry_position"),
    ("by_entry_momentum_bucket.csv", "entry_momentum_bucket", "entry_momentum"),
    ("by_buy_time_bucket.csv", "buy_time_bucket", "buy_time"),
    ("by_sell_time_bucket.csv", "sell_time_bucket", "sell_time"),
]


def derive_shadow_account_review(
    review_dir: Path,
    min_trades: int = 8,
    generated_at: str | None = None,
) -> dict[str, Any]:
    review_dir = Path(review_dir)
    summary = _read_json(review_dir / "summary.json")
    rules: list[dict[str, Any]] = []
    for filename, value_column, rule_group in BUCKET_SPECS:
        rows = _read_csv_rows(review_dir / filename)
        rules.extend(_bucket_rules(rows, value_column, rule_group, min_trades=min_trades))
    symbol_rows = _read_csv_rows(review_dir / "by_symbol.csv")
    rules.extend(_symbol_rules(symbol_rows, min_trades=min_trades))
    rules = _dedupe_rules(rules)
    rules.sort(key=lambda item: (item["priority"], -abs(item["pnl"]), item["source"]))
    return {
        "schema_version": 1,
        "generated_at": generated_at or datetime.now().isoformat(timespec="seconds"),
        "review_dir": str(review_dir),
        "research_only": True,
        "allows_broker_orders": False,
        "scope": "personal_trade_shadow_account",
        "summary": summary,
        "rules": rules[:12],
        "counterfactual": _counterfactual(review_dir, rules),
        "warnings": _warnings(review_dir, summary, rules),
    }


def write_shadow_account_review(
    review_dir: Path,
    output_dir: Path | None = None,
    min_trades: int = 8,
    generated_at: str | None = None,
) -> dict[str, Path]:
    review_dir = Path(review_dir)
    output_dir = Path(output_dir) if output_dir else review_dir / "shadow_account"
    output_dir.mkdir(parents=True, exist_ok=True)
    review = derive_shadow_account_review(review_dir, min_trades=min_trades, generated_at=generated_at)
    json_path = output_dir / "shadow_account_review.json"
    rules_csv = output_dir / "shadow_account_rules.csv"
    markdown_path = output_dir / "shadow_account_review.md"
    json_path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_rules_csv(rules_csv, review["rules"])
    markdown_path.write_text(_markdown(review), encoding="utf-8")
    return {"json": json_path, "rules_csv": rules_csv, "markdown": markdown_path}


def _bucket_rules(
    rows: list[dict[str, str]],
    value_column: str,
    rule_group: str,
    min_trades: int,
) -> list[dict[str, Any]]:
    eligible = [_normalized_row(row) for row in rows if _float(row.get("trades")) >= min_trades]
    if not eligible:
        return []
    rules: list[dict[str, Any]] = []
    positive = [row for row in eligible if row["pnl"] > 0 and row["win_rate"] >= 0.5]
    negative = [row for row in eligible if row["pnl"] < 0 and row["win_rate"] <= 0.45]
    if positive:
        row = max(positive, key=lambda item: (item["pnl"], item["win_rate"], item["trades"]))
        rules.append(
            _rule(
                rule_id=f"prefer_{rule_group}_{row['value']}",
                action="prefer",
                source=rule_group,
                value=row["value"],
                priority=20,
                row=row,
                description=f"Prefer {rule_group}={row['value']} when the model signal also agrees.",
            )
        )
    if negative:
        row = min(negative, key=lambda item: (item["pnl"], -item["trades"]))
        rules.append(
            _rule(
                rule_id=f"avoid_{rule_group}_{row['value']}",
                action="avoid_or_reduce",
                source=rule_group,
                value=row["value"],
                priority=10,
                row=row,
                description=f"Avoid or reduce {rule_group}={row['value']} until new evidence improves.",
            )
        )
    return rules


def _symbol_rules(rows: list[dict[str, str]], min_trades: int) -> list[dict[str, Any]]:
    eligible = [_normalized_row(row, default_value=row.get("symbol", "")) for row in rows if _float(row.get("trades")) >= min_trades]
    if not eligible:
        return []
    rules: list[dict[str, Any]] = []
    winners = [row for row in eligible if row["pnl"] > 0 and row["win_rate"] >= 0.55]
    losers = [row for row in eligible if row["pnl"] < 0 and row["win_rate"] <= 0.45]
    if winners:
        row = max(winners, key=lambda item: (item["pnl"], item["win_rate"], item["trades"]))
        label = _symbol_label(row)
        rules.append(
            _rule(
                rule_id=f"prefer_symbol_{row['symbol']}",
                action="prefer_small_bonus",
                source="symbol_history",
                value=row["symbol"],
                priority=40,
                row=row,
                description=f"{label} has positive personal execution evidence; allow only a small confirmation bonus.",
            )
        )
    if losers:
        row = min(losers, key=lambda item: (item["pnl"], -item["trades"]))
        label = _symbol_label(row)
        rules.append(
            _rule(
                rule_id=f"avoid_symbol_{row['symbol']}",
                action="avoid_or_watch",
                source="symbol_history",
                value=row["symbol"],
                priority=5,
                row=row,
                description=f"{label} has severe personal loss evidence; keep on watch unless fresh model evidence is strong.",
            )
        )
    return rules


def _rule(
    rule_id: str,
    action: str,
    source: str,
    value: str,
    priority: int,
    row: dict[str, Any],
    description: str,
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "action": action,
        "source": source,
        "value": value,
        "priority": priority,
        "trades": int(row["trades"]),
        "pnl": float(row["pnl"]),
        "win_rate": float(row["win_rate"]),
        "avg_return": float(row["avg_ret"]),
        "description": description,
    }


def _counterfactual(review_dir: Path, rules: list[dict[str, Any]]) -> dict[str, Any]:
    round_trips = [_normalized_row(row, default_value=row.get("symbol", "")) for row in _read_csv_rows(review_dir / "round_trips.csv")]
    losing = [row for row in round_trips if row["pnl"] < 0]
    avoid_rules = [rule for rule in rules if rule["action"].startswith("avoid")]
    avoid_pnl = sum(rule["pnl"] for rule in avoid_rules)
    giveback_rows = [row for row in round_trips if row.get("giveback_from_mfe_pct", 0.0) >= 0.05]
    return {
        "note": "counterfactual uses completed personal trades only; it is diagnostic, not a trading instruction",
        "losing_round_trips": len(losing),
        "losing_pnl": round(sum(row["pnl"] for row in losing), 4),
        "avoid_rule_count": len(avoid_rules),
        "avoid_rule_historical_pnl": round(avoid_pnl, 4),
        "large_giveback_round_trips": len(giveback_rows),
    }


def _warnings(review_dir: Path, summary: dict[str, Any], rules: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    if not summary:
        warnings.append("summary_missing")
    if not rules:
        warnings.append("no_rules_met_minimum_support")
    for filename, _, _ in BUCKET_SPECS:
        if not (review_dir / filename).exists():
            warnings.append(f"missing:{filename}")
    return warnings


def _normalized_row(row: dict[str, str], default_value: str | None = None) -> dict[str, Any]:
    value = default_value
    if value is None:
        for key in row:
            if key not in {"trades", "pnl", "win_rate", "avg_ret", "median_ret", "avg_holding_days", "avg_mfe", "avg_mae"}:
                value = row.get(key, "")
                break
    return {
        "value": str(value or ""),
        "symbol": str(row.get("symbol", "")),
        "name": str(row.get("name", "")),
        "trades": _float(row.get("trades")),
        "pnl": _float(row.get("pnl")),
        "win_rate": _float(row.get("win_rate")),
        "avg_ret": _float(row.get("avg_ret") or row.get("return_pct")),
        "giveback_from_mfe_pct": _float(row.get("giveback_from_mfe_pct")),
    }


def _symbol_label(row: dict[str, Any]) -> str:
    name = row.get("name") or ""
    symbol = row.get("symbol") or row.get("value")
    return f"{symbol} {name}".strip()


def _float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _dedupe_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for rule in rules:
        if rule["rule_id"] in seen:
            continue
        seen.add(rule["rule_id"])
        out.append(rule)
    return out


def _write_rules_csv(path: Path, rules: list[dict[str, Any]]) -> None:
    columns = ["rule_id", "action", "source", "value", "priority", "trades", "pnl", "win_rate", "avg_return", "description"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for rule in rules:
            writer.writerow({column: rule.get(column, "") for column in columns})


def _markdown(review: dict[str, Any]) -> str:
    summary = review.get("summary") or {}
    lines = [
        "# Shadow Account Review",
        "",
        "Research-only review derived from personal brokerage execution history. It does not connect to a broker or place orders.",
        "",
        "## Dataset",
        f"- Period: {summary.get('period_start', 'NA')} to {summary.get('period_end', 'NA')}",
        f"- Matched round trips: {summary.get('matched_round_trips', 'NA')}",
        f"- Realized PnL: {summary.get('realized_pnl', 'NA')}",
        f"- Win rate: {summary.get('win_rate', 'NA')}",
        "",
        "## Rules",
    ]
    rules = review.get("rules") or []
    if rules:
        lines.extend(["| action | source | value | trades | pnl | win_rate | description |", "|---|---|---|---:|---:|---:|---|"])
        for rule in rules:
            lines.append(
                f"| {rule['action']} | {rule['source']} | {rule['value']} | {rule['trades']} | "
                f"{rule['pnl']:.2f} | {rule['win_rate']:.2%} | {rule['description']} |"
            )
    else:
        lines.append("- No rule met the minimum support threshold.")
    lines.extend(["", "## Counterfactual"])
    for key, value in (review.get("counterfactual") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Warnings"])
    warnings = review.get("warnings") or []
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- none"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a research-only shadow account review from personal trade outputs.")
    parser.add_argument("--review-dir", default="outputs/personal_trade_review_20260629")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-trades", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_shadow_account_review(
        Path(args.review_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        min_trades=args.min_trades,
    )
    print("Shadow account outputs:")
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
