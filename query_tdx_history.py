from __future__ import annotations

import argparse

from research_database import ResearchDatabase


def main() -> None:
    parser = argparse.ArgumentParser(description="Query coverage from the separate TDX history database.")
    parser.add_argument("--db", default="data/research.sqlite3")
    parser.add_argument("--tdx-db", default="data/tdx_history.sqlite3")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--market", choices=["SH", "SZ"])
    parser.add_argument("--asset-type", default="stock", choices=["stock", "index"])
    args = parser.parse_args()

    db = ResearchDatabase(args.db)
    history = db.query_tdx_history_normalized(
        args.tdx_db,
        symbol=args.symbol,
        market=args.market,
        asset_type=args.asset_type,
    )
    result = (
        history.groupby(["market", "symbol", "asset_type", "raw_asset_type"], as_index=False)
        .agg(first_date=("date", "min"), last_date=("date", "max"), rows=("date", "count"))
        .sort_values(["market", "symbol", "asset_type", "raw_asset_type"])
    )
    print(result.to_markdown(index=False) if not result.empty else "no rows")


if __name__ == "__main__":
    main()
