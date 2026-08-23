"""Shared CSV/Parquet I/O for large point-in-time market panels."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from uuid import uuid4

import pandas as pd


PARQUET_SUFFIXES = {".parquet", ".pq"}


def is_parquet_path(path: Path) -> bool:
    return path.suffix.lower() in PARQUET_SUFFIXES


def panel_columns(path: Path) -> list[str]:
    if is_parquet_path(path):
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema.names)
    return pd.read_csv(path, nrows=0).columns.tolist()


def read_panel(
    path: Path,
    *,
    columns: Sequence[str] | None = None,
    dtype: Mapping[str, object] | None = None,
    parse_dates: Sequence[str] | None = None,
    filters: Sequence[tuple[str, str, object]] | None = None,
    low_memory: bool = False,
) -> pd.DataFrame:
    selected = list(columns) if columns is not None else None
    if is_parquet_path(path):
        frame = pd.read_parquet(path, columns=selected, filters=filters)
    else:
        frame = pd.read_csv(
            path,
            usecols=selected,
            dtype=dtype,
            parse_dates=list(parse_dates) if parse_dates else None,
            low_memory=low_memory,
        )
        for column, operator, value in filters or ():
            if operator == "==":
                keep = frame[column].eq(value)
            elif operator == ">=":
                keep = frame[column].ge(value)
            elif operator == ">":
                keep = frame[column].gt(value)
            elif operator == "<=":
                keep = frame[column].le(value)
            elif operator == "<":
                keep = frame[column].lt(value)
            elif operator == "in":
                keep = frame[column].isin(value)
            else:
                raise ValueError(f"unsupported panel filter operator: {operator}")
            frame = frame.loc[keep]
    for column, value in (dtype or {}).items():
        if column in frame and is_parquet_path(path):
            frame[column] = frame[column].astype(value)
    for column in parse_dates or ():
        if column in frame and is_parquet_path(path):
            frame[column] = pd.to_datetime(frame[column], errors="raise")
    return frame


def iter_panel(
    path: Path,
    *,
    columns: Sequence[str] | None = None,
    dtype: Mapping[str, object] | None = None,
    chunksize: int = 250_000,
) -> Iterator[pd.DataFrame]:
    selected = list(columns) if columns is not None else None
    if is_parquet_path(path):
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=chunksize, columns=selected):
            frame = batch.to_pandas()
            for column, value in (dtype or {}).items():
                if column in frame:
                    frame[column] = frame[column].astype(value)
            yield frame
        return
    yield from pd.read_csv(
        path,
        usecols=selected,
        dtype=dtype,
        low_memory=False,
        chunksize=chunksize,
    )


def write_panel_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp{path.suffix}")
    try:
        if is_parquet_path(path):
            frame.to_parquet(
                temporary,
                index=False,
                engine="pyarrow",
                compression="zstd",
            )
        else:
            frame.to_csv(temporary, index=False, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
