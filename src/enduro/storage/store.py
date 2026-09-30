"""SQL access to recorded market data via DuckDB.

Exposes two views over the Parquet files written by ParquetSink:
  trades(ts, recv_ts, exchange, symbol, price, amount, side, id, date)
  books(ts, recv_ts, exchange, symbol, bid_px[], bid_sz[], ask_px[], ask_sz[], date)
"""

from __future__ import annotations

from pathlib import Path

import duckdb

KINDS = ("trades", "books")


def connect(root: Path | str) -> duckdb.DuckDBPyConnection:
    root = Path(root)
    con = duckdb.connect()
    for kind in KINDS:
        if not any((root / kind).glob("**/*.parquet")):
            continue
        glob = (root / kind / "**" / "*.parquet").as_posix().replace("'", "''")
        con.execute(
            f"CREATE VIEW {kind} AS "
            f"SELECT * FROM read_parquet('{glob}', hive_partitioning = true, union_by_name = true)"
        )
    return con
