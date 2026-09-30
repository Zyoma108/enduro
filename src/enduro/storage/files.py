"""File helpers shared by storage writers."""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet_atomic(table: pa.Table, path: Path) -> None:
    """Write to a temp file and rename, so readers never see a partially written file.

    The temp name does not end in .parquet, so `**/*.parquet` globs skip it.
    """
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)
