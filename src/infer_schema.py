"""Auto-generate ``schema.json`` from a TAAC 2026 Parquet file/dir.

Output schema is consumed verbatim by ``dataset.PCVRParquetDataset._load_schema``:

    {
      "user_int":   [[fid, vocab_size, dim], ...],
      "item_int":   [[fid, vocab_size, dim], ...],
      "user_dense": [[fid, dim], ...],
      "seq": {
        "<domain_key>": {
          "prefix":   "<arrow_column_prefix>",
          "ts_fid":   <int|None>,
          "features": [[fid, vocab_size], ...]
        },
        ...
      }
    }

Naming conventions assumed (per TAAC 2026 demo_1000.parquet, 2026-04-10 layout):
  - ``user_int_feats_{fid}``      : int64 scalar OR list<int64>
  - ``user_dense_feats_{fid}``    : list<float>
  - ``item_int_feats_{fid}``      : int64 scalar OR list<int64>
  - ``domain_{X}_seq_{fid}``      : list<int64>   (X in {a,b,c,d,...})

The inferred ``domain_key`` follows the baseline's convention ``seq_{X}``, so
that ``run.sh``/``train.py`` arg ``--seq_max_lens seq_a:256,seq_b:256,...`` keeps
working unchanged.

Usage:
    python infer_schema.py \
        --parquet /path/to/demo_1000.parquet \
        --output  /path/to/schema.json \
        [--vocab-percentile 1.0] \
        [--dim-percentile  1.0] \
        [--ts-min 1000000000]   # heuristic: median value above this -> ts_fid
"""

import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


_USER_INT_RE   = re.compile(r"^user_int_feats_(\d+)$")
_USER_DENSE_RE = re.compile(r"^user_dense_feats_(\d+)$")
_ITEM_INT_RE   = re.compile(r"^item_int_feats_(\d+)$")
_SEQ_RE        = re.compile(r"^(domain_[a-z]+_seq)_(\d+)$")


def _flat_int_values(col: "pa.ChunkedArray") -> "np.ndarray":
    """Flatten an int column (scalar or list) into a 1D numpy array.

    Null entries become 0; for list columns the offsets are dropped.
    """
    if pa.types.is_list(col.type):
        merged = col.combine_chunks()
        return merged.values.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    return col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)


def _list_lengths(col: "pa.ChunkedArray") -> "np.ndarray":
    """Return per-row list length for a list column (empty array if scalar)."""
    if not pa.types.is_list(col.type):
        return np.zeros(0, dtype=np.int64)
    merged = col.combine_chunks()
    offsets = merged.offsets.to_numpy()
    return np.diff(offsets).astype(np.int64, copy=False)


def _vocab_size(values: "np.ndarray", percentile: float = 1.0) -> int:
    """Return ``vocab_size = max_value + 1`` over positive entries.

    ``percentile`` < 1.0 picks the corresponding quantile instead of max,
    which is helpful when a tiny fraction of OOB ids would otherwise force
    the embedding table to be huge.
    """
    pos = values[values > 0]
    if pos.size == 0:
        return 1
    if percentile >= 1.0:
        return int(pos.max()) + 1
    return int(np.quantile(pos, percentile)) + 1


def _list_dim(col: "pa.ChunkedArray", percentile: float = 1.0) -> int:
    """Return the truncation ``dim`` for a list column.

    For scalar columns this is always 1.
    """
    lengths = _list_lengths(col)
    if lengths.size == 0:
        return 1
    if percentile >= 1.0:
        return max(int(lengths.max()), 1)
    return max(int(np.quantile(lengths, percentile)), 1)


def _looks_like_timestamp(values: "np.ndarray", ts_min: int) -> bool:
    """Heuristic timestamp detector.

    A column is considered a Unix-timestamp column when the median of its
    positive entries is above ``ts_min`` (default 1e9, ~year 2001).
    """
    pos = values[values > 0]
    if pos.size == 0:
        return False
    return float(np.median(pos)) > ts_min


def _resolve_files(parquet_path: str) -> List[str]:
    if os.path.isdir(parquet_path):
        files = sorted(glob.glob(os.path.join(parquet_path, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No .parquet files in {parquet_path}")
        return files
    return [parquet_path]


def _domain_key(prefix: str) -> str:
    """Map ``domain_X_seq`` -> ``seq_X`` (matches baseline convention)."""
    m = re.match(r"^domain_([a-z]+)_seq$", prefix)
    return f"seq_{m.group(1)}" if m else prefix


def infer_schema_from_parquet(
    parquet_path: str,
    output_path: Optional[str] = None,
    vocab_percentile: float = 1.0,
    dim_percentile: float = 1.0,
    ts_min: int = 1_000_000_000,
    sample_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Infer a TAAC 2026 schema dict from a Parquet file or directory.

    Args:
        parquet_path: Either a directory of ``*.parquet`` shards or a single
            ``.parquet`` file.
        output_path: If given, the resulting dict is also written to disk as
            JSON (``indent=2``).
        vocab_percentile: Quantile in (0, 1] used to derive ``vocab_size``
            for int columns. ``1.0`` (default) is exact max; lower values
            clip ultra-high-cardinality outliers.
        dim_percentile: Quantile in (0, 1] used to derive ``dim`` for list
            columns. ``1.0`` (default) covers every observed length.
        ts_min: Minimum median value below which a sequence column is *not*
            considered a timestamp column.
        sample_rows: If set, only the first N rows of the first file are
            scanned (faster on TB-scale datasets at the cost of accuracy).

    Returns:
        The schema dict in the exact format consumed by
        ``dataset.PCVRParquetDataset``.
    """
    files = _resolve_files(parquet_path)
    table = pq.read_table(files[0])
    if sample_rows is not None and table.num_rows > sample_rows:
        table = table.slice(0, sample_rows)

    schema: Dict[str, Any] = {
        "user_int":   [],
        "item_int":   [],
        "user_dense": [],
        "seq":        {},
    }
    seq_groups: Dict[str, List[Tuple[int, "pa.ChunkedArray"]]] = {}

    for col_name in table.schema.names:
        col = table.column(col_name)

        m = _USER_INT_RE.match(col_name)
        if m:
            fid = int(m.group(1))
            vals = _flat_int_values(col)
            vs = _vocab_size(vals, vocab_percentile)
            dim = _list_dim(col, dim_percentile)
            schema["user_int"].append([fid, vs, dim])
            continue

        m = _ITEM_INT_RE.match(col_name)
        if m:
            fid = int(m.group(1))
            vals = _flat_int_values(col)
            vs = _vocab_size(vals, vocab_percentile)
            dim = _list_dim(col, dim_percentile)
            schema["item_int"].append([fid, vs, dim])
            continue

        m = _USER_DENSE_RE.match(col_name)
        if m:
            fid = int(m.group(1))
            dim = _list_dim(col, dim_percentile)
            schema["user_dense"].append([fid, dim])
            continue

        m = _SEQ_RE.match(col_name)
        if m:
            prefix = m.group(1)
            fid = int(m.group(2))
            seq_groups.setdefault(prefix, []).append((fid, col))
            continue

    for prefix, fid_cols in seq_groups.items():
        ts_fid: Optional[int] = None
        features: List[List[int]] = []
        for fid, col in sorted(fid_cols, key=lambda x: x[0]):
            vals = _flat_int_values(col)
            vs = _vocab_size(vals, vocab_percentile)
            features.append([fid, vs])
            if ts_fid is None and _looks_like_timestamp(vals, ts_min):
                ts_fid = fid
        schema["seq"][_domain_key(prefix)] = {
            "prefix":   prefix,
            "ts_fid":   ts_fid,
            "features": features,
        }

    schema["user_int"].sort(key=lambda x: x[0])
    schema["item_int"].sort(key=lambda x: x[0])
    schema["user_dense"].sort(key=lambda x: x[0])

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)

    return schema


def _summarize(schema: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"  user_int   : {len(schema['user_int']):3d} cols")
    lines.append(f"  item_int   : {len(schema['item_int']):3d} cols")
    lines.append(f"  user_dense : {len(schema['user_dense']):3d} cols")
    lines.append(f"  seq        : {len(schema['seq']):3d} domains")
    for k, v in sorted(schema["seq"].items()):
        ts = v["ts_fid"] if v["ts_fid"] is not None else "<none>"
        lines.append(f"    - {k:<10s} prefix={v['prefix']:<20s} "
                     f"ts_fid={ts}  features={len(v['features'])}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer schema.json from a TAAC 2026 parquet file/dir.")
    parser.add_argument("--parquet", required=True,
                        help="Path to a parquet file OR directory of *.parquet shards.")
    parser.add_argument("--output", required=True,
                        help="Destination schema.json path.")
    parser.add_argument("--vocab-percentile", type=float, default=1.0,
                        help="Quantile (0,1] for vocab_size (default 1.0 = exact max).")
    parser.add_argument("--dim-percentile", type=float, default=1.0,
                        help="Quantile (0,1] for list dim (default 1.0 = exact max).")
    parser.add_argument("--ts-min", type=int, default=1_000_000_000,
                        help="Median-value threshold for ts_fid detection (default 1e9).")
    parser.add_argument("--sample-rows", type=int, default=None,
                        help="If set, only scan the first N rows of the first shard.")
    args = parser.parse_args()

    schema = infer_schema_from_parquet(
        parquet_path=args.parquet,
        output_path=args.output,
        vocab_percentile=args.vocab_percentile,
        dim_percentile=args.dim_percentile,
        ts_min=args.ts_min,
        sample_rows=args.sample_rows,
    )
    print(f"Wrote schema to {args.output}")
    print(_summarize(schema))


if __name__ == "__main__":
    main()
