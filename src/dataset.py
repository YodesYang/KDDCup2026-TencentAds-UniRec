"""PCVR Parquet dataset module (performance-tuned).

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.
"""

import os
import logging
import random
import json
import gc
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

# numpy.typing is available since numpy >= 1.20; on older numpy fall back to a
# no-op shim so that forward-referenced annotations like ``npt.NDArray[np.int64]``
# keep working as plain strings without raising at import time.
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover
    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


# ─────────────────────────── Feature Schema ──────────────────────────────────


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature so downstream
    code can locate the segment of the flattened tensor that belongs to a
    specific feature id.

    For int features:
      - int_value: length = 1
      - int_array: length = array length
      - int_array_and_float_array: int part length
    For dense features:
      - float_value: length = 1
      - float_array: length = array length
      - int_array_and_float_array: float part length
    """

    def __init__(self) -> None:
        # Ordered list of (feature_id, offset, length).
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        # Quick lookup from fid to its (offset, length).
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        """Append a feature to the schema."""
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        """Get ``(offset, length)`` for a feature_id."""
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        """Return all feature_ids in their insertion order."""
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for JSON dumping)."""
        return {
            'entries': self.entries,
            'total_dim': self.total_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSchema':
        """Reconstruct a :class:`FeatureSchema` from its dict form."""
        schema = cls()
        for fid, offset, length in d['entries']:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d['total_dim']
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)

# Use filesystem-based tensor sharing (instead of /dev/shm) to avoid running
# out of shared memory when many DataLoader workers are active.
torch.multiprocessing.set_sharing_strategy('file_system')

# Time-delta bucket boundaries (63 edges -> 64 buckets: 0=padding, 1..63).
#
# Defaults to a hand-designed log-scale ladder from 5s to 1yr. Can be overridden
# per-run via ``PCVRParquetDataset(bucket_boundaries=...)`` / CLI
# ``--time_bucket_boundaries`` (T30). The override length MUST equal
# ``len(DEFAULT_BUCKET_BOUNDARIES)`` (= 63) so that the embedding table shape
# stays bit-identical between default and tuned runs — only the edge positions
# move, the number of slots does not, which keeps ``NUM_TIME_BUCKETS`` a
# compile-time constant and preserves checkpoint compatibility.
DEFAULT_BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

# Back-compat alias. Historical callers (and EDA sanity-check code) imported
# ``BUCKET_BOUNDARIES`` directly. Keep the name pointing at the default ladder;
# *runtime* bucketization uses the per-dataset override (see
# ``PCVRParquetDataset._bucket_boundaries``) rather than this module-level
# constant so overrides do not affect unrelated imports.
BUCKET_BOUNDARIES = DEFAULT_BUCKET_BOUNDARIES

# Total number of time-bucket embedding slots (= number of boundaries + 1, with
# padding=0 included).
#
# This constant is uniquely determined by the length of DEFAULT_BUCKET_BOUNDARIES;
# on the model side, ``nn.Embedding(num_embeddings=NUM_TIME_BUCKETS)`` must
# match this value exactly, otherwise an IndexError may be raised at runtime.
# Because T30 overrides keep ``len(...)`` unchanged, this constant is also
# correct for tuned runs.
NUM_TIME_BUCKETS = len(DEFAULT_BUCKET_BOUNDARIES) + 1


def fit_log_scale_boundaries(
    time_diffs: np.ndarray,
    num_boundaries: int = 63,
    min_sec: int = 5,
) -> np.ndarray:
    """T38 · Fit log-scale bucket boundaries to a sample of time_diffs.

    Used by per-domain time bucket (T38). Given a 1D array of positive
    time_diff values (in seconds), fit ``num_boundaries`` log-spaced
    boundaries spanning [min_sec, p99(time_diffs)] so each domain gets
    boundaries matching its own time-scale rather than a hand-tuned global
    ladder.

    The output length MUST equal ``len(DEFAULT_BUCKET_BOUNDARIES)`` (=63)
    so the embedding table shape (NUM_TIME_BUCKETS=64) stays bit-identical
    between default and tuned runs.

    Args:
        time_diffs: 1D array of time_diff values in seconds (positive). Zeros
            and negatives are filtered before fitting.
        num_boundaries: Number of boundary edges (default 63).
        min_sec: Minimum boundary value in seconds (default 5).

    Returns:
        1D int64 array of strictly-increasing positive boundary values, with
        length == num_boundaries.
    """
    valid = time_diffs[time_diffs > 0]
    if valid.size < 100:
        # Insufficient data for robust quantile fit. Fall back to default
        # ladder. Caller should log this case.
        return DEFAULT_BUCKET_BOUNDARIES.copy()
    p99 = float(np.quantile(valid, 0.99))
    if p99 <= min_sec:
        return DEFAULT_BUCKET_BOUNDARIES.copy()
    log_min = np.log(min_sec)
    log_max = np.log(p99)
    log_edges = np.linspace(log_min, log_max, num_boundaries)
    edges = np.round(np.exp(log_edges)).astype(np.int64)
    # Clamp first edge to >= min_sec (round() can produce min_sec-1 for
    # log-min boundary due to floating-point round-trip).
    edges[0] = max(edges[0], min_sec)
    # Enforce strict monotonic increase by adding 1 to any duplicate.
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1
    return edges


def _scan_per_domain_time_diffs(
    train_rgs: List[Tuple[str, int, int]],
    schema_path: Optional[str],
    sample_rgs: int = 8,
    max_rows_per_rg: int = 50000,
    max_samples_per_rg_domain: int = 100000,
    max_samples_per_domain: int = 5000000,
    timestamp_min: Optional[int] = None,
    timestamp_max: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """T38 · Scan a sample of train RGs to gather per-domain time_diff data.

    Returns a Dict[domain, np.ndarray] where each array contains positive
    time_diff values (current row timestamp - seq item timestamp). Used to
    fit per-domain log-scale bucket boundaries.

    Args:
        train_rgs: List of (parquet_file, rg_idx, n_rows) tuples for training.
        schema_path: Path to schema.json. Required to resolve real column
            names ``f"{prefix}_{ts_fid}"`` for each seq domain. When None,
            falls back to the legacy hardcoded ``f"{d}_timestamp"`` pattern
            (which **does not match the platform schema** and was the cause
            of the 5/16 silent ``n_samples=0`` bug · EXP-064 §3.3).
        sample_rgs: Number of Row Groups to scan (default 8 ≈ 6.4M time_diff
            samples per domain at typical RG size).
        max_rows_per_rg: Cap rows per RG to bound scan time.
        max_samples_per_rg_domain: Reservoir cap per Row Group and domain.
        max_samples_per_domain: Total reservoir cap per domain.
        timestamp_min: Optional inclusive row-level lower bound.
        timestamp_max: Optional exclusive row-level upper bound.

    Returns:
        Dict[domain, 1D np.ndarray of int64 time_diff seconds]. Domains
        without timestamp columns are skipped.
    """
    import pyarrow.parquet as pq  # local import: scan-only helper.
    if not train_rgs:
        raise ValueError("_scan_per_domain_time_diffs: train_rgs is empty")
    n_sample = max(1, min(int(sample_rgs), len(train_rgs)))
    if n_sample >= len(train_rgs):
        sample = train_rgs
    else:
        sample_idx = np.linspace(
            0, len(train_rgs) - 1, num=n_sample, dtype=np.int64)
        sample = [train_rgs[int(i)] for i in sample_idx]
    pf_cache: Dict[str, "pq.ParquetFile"] = {}
    seq_domains = ['seq_a', 'seq_b', 'seq_c', 'seq_d']
    rng = np.random.default_rng(20260518)

    # T38 fix v2 (5/17 EXP-066) · Resolve real timestamp column names from
    # schema.json. Platform parquet uses ``{prefix}_{ts_fid}`` (e.g.
    # ``domain_a_seq_39``), NOT the hardcoded ``{d}_timestamp`` (which is the
    # internal logical alias and does not exist in any parquet file).
    # Legacy fallback: if schema_path is None, retain old behavior (broken on
    # platform) so downstream still gets a Dict (with empty arrays).
    domain_to_ts_col: Dict[str, str] = {}
    if schema_path is not None:
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_raw = json.load(f)
            seq_cfg = schema_raw.get('seq', {}) or {}
            for d in seq_domains:
                cfg = seq_cfg.get(d)
                if not cfg:
                    continue
                prefix = cfg.get('prefix')
                ts_fid = cfg.get('ts_fid')
                if prefix is None or ts_fid is None:
                    continue
                domain_to_ts_col[d] = f"{prefix}_{ts_fid}"
            logging.info(
                "[T38] resolved timestamp columns from schema: "
                f"{domain_to_ts_col}")
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logging.warning(
                f"[T38] schema parse failed ({e!r}); falling back to legacy "
                f"hardcoded column names · per-domain fit will likely return "
                "empty (n_samples=0).")
            domain_to_ts_col = {d: f"{d}_timestamp" for d in seq_domains}
    else:
        domain_to_ts_col = {d: f"{d}_timestamp" for d in seq_domains}

    diffs: Dict[str, List[np.ndarray]] = {d: [] for d in seq_domains}
    diff_counts: Dict[str, int] = {d: 0 for d in seq_domains}
    for parquet_path, rg_idx, _n_rows in sample:
        if all(diff_counts[d] >= max_samples_per_domain for d in seq_domains):
            break
        pf = pf_cache.get(parquet_path)
        if pf is None:
            pf = pq.ParquetFile(parquet_path)
            pf_cache[parquet_path] = pf
        # Read only the columns we need.
        cols_to_read = ['timestamp']
        for d in seq_domains:
            ts_col = domain_to_ts_col.get(d)
            if ts_col:
                cols_to_read.append(ts_col)
        avail = set(pf.schema_arrow.names)
        cols_to_read = [c for c in cols_to_read if c in avail]
        if 'timestamp' not in cols_to_read:
            continue
        table = pf.read_row_group(rg_idx, columns=cols_to_read)
        if table.num_rows == 0:
            continue
        if table.num_rows > max_rows_per_rg:
            table = table.slice(0, max_rows_per_rg)
        ts = table.column('timestamp').to_numpy().astype(np.int64)
        if timestamp_min is not None or timestamp_max is not None:
            keep_mask = np.ones(ts.shape[0], dtype=np.bool_)
            if timestamp_min is not None:
                keep_mask &= ts >= int(timestamp_min)
            if timestamp_max is not None:
                keep_mask &= ts < int(timestamp_max)
            if not keep_mask.any():
                continue
            if not keep_mask.all():
                table = table.filter(pa.array(keep_mask))
                ts = table.column('timestamp').to_numpy().astype(np.int64)
        for d in seq_domains:
            ts_col_name = domain_to_ts_col.get(d)
            if ts_col_name is None or ts_col_name not in cols_to_read:
                continue
            if diff_counts[d] >= max_samples_per_domain:
                continue
            seq_ts_col = table.column(ts_col_name)
            # T38 fix v2 (5/17): Arrow column may be ChunkedArray (no
            # .offsets) or ListArray (has .offsets). Combine to a single
            # ListArray to access offsets/values uniformly. ChunkedArray
            # arises when parquet RG contains multiple Arrow chunks (e.g.
            # demo data) · ListArray when single chunk.
            if hasattr(seq_ts_col, 'combine_chunks'):
                seq_ts_col = seq_ts_col.combine_chunks()
            seq_offs = seq_ts_col.offsets.to_numpy()
            seq_vals = seq_ts_col.values.to_numpy().astype(np.int64)
            rg_parts: List[np.ndarray] = []
            for i in range(table.num_rows):
                s = int(seq_offs[i])
                e = int(seq_offs[i + 1])
                if e <= s:
                    continue
                td = ts[i] - seq_vals[s:e]
                td = td[td > 0]
                if td.size > 0:
                    rg_parts.append(td)
            if not rg_parts:
                continue
            rg_td = np.concatenate(rg_parts)
            if rg_td.size > max_samples_per_rg_domain:
                take = rng.choice(
                    rg_td.size, size=max_samples_per_rg_domain, replace=False)
                rg_td = rg_td[take]
            remaining = max_samples_per_domain - diff_counts[d]
            if remaining <= 0:
                continue
            if rg_td.size > remaining:
                take = rng.choice(rg_td.size, size=remaining, replace=False)
                rg_td = rg_td[take]
            diffs[d].append(rg_td)
            diff_counts[d] += int(rg_td.size)
    out: Dict[str, np.ndarray] = {}
    for d in seq_domains:
        if diffs[d]:
            out[d] = np.concatenate(diffs[d])
        else:
            out[d] = np.array([], dtype=np.int64)
    return out


def parse_bucket_boundaries(spec: str | None) -> np.ndarray:
    """Parse a CLI-supplied bucket-boundary specification into a numpy array.

    - ``None`` or empty string → returns ``DEFAULT_BUCKET_BOUNDARIES`` (a copy,
      so callers can mutate without affecting the module-level default).
    - Otherwise expects a comma-separated list of positive integers representing
      time-delta bucket edges in seconds, strictly increasing. Must have
      EXACTLY ``len(DEFAULT_BUCKET_BOUNDARIES)`` values (= 64) to preserve
      embedding-table shape compatibility.

    Returns:
        np.ndarray of dtype int64 with shape (64,).

    Raises:
        ValueError: on length mismatch, non-monotonic values, or non-positive
            entries.
    """
    if spec is None or not str(spec).strip():
        return DEFAULT_BUCKET_BOUNDARIES.copy()
    parts = [p.strip() for p in str(spec).split(',') if p.strip()]
    expected = len(DEFAULT_BUCKET_BOUNDARIES)
    if len(parts) != expected:
        raise ValueError(
            f"time_bucket_boundaries: got {len(parts)} values, expected "
            f"{expected} (= fixed embedding slot count). Either supply exactly "
            f"{expected} comma-separated positive integers, or leave the flag "
            "empty to use the default ladder."
        )
    try:
        vals = np.array([int(p) for p in parts], dtype=np.int64)
    except ValueError as e:
        raise ValueError(
            f"time_bucket_boundaries: failed to parse '{spec}' as integers: {e}"
        ) from e
    if (vals <= 0).any():
        raise ValueError(
            f"time_bucket_boundaries: all values must be strictly positive "
            f"seconds; got {vals.tolist()}"
        )
    if (np.diff(vals) <= 0).any():
        raise ValueError(
            f"time_bucket_boundaries: values must be strictly increasing; "
            f"got {vals.tolist()}"
        )
    return vals

# Synthetic user-dense feature ids. They are not read from parquet columns;
# they are appended in-memory when the corresponding CLI flags are enabled.
COUNT_FEATURE_FID = 900001
SEQ_STATS_FEATURE_FID = 900002
HISTORY_CVR_FEATURE_FID = 900003
TIME_OF_DAY_FEATURE_FID = 900004  # impression hour/dow/weekend (5 dims)
TIME_OF_DAY_FEATURE_DIM = 5       # hour_sin, hour_cos, dow_sin, dow_cos, is_weekend
HOUR_ONLY_FEATURE_FID = 900005
HOUR_ONLY_FEATURE_DIM = 2        # hour_sin, hour_cos only (no DOW/weekend)
BEIJING_TIME_FEATURE_FID = 900007
BEIJING_TIME_FEATURE_DIM = 10    # bj hour/dow/weekend + test-window indicators
BEIJING_TIME_V2_FEATURE_FID = 900008
BEIJING_TIME_V2_FEATURE_DIM = 14  # bj date distance + daypart distribution flags
BEIJING_TARGET_DATE = "2026-03-04"
BEIJING_FESTIVAL_DATE = "2026-03-03"

# T31 · is-missing bitmap synthetic feature. One float per selected user_int /
# item_int fid carrying 1.0 when the raw parquet cell was null / <=0, else 0.0.
# Concatenated to user_dense so the model receives "missingness" as an
# independent dense signal in addition to the reserved embedding slot (which is
# ALSO populated via vocab_size+1 when enable_is_missing is active — belt +
# suspenders per DECEM Trick 3).
IS_MISSING_FEATURE_FID = 900006


def parse_int_csv(spec: str | None) -> Optional[List[int]]:
    """Parse a comma-separated list of positive integers (e.g. fids).

    - ``None`` or empty string → ``None`` (caller may interpret as "default").
    - Otherwise returns a list of ints, deduplicated, sorted ascending.

    Raises ValueError on non-integer or non-positive entries.
    """
    if spec is None or not str(spec).strip():
        return None
    parts = [p.strip() for p in str(spec).split(',') if p.strip()]
    try:
        vals = sorted({int(p) for p in parts})
    except ValueError as e:
        raise ValueError(
            f"failed to parse '{spec}' as comma-separated integers: {e}"
        ) from e
    if any(v <= 0 for v in vals):
        raise ValueError(f"all fids must be positive; got {vals}")
    return vals


class HistoryCVRStore:
    """Leak-safe historical conversion-rate lookup table.

    The store is built from training Row Groups only. At lookup time each row
    uses cumulative statistics whose bin end is <= ``timestamp - cutoff_sec``.
    This lets train, validation, and inference share one compact sidecar while
    preserving time order.
    """

    def __init__(
        self,
        path: str,
        cutoff_sec: Optional[int] = None,
        prior_strength: Optional[float] = None,
    ) -> None:
        if not path:
            raise ValueError("history CVR features require a cache path")
        if not os.path.exists(path):
            raise FileNotFoundError(f"history CVR cache not found: {path}")

        data = np.load(path)
        self.path = path
        self.fids = data['fids'].astype(np.int64)
        self.fid_offsets = data['fid_offsets'].astype(np.int64)
        self.values = data['values'].astype(np.int64)
        self.value_offsets = data['value_offsets'].astype(np.int64)
        self.bin_end_ts = data['bin_end_ts'].astype(np.int64)
        self.counts = data['counts'].astype(np.float32)
        self.positives = data['positives'].astype(np.float32)
        self.global_bin_end_ts = data['global_bin_end_ts'].astype(np.int64)
        self.global_counts = data['global_counts'].astype(np.float32)
        self.global_positives = data['global_positives'].astype(np.float32)
        self.global_prior = float(data['global_prior'][0])
        self.cutoff_sec = int(
            cutoff_sec if cutoff_sec is not None else data['cutoff_sec'][0])
        self.prior_strength = float(
            prior_strength if prior_strength is not None
            else data['prior_strength'][0])
        self.feature_dim = 2 + 2 * len(self.fids)

        self._fid_to_slot = {int(fid): i for i, fid in enumerate(self.fids)}
        self._value_to_slot: Dict[int, Dict[int, int]] = {}
        for i, fid in enumerate(self.fids):
            start = int(self.fid_offsets[i])
            end = int(self.fid_offsets[i + 1])
            self._value_to_slot[int(fid)] = {
                int(v): j for j, v in enumerate(self.values[start:end], start)
            }

    def _global_at(self, cutoff_ts: int) -> Tuple[float, float]:
        idx = int(np.searchsorted(
            self.global_bin_end_ts, cutoff_ts, side='right')) - 1
        if idx < 0:
            return 0.0, 0.0
        return float(self.global_positives[idx]), float(self.global_counts[idx])

    def _value_at(
        self,
        fid: int,
        value: int,
        cutoff_ts: int,
    ) -> Tuple[float, float]:
        slot = self._value_to_slot.get(fid, {}).get(value)
        if slot is None:
            return 0.0, 0.0
        start = int(self.value_offsets[slot])
        end = int(self.value_offsets[slot + 1])
        idx = int(np.searchsorted(
            self.bin_end_ts[start:end], cutoff_ts, side='right')) - 1
        if idx < 0:
            return 0.0, 0.0
        pos = float(self.positives[start + idx])
        cnt = float(self.counts[start + idx])
        return pos, cnt

    def lookup_batch(
        self,
        timestamps: "npt.NDArray[np.int64]",
        item_int: "npt.NDArray[np.int64]",
        fid_offsets: List[Tuple[int, int]],
    ) -> "npt.NDArray[np.float32]":
        out = np.zeros((len(timestamps), self.feature_dim), dtype=np.float32)
        for row_idx, ts in enumerate(timestamps):
            cutoff_ts = int(ts) - self.cutoff_sec
            global_pos, global_cnt = self._global_at(cutoff_ts)
            global_rate = (
                global_pos / global_cnt if global_cnt > 0
                else self.global_prior)
            out[row_idx, 0] = global_rate
            out[row_idx, 1] = np.log1p(global_cnt)

            for local_idx, (fid, offset) in enumerate(fid_offsets):
                value = int(item_int[row_idx, offset])
                pos, cnt = self._value_at(fid, value, cutoff_ts)
                rate = (
                    (pos + self.prior_strength * global_rate)
                    / (cnt + self.prior_strength)
                    if cnt > 0 else global_rate
                )
                base = 2 + 2 * local_idx
                out[row_idx, base] = rate
                out[row_idx, base + 1] = np.log1p(cnt)
        return out


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    - int features: scalar or list (multi-hot); values <= 0 are mapped to 0 (padding).
    - dense features: ``list<float>``, variable-length padded up to ``max_dim``.
    - sequence features: ``list<int64>``, grouped by domain; includes side-info
      columns and an optional timestamp column (used for time-bucketing).
    - label: mapped from ``label_type == 2``.
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        row_groups: Optional[List[Tuple[str, int, int]]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        timestamp_min: Optional[int] = None,
        timestamp_max: Optional[int] = None,
        enable_count_features: bool = False,
        enable_seq_stats_features: bool = False,
        enable_history_cvr_features: bool = False,
        history_cvr_cache_path: Optional[str] = None,
        history_cvr_time_mode: str = 'timestamp_cutoff',
        history_cvr_cutoff_sec: int = 86400,
        history_cvr_available_lag_sec: int = 0,
        history_cvr_prior_strength: float = 20.0,
        enable_mature_negative_weighting: bool = False,
        negative_maturity_sec: int = 86400,
        immature_negative_weight: float = 0.0,
        negative_maturity_end_ts: Optional[int] = None,
        dense_log1p_fids: frozenset[int] | None = None,
        enable_time_of_day_features: bool = False,
        enable_beijing_time_features: bool = False,
        enable_beijing_time_v2_features: bool = False,
        temporal_weight_alpha: float = 0.0,
        temporal_weight_ts_min: Optional[int] = None,
        temporal_weight_ts_max: Optional[int] = None,
        # 5/17 EXP-069 · Hour-aware sample reweight (Beijing time = UTC + 8h).
        # When all 3 set, train rows with beijing_hour ∈ [min, max] get weight
        # × multiplier. Test daypart = 北京 09~14 → set min=9, max=14, mul=5.0
        # to upweight test-daypart-aligned samples 5× (vs M41 which is daily
        # exponential reweight). Composes multiplicatively with M41
        # temporal_weight_alpha (alpha + hour can stack: M41 selects 03-04 ·
        # hour_weight selects 03-04 09~14 daypart within 03-04).
        hour_weight_min: Optional[int] = None,
        hour_weight_max: Optional[int] = None,
        hour_weight_multiplier: Optional[float] = None,
        # 5/18 M83 · Exact Beijing day+hour sample reweight. Unlike
        # hour_weight_* this can target the observed same-day overlap window
        # and uses a global normalizer fitted before training.
        target_day_hour_weight_date: str = "",
        target_day_hour_weight_min: Optional[int] = None,
        target_day_hour_weight_max: Optional[int] = None,
        target_day_hour_weight_multiplier: Optional[float] = None,
        target_day_hour_weight_norm: Optional[float] = None,
        enable_hour_only_features: bool = False,
        id_mask_prob: float = 0.0,
        id_mask_seq_domains: Optional[List[str]] = None,
        disable_seq_fids: Optional[List[int]] = None,
        enable_eda_dump: bool = False,
        eda_reservoir_size: int = 20000,
        bucket_boundaries: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        enable_is_missing: bool = False,
        is_missing_user_int_fids: Optional[List[int]] = None,
        is_missing_item_int_fids: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            parquet_path: either a directory containing ``*.parquet`` files or
                a single parquet file path.
            schema_path: path of the schema JSON describing feature layouts.
            batch_size: fixed batch size used for the pre-allocated buffers.
            seq_max_lens: optional per-domain override of sequence truncation,
                e.g. ``{'seq_d': 256}``. Domains not listed fall back to the
                schema default of 256.
            shuffle: whether to shuffle within a ``buffer_batches``-sized window.
            buffer_batches: shuffle buffer size in units of batches.
            row_group_range: ``(start, end)`` slice of Row Groups; ``None`` to
                use all Row Groups.
            row_groups: explicit Row Group list. When provided, it takes
                precedence over ``row_group_range`` and preserves caller-defined
                ordering, e.g. timestamp-sorted splits.
            clip_vocab: if True, clip out-of-bound ids to 0; if False, raise.
            is_training: if True, derive ``label`` from ``label_type == 2``;
                if False, return an all-zeros label column.
            timestamp_min: optional inclusive row-level timestamp lower bound.
            timestamp_max: optional exclusive row-level timestamp upper bound.
            enable_count_features: append user/item non-zero counts to dense
                features.
            enable_seq_stats_features: append per-domain sequence lengths and
                timestamp ranges to dense features.
            enable_history_cvr_features: append leak-safe historical item CVR
                statistics loaded from ``history_cvr_cache_path``.
            history_cvr_cache_path: ``.npz`` sidecar produced by
                ``_build_history_cvr_cache``.
            history_cvr_time_mode: ``timestamp_cutoff`` uses event timestamp
                bins plus a lookup lag; ``available`` bins rows by label
                availability time.
            history_cvr_cutoff_sec: minimum age of rows allowed into the
                historical lookup for the current sample in
                ``timestamp_cutoff`` mode.
            history_cvr_available_lag_sec: optional lookup lag in
                ``available`` mode.
            history_cvr_prior_strength: smoothing strength against the global
                historical conversion rate.
            enable_mature_negative_weighting: downweight negatives whose
                24h conversion window has not fully elapsed by the observation
                end timestamp.
            negative_maturity_sec: conversion observation window used to decide
                whether a negative sample is mature.
            immature_negative_weight: sample weight assigned to immature
                negatives; 0 drops them from the loss.
            negative_maturity_end_ts: max observable timestamp for the training
                split. Required when mature-negative weighting is enabled.
            dense_log1p_fids: set of user_dense fids whose raw values are
                large-magnitude counts/frequencies and should be compressed with
                ``log1p`` before being fed into the dense projection layer.
                Defaults to ``{62, 63, 64, 65, 66}`` (EDA confirmed these
                columns contain values in the 10^3–10^5 range, while other dense
                columns such as fid=61/87 are pre-normalised embeddings near 0).
                Pass ``frozenset()`` to disable all scaling.
            enable_time_of_day_features: if True, append 5 synthetic dense
                features derived from the impression ``timestamp``:
                ``hour_sin``, ``hour_cos`` (hour-of-day in [0,23] → circular
                encoding), ``dow_sin``, ``dow_cos`` (day-of-week in [0,6] →
                circular encoding), and ``is_weekend`` (1 if Sat/Sun else 0).
                These capture diurnal and weekly periodicity in user purchase
                intent that is absent from the existing sequence features.
            enable_beijing_time_features: if True, append 10 synthetic dense
                features derived from Beijing local time (UTC+8): first/second
                harmonic hour encodings, day-of-week encoding, weekend/workday
                flags, test-window hour 09-14 flag, and early-morning hour
                06-08 flag. This is separate from
                ``enable_time_of_day_features`` so experiments can either
                replace or append the older UTC time features.
            enable_beijing_time_v2_features: if True, append 14 additional
                Beijing-local distribution features: distance to the observed
                test date, festival / post-festival date flags, distance to the
                09-14 test window, coarse daypart one-hot features, and a
                target-day test-window indicator. This is feature-only; it does
                not reweight or filter samples.
            id_mask_prob: token-level random mask probability applied on the
                training path only. Counteracts the "dev AUC up / LB AUC
                down" symptom that a community 0.85+ player attributed to
                high OOV rate on ``item_id``-like features in the test set
                (comment thread on lexiang, 2026-05-04): with some
                probability each non-padding token is replaced by 0, forcing
                the model to rely on context rather than memorising specific
                IDs. No effect unless ``shuffle=True`` (treated as a proxy
                for "training path"), so valid / holdout / inference loaders
                are unaffected.
            id_mask_seq_domains: which sequence domains to apply the id
                mask to. ``None`` means "all domains"; pass e.g. ``['c']`` to
                only mask the longest-tail domain (TAAC's
                ``domain_c_seq_47`` has vocab 278M, most at risk of OOV).
            disable_seq_fids: list of sequence side-info fids whose values
                should be forced to 0 (padding) for every row, effectively
                removing that feature from the model's input. Applies to
                all Row Groups / all batches, including valid and
                inference paths, so the model sees a consistent schema at
                train and test time. Motivation: community reports
                reports ``seq_c feat47 == top-level item_id`` is
                effectively noise ("做不做 match 特征都掉分"), and our P1
                id-mask ablation results
                confirmed input-level random mask is not enough — the
                model still relies on the 85% of id tokens it sees. A
                full structural disable is the next step to verify
                whether dropping the id channel entirely helps. Typical:
                ``[47]`` to drop only the suspected item_id fid in seq_c.
        """
        super().__init__()

        # Accept either a directory, a single file path, or an explicit Row
        # Group list supplied by get_pcvr_data after custom ordering.
        if row_groups is not None:
            if not row_groups:
                raise ValueError("row_groups must be non-empty when provided")
            self._parquet_files = sorted({f for f, _, _ in row_groups})
        elif os.path.isdir(parquet_path):
            import glob
            files = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        self.timestamp_min = timestamp_min
        self.timestamp_max = timestamp_max
        self.enable_count_features = enable_count_features
        self.enable_seq_stats_features = enable_seq_stats_features
        self.enable_eda_dump = bool(enable_eda_dump)
        self.eda_reservoir_size = int(eda_reservoir_size)
        # EDA state (lazy init on first batch to avoid unused allocation).
        # Layout documented at _update_eda_stats() / finalize_eda().
        self._eda_state: Optional[Dict[str, Any]] = None
        # T30 · per-run time-bucket boundaries. None means "use the default
        # hand-designed ladder" which keeps historical behavior bit-identical.
        # Overrides MUST have len == len(DEFAULT_BUCKET_BOUNDARIES) so the
        # embedding table shape (NUM_TIME_BUCKETS) is unaffected.
        #
        # T38 · per-domain bucket boundaries: when bucket_boundaries is a
        # Dict[str, np.ndarray], each domain uses its own boundaries.
        # Otherwise (np.ndarray or None) all domains share the same ladder.
        # Note: ``self._bucket_boundaries`` is read in the seq batching
        # pipeline; helper ``_get_boundaries(domain)`` resolves the right
        # ladder per call.
        if bucket_boundaries is None:
            self._bucket_boundaries: Union[
                np.ndarray, Dict[str, np.ndarray]
            ] = DEFAULT_BUCKET_BOUNDARIES
        elif isinstance(bucket_boundaries, dict):
            per_domain: Dict[str, np.ndarray] = {}
            for domain_key, arr_in in bucket_boundaries.items():
                arr_d = np.asarray(arr_in, dtype=np.int64)
                if arr_d.ndim != 1 or arr_d.shape[0] != len(
                        DEFAULT_BUCKET_BOUNDARIES):
                    raise ValueError(
                        f"bucket_boundaries[{domain_key!r}] must be a 1D "
                        f"array of length {len(DEFAULT_BUCKET_BOUNDARIES)}; "
                        f"got shape {arr_d.shape}")
                per_domain[domain_key] = arr_d
            self._bucket_boundaries = per_domain
        else:
            arr = np.asarray(bucket_boundaries, dtype=np.int64)
            if arr.ndim != 1 or arr.shape[0] != len(DEFAULT_BUCKET_BOUNDARIES):
                raise ValueError(
                    f"bucket_boundaries must be a 1D array of length "
                    f"{len(DEFAULT_BUCKET_BOUNDARIES)}; got shape {arr.shape}"
                )
            self._bucket_boundaries = arr
        # T31 · is-missing signal. When enabled, null / <=0 values of selected
        # user_int / item_int fids are routed to a reserved vocab slot (vs+1)
        # rather than the padding slot (0) — embedding-level signal. In
        # parallel, a per-fid is-missing bitmap is appended to user_dense —
        # dense-level signal. Both signals are redundant by design (DECEM
        # Trick 3, belt+suspenders) and cost negligible parameters.
        self.enable_is_missing = bool(enable_is_missing)
        # Resolved at feature-plan build time (after self._user_int_cols etc.
        # are populated). None at __init__ entry means "use default fid sets".
        self._is_missing_user_int_fids_arg = is_missing_user_int_fids
        self._is_missing_item_int_fids_arg = is_missing_item_int_fids
        # These three lists are populated by _resolve_is_missing_plan():
        #   _is_missing_user_int_plan: [(ci, offset, fid), ...] for user_int
        #   _is_missing_item_int_plan: [(ci, offset, fid), ...] for item_int
        #   _is_missing_dense_offset: int offset into user_dense where the
        #       bitmap begins. -1 when disabled.
        self._is_missing_user_int_plan: List[Tuple[int, int, int]] = []
        self._is_missing_item_int_plan: List[Tuple[int, int, int]] = []
        self._is_missing_dense_offset: int = -1
        self.enable_history_cvr_features = enable_history_cvr_features
        self.history_cvr_cache_path = history_cvr_cache_path
        self.history_cvr_time_mode = history_cvr_time_mode
        self.history_cvr_cutoff_sec = history_cvr_cutoff_sec
        self.history_cvr_available_lag_sec = history_cvr_available_lag_sec
        self.history_cvr_prior_strength = history_cvr_prior_strength
        self.enable_mature_negative_weighting = enable_mature_negative_weighting
        self.negative_maturity_sec = negative_maturity_sec
        self.immature_negative_weight = immature_negative_weight
        self.negative_maturity_end_ts = negative_maturity_end_ts
        self.dense_log1p_fids: frozenset[int] = (
            frozenset({62, 63, 64, 65, 66})
            if dense_log1p_fids is None
            else dense_log1p_fids
        )
        self.enable_time_of_day_features = enable_time_of_day_features
        self.enable_beijing_time_features = enable_beijing_time_features
        self.enable_beijing_time_v2_features = enable_beijing_time_v2_features
        self._beijing_time_v2_target_day = int(
            datetime.strptime(BEIJING_TARGET_DATE, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
            // 86400
        )
        self._beijing_time_v2_festival_day = int(
            datetime.strptime(BEIJING_FESTIVAL_DATE, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
            // 86400
        )
        self.temporal_weight_alpha = float(temporal_weight_alpha)
        self.temporal_weight_ts_min = temporal_weight_ts_min
        self.temporal_weight_ts_max = temporal_weight_ts_max
        # 5/17 EXP-069 · hour-aware reweight state
        self.hour_weight_min = hour_weight_min
        self.hour_weight_max = hour_weight_max
        self.hour_weight_multiplier = (
            float(hour_weight_multiplier)
            if hour_weight_multiplier is not None else None
        )
        self.target_day_hour_weight_day: Optional[int] = None
        if target_day_hour_weight_date:
            self.target_day_hour_weight_day = int(
                datetime.strptime(target_day_hour_weight_date, "%Y-%m-%d")
                .replace(tzinfo=timezone.utc)
                .timestamp()
                // 86400
            )
        self.target_day_hour_weight_min = target_day_hour_weight_min
        self.target_day_hour_weight_max = target_day_hour_weight_max
        self.target_day_hour_weight_multiplier = (
            float(target_day_hour_weight_multiplier)
            if target_day_hour_weight_multiplier is not None else None
        )
        self.target_day_hour_weight_norm = (
            float(target_day_hour_weight_norm)
            if target_day_hour_weight_norm is not None
            else None
        )
        self.enable_hour_only_features = enable_hour_only_features
        # P1 Item-ID Random Mask (community 0.85+ hint 2026-05-04): gate the
        # mask on shuffle=True so only the *training* DataLoader ever
        # touches it. Valid / holdout / inference construct the dataset
        # with shuffle=False and therefore get id_mask_prob=0 effectively,
        # even if the same parameters are passed through.
        if not 0.0 <= float(id_mask_prob) <= 1.0:
            raise ValueError(
                f"id_mask_prob must be in [0, 1]; got {id_mask_prob}")
        self.id_mask_prob = float(id_mask_prob)
        # Accept both short names ('c') and schema-native names ('seq_c').
        # Matching is done case-insensitively on the trailing letter when a
        # single-character entry is passed in, so users can write the
        # natural "domain c" shorthand from the community discussion.
        if id_mask_seq_domains is None:
            self.id_mask_seq_domains: Optional[frozenset[str]] = None
        else:
            normalized: set[str] = set()
            for d in id_mask_seq_domains:
                s = str(d).strip().lower()
                if not s:
                    continue
                normalized.add(s)
                if not s.startswith('seq_'):
                    normalized.add(f'seq_{s}')
                else:
                    normalized.add(s[len('seq_'):])
            self.id_mask_seq_domains = frozenset(normalized) if normalized else None

        # T23 · Structural disable of specified sequence fids.
        # Any fid in ``disable_seq_fids`` will have its column zeroed out
        # during ``_convert_batch``. Unlike ``id_mask_prob`` (token-level
        # random drop on the training path), this disable is
        # deterministic and applies uniformly to train / valid /
        # inference, so the model sees a consistent "fid-gone" view at
        # both training and evaluation time. Motivated by EXP-025's
        # finding that input-level random mask (15%) caused dev→LB to
        # move in opposite directions (dev +0.0003, LB -0.0042).
        if disable_seq_fids is None:
            self.disable_seq_fids: frozenset[int] = frozenset()
        else:
            fids_parsed: set[int] = set()
            for f in disable_seq_fids:
                try:
                    fids_parsed.add(int(f))
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"disable_seq_fids entries must be integer-like; "
                        f"got {f!r}") from e
            self.disable_seq_fids = frozenset(fids_parsed)
        if self.history_cvr_time_mode not in ('timestamp_cutoff', 'available'):
            raise ValueError(
                "history_cvr_time_mode must be 'timestamp_cutoff' or "
                "'available'")
        if self.history_cvr_available_lag_sec < 0:
            raise ValueError("history_cvr_available_lag_sec must be >= 0")
        self._history_store: Optional[HistoryCVRStore] = None
        if self.enable_history_cvr_features:
            store_cutoff_sec = (
                history_cvr_available_lag_sec
                if history_cvr_time_mode == 'available'
                else history_cvr_cutoff_sec
            )
            self._history_store = HistoryCVRStore(
                history_cvr_cache_path or "",
                cutoff_sec=store_cutoff_sec,
                prior_strength=history_cvr_prior_strength,
            )
        if self.enable_mature_negative_weighting:
            if self.negative_maturity_end_ts is None:
                raise ValueError(
                    "enable_mature_negative_weighting requires "
                    "negative_maturity_end_ts")
            if self.negative_maturity_sec < 0:
                raise ValueError("negative_maturity_sec must be >= 0")
            if self.immature_negative_weight < 0:
                raise ValueError("immature_negative_weight must be >= 0")
        # Out-of-bound statistics:
        #   {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # Build the list of Row Groups.
        if row_groups is not None:
            self._rg_list = list(row_groups)
        else:
            self._rg_list = []
            for f in self._parquet_files:
                pf = pq.ParquetFile(f)
                for i in range(pf.metadata.num_row_groups):
                    self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        if row_group_range is not None and row_groups is None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)

        # Load schema.json.
        self._load_schema(schema_path, seq_max_lens or {})

        # T31 · is-missing · finalize the list of selected fids and bump the
        # vocab_size of each selected fid by +1 (reserving the top slot as the
        # "missing" sentinel), plus extend user_dense_schema with the bitmap
        # columns. Must happen AFTER _load_schema (needs _user_int_cols etc.
        # populated) and BEFORE the _buf_user_dense / _user_int_plan allocation
        # (which both snapshot dims post-adjustment).
        self._resolve_is_missing_plan()

        # ---- Pre-compute column index lookup ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # T31 · now that col_idx exists, link the is-missing plan entries.
        self._link_is_missing_plan_col_idx()

        # ---- Pre-allocate numpy buffers ----
        B = batch_size
        self._buf_user_int = np.zeros((B, self.user_int_schema.total_dim), dtype=np.int64)
        self._buf_item_int = np.zeros((B, self.item_int_schema.total_dim), dtype=np.int64)
        self._buf_user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_lens = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)

        # ---- Pre-compute (col_idx, offset, vocab_size) plans for int columns ----
        self._user_int_plan = []  # [(col_idx, dim, offset, vocab_size), ...]
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            self._user_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim

        # T31 · pre-compute fast-lookup sets of selected fids for use in the
        # _convert_batch hot loop. Derived from the plans built above so
        # fid→fid membership is O(1). Empty when enable_is_missing=False.
        self._is_missing_user_int_fids_set: set[int] = {
            fid for _, _, fid in self._is_missing_user_int_plan
        }
        self._is_missing_item_int_fids_set: set[int] = {
            fid for _, _, fid in self._is_missing_item_int_plan
        }
        # Also pre-compute per-plan (ci, dim, offset, vs, is_selected) tuples
        # to avoid re-hashing fids in the hot path. Map ci → fid once.
        user_int_ci_to_fid = {
            self._col_idx.get(f'user_int_feats_{fid}'): fid
            for fid, _, _ in self._user_int_cols
        }
        item_int_ci_to_fid = {
            self._col_idx.get(f'item_int_feats_{fid}'): fid
            for fid, _, _ in self._item_int_cols
        }
        self._user_int_plan_x = [
            (ci, dim, offset, vs,
             user_int_ci_to_fid.get(ci) in self._is_missing_user_int_fids_set)
            for ci, dim, offset, vs in self._user_int_plan
        ]
        self._item_int_plan_x = [
            (ci, dim, offset, vs,
             item_int_ci_to_fid.get(ci) in self._is_missing_item_int_fids_set)
            for ci, dim, offset, vs in self._item_int_plan
        ]
        # Kept as attributes for use in the hot loop (fid recovery from ci).
        self._ci_to_user_int_fid = user_int_ci_to_fid
        self._ci_to_item_int_fid = item_int_ci_to_fid

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            do_log1p = fid in self.dense_log1p_fids
            self._user_dense_plan.append((ci, dim, offset, do_log1p))
            offset += dim

        # Sequence column plan: {domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        self._seq_plan = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f'{prefix}_{fid}')
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
            ts_ci = self._col_idx.get(f'{prefix}_{ts_fid}') if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}, "
            f"timestamp_min={timestamp_min}, timestamp_max={timestamp_max}, "
            f"count_features={enable_count_features}, "
            f"seq_stats_features={enable_seq_stats_features}, "
            f"time_of_day_features={enable_time_of_day_features}, "
            f"beijing_time_features={enable_beijing_time_features}, "
            f"beijing_time_v2_features={enable_beijing_time_v2_features}, "
            f"history_cvr_features={enable_history_cvr_features}, "
            f"history_cvr_time_mode={history_cvr_time_mode}, "
            f"mature_negative_weighting={enable_mature_negative_weighting}, "
            f"id_mask_prob={self.id_mask_prob}, "
            f"id_mask_seq_domains={sorted(self.id_mask_seq_domains) if self.id_mask_seq_domains else 'all'}, "
            f"disable_seq_fids={sorted(self.disable_seq_fids) if self.disable_seq_fids else 'none'}")

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        """Populate per-group schema information from ``schema_path``."""
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        self._user_int_cols: List[List[int]] = raw['user_int']
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)

        # ---- item_int ----
        self._item_int_cols: List[List[int]] = raw['item_int']
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        # ---- user_dense: [[fid, dim], ...] ----
        self._user_dense_cols: List[List[int]] = raw['user_dense']
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)
        if self.enable_count_features:
            self.user_dense_schema.add(COUNT_FEATURE_FID, 2)
        if self.enable_seq_stats_features:
            self.user_dense_schema.add(SEQ_STATS_FEATURE_FID, 2 * len(raw['seq']))
        if self.enable_time_of_day_features:
            self.user_dense_schema.add(TIME_OF_DAY_FEATURE_FID, TIME_OF_DAY_FEATURE_DIM)
        if self.enable_beijing_time_features:
            self.user_dense_schema.add(BEIJING_TIME_FEATURE_FID, BEIJING_TIME_FEATURE_DIM)
        if self.enable_beijing_time_v2_features:
            self.user_dense_schema.add(
                BEIJING_TIME_V2_FEATURE_FID, BEIJING_TIME_V2_FEATURE_DIM)
        if self.enable_hour_only_features:
            self.user_dense_schema.add(HOUR_ONLY_FEATURE_FID, HOUR_ONLY_FEATURE_DIM)
        self._history_item_offsets: List[Tuple[int, int]] = []
        if self.enable_history_cvr_features:
            if self._history_store is None:
                raise ValueError("history CVR store was not loaded")
            for fid in self._history_store.fids:
                try:
                    offset, length = self.item_int_schema.get_offset_length(int(fid))
                except KeyError as e:
                    raise ValueError(
                        f"history CVR fid {int(fid)} is not in item_int schema"
                    ) from e
                if length != 1:
                    raise ValueError(
                        f"history CVR fid {int(fid)} must be scalar; got "
                        f"length={length}")
                self._history_item_offsets.append((int(fid), int(offset)))
            self.user_dense_schema.add(
                HISTORY_CVR_FEATURE_FID, self._history_store.feature_dim)

        # ---- item_dense (empty) ----
        self.item_dense_schema: FeatureSchema = FeatureSchema()

        # ---- sequence domains ----
        self._seq_cfg: Dict[str, Dict[str, Any]] = raw['seq']
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg['prefix']
            ts_fid = cfg['ts_fid']
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg['features']]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg['features']}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]

            # max_len: from seq_max_lens arg; unspecified domains fall back to 256.
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

    def _resolve_is_missing_plan(self) -> None:
        """T31 · Decide which user_int / item_int fids carry is-missing signal
        and register the bookkeeping.

        Effects (only when ``self.enable_is_missing`` is True):
        - Mutates ``self._user_int_cols`` / ``self._item_int_cols`` in place to
          bump the vocab_size of each selected fid by +1. The new top slot
          (= original_vs) is reserved as the "missing" sentinel.
        - Mutates ``self.user_int_vocab_sizes`` / ``self.item_int_vocab_sizes``
          accordingly (per-column expansion for multi-dim fids is not needed
          since selected fids are scalar; see assert below).
        - Extends ``self.user_dense_schema`` with ``IS_MISSING_FEATURE_FID`` of
          length ``len(user_int_selected) + len(item_int_selected)``.
        - Populates ``self._is_missing_user_int_plan`` /
          ``self._is_missing_item_int_plan`` for fast lookup in
          ``_convert_batch``. Entries are ``(col_idx, user_int_scalar_offset,
          fid)``.
        - Sets ``self._is_missing_dense_offset`` to the starting offset in
          user_dense where the bitmap begins.

        No-op when ``self.enable_is_missing`` is False — preserves the exact
        pre-T31 buffer layout for backward compatibility.
        """
        if not self.enable_is_missing:
            return

        # Default to all scalar user_int fids when caller passes None.
        # For item_int, default to EMPTY set because DECEM only lists user_int
        # fids as high-null. Item_int is opt-in via explicit fid list.
        all_user_scalar = [fid for fid, vs, dim in self._user_int_cols
                           if dim == 1]
        all_item_scalar = [fid for fid, vs, dim in self._item_int_cols
                           if dim == 1]

        if self._is_missing_user_int_fids_arg is None:
            selected_user = set(all_user_scalar)
        else:
            selected_user = set(self._is_missing_user_int_fids_arg)
        if self._is_missing_item_int_fids_arg is None:
            selected_item: set[int] = set()
        else:
            selected_item = set(self._is_missing_item_int_fids_arg)

        # Intersect with actual schema; drop fids not in schema (warn).
        unknown_user = selected_user - set(fid for fid, _, _ in self._user_int_cols)
        unknown_item = selected_item - set(fid for fid, _, _ in self._item_int_cols)
        if unknown_user:
            logging.warning(
                f"[T31] is_missing_user_int_fids: ignoring unknown fids "
                f"{sorted(unknown_user)} (not in schema)")
        if unknown_item:
            logging.warning(
                f"[T31] is_missing_item_int_fids: ignoring unknown fids "
                f"{sorted(unknown_item)} (not in schema)")
        selected_user -= unknown_user
        selected_item -= unknown_item

        # Restrict to scalar dim=1 fids (multi-dim is_missing is ambiguous;
        # would need per-position bitmap which is out of scope for T31).
        non_scalar_user = {fid for fid, vs, dim in self._user_int_cols
                           if fid in selected_user and dim != 1}
        non_scalar_item = {fid for fid, vs, dim in self._item_int_cols
                           if fid in selected_item and dim != 1}
        if non_scalar_user:
            logging.warning(
                f"[T31] is_missing: skipping non-scalar user_int fids "
                f"{sorted(non_scalar_user)} (dim != 1 is out of scope)")
        if non_scalar_item:
            logging.warning(
                f"[T31] is_missing: skipping non-scalar item_int fids "
                f"{sorted(non_scalar_item)} (dim != 1 is out of scope)")
        selected_user -= non_scalar_user
        selected_item -= non_scalar_item

        # Nothing selected → disable cleanly. This avoids a zero-width dense
        # FEATURE_FID add (which would make FeatureSchema silently noop but
        # leave self.enable_is_missing True confusingly).
        if not selected_user and not selected_item:
            logging.warning(
                "[T31] enable_is_missing=True but no valid fids selected; "
                "effectively disabled (no schema / dense changes).")
            self.enable_is_missing = False
            return

        # (1) Bump vocab_size by +1 in both _user_int_cols and
        # user_int_vocab_sizes. The new top slot becomes the "missing"
        # sentinel. For scalar fids (dim=1), the per-column sizes list is
        # also dim=1 so a single-position bump suffices.
        def _bump(cols: List[List[int]], vocab_sizes: List[int],
                  schema: 'FeatureSchema', selected: set[int]) -> None:
            offset = 0
            for i, (fid, vs, dim) in enumerate(cols):
                if fid in selected:
                    assert dim == 1, \
                        f"is_missing supports scalar fids only; fid={fid} "\
                        f"has dim={dim} (should have been filtered out)"
                    cols[i] = [fid, vs + 1, dim]
                    vocab_sizes[offset] = vs + 1
                offset += dim

        _bump(self._user_int_cols, self.user_int_vocab_sizes,
              self.user_int_schema, selected_user)
        _bump(self._item_int_cols, self.item_int_vocab_sizes,
              self.item_int_schema, selected_item)

        # (2) Extend user_dense_schema with bitmap columns at the tail.
        # Important: this add MUST happen AFTER all other enable_* adds
        # in _load_schema so the bitmap sits at a deterministic offset.
        self._is_missing_dense_offset = self.user_dense_schema.total_dim
        n_bitmap = len(selected_user) + len(selected_item)
        self.user_dense_schema.add(IS_MISSING_FEATURE_FID, n_bitmap)

        # (3) Build plan lists for _convert_batch. Use sorted order so the
        # bitmap layout is deterministic (user_int fids first ascending,
        # then item_int fids ascending). The layout is reproducible from
        # (selected_user_sorted, selected_item_sorted) at infer time.
        user_int_offsets: Dict[int, int] = {}
        cur = 0
        for fid, vs, dim in self._user_int_cols:
            user_int_offsets[fid] = cur
            cur += dim
        item_int_offsets: Dict[int, int] = {}
        cur = 0
        for fid, vs, dim in self._item_int_cols:
            item_int_offsets[fid] = cur
            cur += dim

        sorted_user_fids = sorted(selected_user)
        sorted_item_fids = sorted(selected_item)
        # col_idx filled later (after self._col_idx exists); store fid for now.
        # Placeholder entries: (col_idx=-1, int_buf_offset, fid). Will be
        # re-linked after self._col_idx is built.
        self._is_missing_user_int_plan = [
            (-1, user_int_offsets[fid], fid) for fid in sorted_user_fids
        ]
        self._is_missing_item_int_plan = [
            (-1, item_int_offsets[fid], fid) for fid in sorted_item_fids
        ]

        logging.info(
            f"[T31] enable_is_missing=True · user_int fids={sorted_user_fids} "
            f"· item_int fids={sorted_item_fids} · user_dense offset="
            f"{self._is_missing_dense_offset} · bitmap dim={n_bitmap}")

    def _link_is_missing_plan_col_idx(self) -> None:
        """T31 · Resolve (-1, offset, fid) entries in the is-missing plans to
        (col_idx, offset, fid) using ``self._col_idx``.

        Called once after ``self._col_idx`` is populated. Splitting this out
        keeps ``_resolve_is_missing_plan`` runnable before parquet inspection.
        """
        if not self.enable_is_missing:
            return
        self._is_missing_user_int_plan = [
            (self._col_idx.get(f'user_int_feats_{fid}'), off, fid)
            for _, off, fid in self._is_missing_user_int_plan
        ]
        self._is_missing_item_int_plan = [
            (self._col_idx.get(f'item_int_feats_{fid}'), off, fid)
            for _, off, fid in self._is_missing_item_int_plan
        ]

    def __len__(self) -> int:
        # Ceiling per Row Group; this is an upper bound on the true batch count.
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                batch_dict = self._convert_batch(batch)
                if batch_dict is None:
                    continue
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Concatenate the buffered batches, shuffle at the row level, then
        re-slice and yield batch-sized chunks.
        """
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged['label'].shape[0]
        rand_idx = torch.randperm(total_rows) if self.shuffle else torch.arange(total_rows)
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    # ---- Helpers ----

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """Record out-of-bound indices and (optionally) clip them to 0,
        without printing to the console.
        """
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s['count'] += n
            s['max'] = max(s['max'], mx)
            s['min_oob'] = min(s['min_oob'], mn)
        else:
            self._oob_stats[key] = {
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json")

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        """Dump out-of-bound statistics to a file if ``path`` is provided,
        otherwise to ``logging.info``.
        """
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s['min_oob'] >= s['vocab'] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}")
        msg = "\n".join(lines)
        if path:
            with open(path, 'w') as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _init_eda_state(self) -> None:
        """Lazy init EDA accumulator on first batch.

        Layout shared by train EDA (``_update_eda_stats`` · uses labels) and
        test EDA (``_update_test_eda_stats`` · no labels). The label-dependent
        slots stay at 0 in the test-only case.
        """
        self._eda_state = {
            # Q1: user_id / item_id distribution
            'total_samples': 0,
            'total_pos': 0,  # label==1 count (test EDA leaves at 0)
            'user_id_counter': defaultdict(int),  # user_id -> count
            'item_id_counter': defaultdict(int),
            # Q2: per-fid null_rate + null_CVR_gap (for user_int + item_int)
            # Accumulators per fid: {fid: [n_null, n_total, pos_null, pos_notnull]}
            # Test EDA fills [0, 1] only; pos_null/pos_notnull stay at 0.
            'user_int_missing': defaultdict(lambda: [0, 0, 0, 0]),
            'item_int_missing': defaultdict(lambda: [0, 0, 0, 0]),
            # Q5: time_diff reservoir sample (10k samples per seq domain)
            # reservoir[domain] = {'samples': list, 'seen': int}
            'time_diff_reservoir': {
                d: {'samples': [], 'seen': 0}
                for d in self.seq_domains
            },
            # Q4 (bonus): seq true_length reservoir (non-padding token count)
            'seq_length_reservoir': {
                d: {'samples': [], 'seen': 0}
                for d in self.seq_domains
            },
            # Q6 (test EDA only): impression timestamp absolute reservoir.
            # Used to measure train↔test temporal gap without depending on
            # parquet metadata.
            'abs_ts_reservoir': {'samples': [], 'seen': 0},
            # Q7 (test EDA only): per-fid non-null value reservoir, used for
            # train↔test marginal distribution comparison (covariate shift).
            # Only populated when ``self.is_training=False``. Limited to a
            # few hundred fids × 5k values each (~few MB total).
            'user_int_value_reservoir': defaultdict(
                lambda: {'samples': [], 'seen': 0}),
            'item_int_value_reservoir': defaultdict(
                lambda: {'samples': [], 'seen': 0}),
        }

    def _reservoir_extend(
        self,
        entry: Dict[str, Any],
        new_values: "npt.NDArray[np.int64]",
        max_size: int,
    ) -> None:
        """Reservoir sampling · extend entry with new numpy values.

        Entry layout: {'samples': list, 'seen': int}
        Uses classic Algorithm R: for each new value, if reservoir not full add;
        else replace a random existing slot with prob k/seen. Implemented in
        numpy vectorized form for speed.
        """
        samples = entry['samples']
        seen = entry['seen']
        need = max_size - len(samples)
        if need > 0:
            # Phase 1: fill reservoir directly
            take = min(need, len(new_values))
            samples.extend(new_values[:take].tolist())
            seen += take
            remaining = new_values[take:]
        else:
            remaining = new_values
        # Phase 2: random-replace for the rest
        if len(remaining) > 0:
            # For each of `remaining`, pick position in [0, seen+i) and replace
            # if pos < max_size. Vectorize via numpy for speed.
            positions = np.random.randint(
                low=0, high=seen + np.arange(1, len(remaining) + 1),
                size=len(remaining),
            )
            replace_mask = positions < max_size
            replace_positions = positions[replace_mask]
            replace_values = remaining[replace_mask]
            for pos, val in zip(replace_positions, replace_values):
                samples[int(pos)] = int(val)
            seen += len(remaining)
        entry['seen'] = seen

    def _update_id_counters(
        self,
        batch: "pa.RecordBatch",
        st: Dict[str, Any],
        cap: int = 200000,
    ) -> None:
        """Vectorized user_id / item_id frequency counter update.

        Replaces the legacy Python-level loop (``for uid, iid in zip(...)``).
        Uses ``np.unique(return_counts=True)`` per batch then dict-merge into
        the running Counter, capping distinct keys at ``cap``.

        Output is bit-identical to the legacy path because we still cap on
        the running ``len(counter)`` before inserting new keys (existing keys
        are unconditionally incremented).

        Shared by both train EDA (``_update_eda_stats``) and test EDA
        (``_update_test_eda_stats``).
        """
        for col_name, counter_key in (
            ('user_id', 'user_id_counter'),
            ('item_id', 'item_id_counter'),
        ):
            if len(st[counter_key]) >= cap:
                continue
            ci = self._col_idx.get(col_name)
            if ci is None:
                continue
            arr = batch.column(ci)
            try:
                vals = arr.to_numpy(zero_copy_only=False)
            except Exception:
                continue
            if vals.dtype == object:
                vals = np.array([0 if v is None else int(v) for v in vals],
                                dtype=np.int64)
            else:
                vals = vals.astype(np.int64, copy=False)
            uniq, cnts = np.unique(vals, return_counts=True)
            counter = st[counter_key]
            remaining_capacity = cap - len(counter)
            for u, c in zip(uniq.tolist(), cnts.tolist()):
                if u in counter:
                    counter[u] += int(c)
                elif remaining_capacity > 0:
                    counter[u] = int(c)
                    remaining_capacity -= 1

    def _update_eda_stats(
        self,
        batch: "pa.RecordBatch",
        B: int,
        labels: "npt.NDArray[np.int64]",
        user_int: "npt.NDArray[np.int64]",
        item_int: "npt.NDArray[np.int64]",
        timestamps: "npt.NDArray[np.int64]",
    ) -> None:
        """Update EDA accumulators with one batch worth of data.

        Triggered from ``_convert_batch`` when ``enable_eda_dump=True`` and
        ``is_training=True`` (we skip during inference to avoid contamination).

        Collects:
        - Q1: user_id/item_id frequency count (for repetition analysis)
        - Q2: per-fid null_rate and null-vs-notnull CVR gap
        - Q4: reservoir-sampled sequence true lengths per domain
        - Q5: reservoir-sampled time_diff values per sequence domain
        """
        if self._eda_state is None:
            self._init_eda_state()
        st = self._eda_state

        # Q1: user_id / item_id frequency (bounded size; cap at 200k each to
        # avoid unbounded memory growth on the full 1.9M train set).
        # 2026-05-13: vectorized (np.unique → dict merge) replaces Python-level
        # loop. ~50× faster, output is bit-identical.
        if len(st['user_id_counter']) < 200000 or len(st['item_id_counter']) < 200000:
            self._update_id_counters(batch, st, cap=200000)

        # Q1 (always): total sample / pos count
        st['total_samples'] += B
        st['total_pos'] += int(labels.sum())

        # Q2: per-fid null_rate + CVR gap (user_int)
        # A slot is "null" iff the materialized value is 0 (since fill_null(0)
        # + arr[arr<=0]=0 collapsed both null and <=0 into 0 upstream).
        # For dim>1 (multi-hot), we consider the fid "null" iff the entire
        # row's slot is all zeros.
        for ci, dim, offset, _vs in self._user_int_plan:
            fid_vals = user_int[:, offset:offset + dim]
            is_null = (fid_vals == 0).all(axis=1)  # (B,) bool
            n_null = int(is_null.sum())
            n_total = B
            pos_null = int(labels[is_null].sum())
            pos_notnull = int(labels[~is_null].sum())
            acc = st['user_int_missing'][ci]
            acc[0] += n_null
            acc[1] += n_total
            acc[2] += pos_null
            acc[3] += pos_notnull
        for ci, dim, offset, _vs in self._item_int_plan:
            fid_vals = item_int[:, offset:offset + dim]
            is_null = (fid_vals == 0).all(axis=1)
            n_null = int(is_null.sum())
            n_total = B
            pos_null = int(labels[is_null].sum())
            pos_notnull = int(labels[~is_null].sum())
            acc = st['item_int_missing'][ci]
            acc[0] += n_null
            acc[1] += n_total
            acc[2] += pos_null
            acc[3] += pos_notnull

        # Q4 + Q5: per-domain seq true length + time_diff reservoir
        for domain in self.seq_domains:
            _, ts_ci = self._seq_plan[domain]
            if ts_ci is None:
                continue
            ts_col = batch.column(ts_ci)
            ts_offs = ts_col.offsets.to_numpy()
            ts_vals = ts_col.values.to_numpy().astype(np.int64)
            max_len = self._seq_maxlen[domain]
            # Build per-sample true lengths and collect time_diffs.
            lengths = np.zeros(B, dtype=np.int64)
            time_diffs_list: List[np.ndarray] = []
            for i in range(B):
                s = int(ts_offs[i])
                e = int(ts_offs[i + 1])
                if e <= s:
                    continue
                vals = ts_vals[s:s + min(e - s, max_len)]
                # true length = number of non-padding (>0) tokens
                valid = vals > 0
                lengths[i] = int(valid.sum())
                if valid.any():
                    valid_ts = vals[valid]
                    time_diffs_list.append(timestamps[i] - valid_ts)

            # Update length reservoir
            self._reservoir_extend(
                st['seq_length_reservoir'][domain],
                lengths,
                self.eda_reservoir_size,
            )
            # Update time_diff reservoir (flatten all samples' time diffs)
            if time_diffs_list:
                flat_td = np.concatenate(time_diffs_list)
                # Keep only non-negative (skip anomalies where seq_ts > ref_ts)
                flat_td = flat_td[flat_td >= 0]
                if len(flat_td) > 0:
                    self._reservoir_extend(
                        st['time_diff_reservoir'][domain],
                        flat_td,
                        self.eda_reservoir_size,
                    )

    def _update_test_eda_stats(
        self,
        batch: "pa.RecordBatch",
        B: int,
        user_int: "npt.NDArray[np.int64]",
        item_int: "npt.NDArray[np.int64]",
        timestamps: "npt.NDArray[np.int64]",
    ) -> None:
        """Update test-EDA accumulators (no labels available on test set).

        Triggered from ``_convert_batch`` when ``enable_eda_dump=True`` and
        ``is_training=False``. Collects a superset of the train-EDA quantities
        that don't depend on labels:

        - Q1: user_id / item_id frequency (shared cap with train EDA)
        - Q2: per-fid null_rate (no CVR gap · pos_null/pos_notnull stay 0)
        - Q4: per-domain sequence true length reservoir
        - Q5: per-domain time_diff reservoir (positive only)
        - Q6 (test-specific): impression absolute timestamp reservoir →
          measures train↔test temporal gap
        - Q7 (test-specific): per-fid non-null value reservoir for user_int +
          item_int → enables KL-divergence / Wasserstein comparison vs train

        Performance: targeted vectorized ops only. Empirically ~3% wall-time
        overhead on inference (well below the +20% budget set in the design
        doc). The blob is emitted by ``finalize_eda()`` which infer.py
        prints to stdout in base64+gzip form.
        """
        if self._eda_state is None:
            self._init_eda_state()
        st = self._eda_state

        if len(st['user_id_counter']) < 200000 or \
                len(st['item_id_counter']) < 200000:
            self._update_id_counters(batch, st, cap=200000)

        st['total_samples'] += B

        for ci, dim, offset, _vs in self._user_int_plan:
            fid_vals = user_int[:, offset:offset + dim]
            is_null = (fid_vals == 0).all(axis=1)
            n_null = int(is_null.sum())
            acc = st['user_int_missing'][ci]
            acc[0] += n_null
            acc[1] += B
        for ci, dim, offset, _vs in self._item_int_plan:
            fid_vals = item_int[:, offset:offset + dim]
            is_null = (fid_vals == 0).all(axis=1)
            n_null = int(is_null.sum())
            acc = st['item_int_missing'][ci]
            acc[0] += n_null
            acc[1] += B

        for domain in self.seq_domains:
            _, ts_ci = self._seq_plan[domain]
            if ts_ci is None:
                continue
            ts_col = batch.column(ts_ci)
            ts_offs = ts_col.offsets.to_numpy()
            ts_vals = ts_col.values.to_numpy().astype(np.int64)
            max_len = self._seq_maxlen[domain]
            lengths = np.zeros(B, dtype=np.int64)
            time_diffs_list: List[np.ndarray] = []
            for i in range(B):
                s = int(ts_offs[i])
                e = int(ts_offs[i + 1])
                if e <= s:
                    continue
                vals = ts_vals[s:s + min(e - s, max_len)]
                valid = vals > 0
                lengths[i] = int(valid.sum())
                if valid.any():
                    valid_ts = vals[valid]
                    time_diffs_list.append(timestamps[i] - valid_ts)
            self._reservoir_extend(
                st['seq_length_reservoir'][domain],
                lengths,
                self.eda_reservoir_size,
            )
            if time_diffs_list:
                flat_td = np.concatenate(time_diffs_list)
                flat_td = flat_td[flat_td >= 0]
                if len(flat_td) > 0:
                    self._reservoir_extend(
                        st['time_diff_reservoir'][domain],
                        flat_td,
                        self.eda_reservoir_size,
                    )

        # Q6: impression absolute timestamps → train↔test temporal gap
        ts_pos = timestamps[timestamps > 0]
        if ts_pos.size > 0:
            self._reservoir_extend(
                st['abs_ts_reservoir'],
                ts_pos,
                self.eda_reservoir_size,
            )

        # Q7: per-fid non-null value reservoir (covariate shift). Use a
        # smaller cap (~5k) per fid to keep blob size bounded. We pick the
        # FIRST scalar slot per fid as a representative value (multi-hot
        # fids contribute their first non-zero slot).
        per_fid_cap = max(5000, self.eda_reservoir_size // 4)
        for ci, dim, offset, _vs in self._user_int_plan:
            fid_vals = user_int[:, offset:offset + dim]
            row_first_nonzero = self._first_nonzero_per_row(fid_vals)
            if row_first_nonzero.size > 0:
                self._reservoir_extend(
                    st['user_int_value_reservoir'][ci],
                    row_first_nonzero,
                    per_fid_cap,
                )
        for ci, dim, offset, _vs in self._item_int_plan:
            fid_vals = item_int[:, offset:offset + dim]
            row_first_nonzero = self._first_nonzero_per_row(fid_vals)
            if row_first_nonzero.size > 0:
                self._reservoir_extend(
                    st['item_int_value_reservoir'][ci],
                    row_first_nonzero,
                    per_fid_cap,
                )

    @staticmethod
    def _first_nonzero_per_row(
        arr: "npt.NDArray[np.int64]",
    ) -> "npt.NDArray[np.int64]":
        """Return the first non-zero value per row (drops all-zero rows).

        For dim=1 this is just non-zero filter. For multi-hot (dim>1) we
        take the first column's value when the row has any non-zero, which
        is a stable proxy for the fid's "primary" value.
        """
        if arr.ndim == 1 or arr.shape[1] == 1:
            flat = arr.reshape(-1)
            return flat[flat != 0]
        nonzero_mask = (arr != 0).any(axis=1)
        if not nonzero_mask.any():
            return np.empty(0, dtype=np.int64)
        first_col = arr[:, 0]
        return first_col[nonzero_mask]

    def finalize_eda(self) -> Optional[Dict[str, Any]]:
        """Called at end of training · returns EDA summary dict.

        Returns None if EDA was never active (no batches processed with
        ``enable_eda_dump=True`` + ``is_training=True``).

        Output is JSON-serializable (plain dict of scalars + small arrays)
        so it can be embedded directly in training_summary.json.
        """
        if self._eda_state is None:
            return None
        st = self._eda_state
        total = max(st['total_samples'], 1)
        global_cvr = st['total_pos'] / total

        # Q1: user_id / item_id repetition analysis
        def _freq_bucket_stats(counter: Dict[int, int]) -> Dict[str, Any]:
            freqs = np.array(list(counter.values()), dtype=np.int64)
            if len(freqs) == 0:
                return {'distinct': 0, 'freq_1': 0.0, 'freq_2_10': 0.0,
                        'freq_gt10': 0.0, 'max_freq': 0, 'mean_freq': 0.0}
            return {
                'distinct': int(len(freqs)),
                'freq_1': float((freqs == 1).sum() / len(freqs)),
                'freq_2_10': float(((freqs >= 2) & (freqs <= 10)).sum() / len(freqs)),
                'freq_gt10': float((freqs > 10).sum() / len(freqs)),
                'max_freq': int(freqs.max()),
                'mean_freq': float(freqs.mean()),
                'total_observations': int(freqs.sum()),
                'cap_reached': bool(len(freqs) >= 200000),
            }

        # Q2: per-fid missing analysis (user_int + item_int)
        def _missing_summary(
            missing_accum: Dict[int, List[int]],
            plan: List[Tuple[int, int, int, int]],
        ) -> List[Dict[str, Any]]:
            out = []
            # Invert ci → fid via schema: plan entries are (ci, dim, offset, vs);
            # but ci only maps back to col name, not fid. So here we return
            # results keyed by ci and include column name for interpretability.
            # schema_names lookup:
            col_name_by_idx = {v: k for k, v in self._col_idx.items()}
            for ci, _dim, _offset, _vs in plan:
                if ci not in missing_accum:
                    continue
                n_null, n_total, pos_null, pos_notnull = missing_accum[ci]
                n_notnull = n_total - n_null
                cvr_null = (pos_null / n_null) if n_null > 0 else 0.0
                cvr_notnull = (pos_notnull / n_notnull) if n_notnull > 0 else 0.0
                out.append({
                    'col': col_name_by_idx.get(ci, f'ci_{ci}'),
                    'null_rate': float(n_null / n_total),
                    'cvr_null': float(cvr_null),
                    'cvr_notnull': float(cvr_notnull),
                    'cvr_gap': float(cvr_null - cvr_notnull),
                    'n_null': int(n_null),
                    'n_notnull': int(n_notnull),
                })
            # Sort by |cvr_gap| desc so top entries are highest-signal missings
            out.sort(key=lambda x: -abs(x['cvr_gap']))
            return out

        # Q4 + Q5: reservoir → quantiles
        def _quantiles(entry: Dict[str, Any]) -> Dict[str, Any]:
            samples = entry['samples']
            if not samples:
                return {'seen': entry['seen'], 'reservoir_size': 0}
            arr = np.array(samples, dtype=np.int64)
            qs = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
            return {
                'seen': entry['seen'],
                'reservoir_size': len(arr),
                'mean': float(arr.mean()),
                'min': int(arr.min()),
                'max': int(arr.max()),
                'quantiles': {
                    f'p{q}': float(np.percentile(arr, q)) for q in qs
                },
            }

        # Q7: per-fid value-distribution reservoir → quantiles + count
        def _value_dist_summary(
            value_reservoir: Dict[int, Dict[str, Any]],
            plan: List[Tuple[int, int, int, int]],
        ) -> List[Dict[str, Any]]:
            col_name_by_idx = {v: k for k, v in self._col_idx.items()}
            out = []
            for ci, _dim, _offset, _vs in plan:
                if ci not in value_reservoir:
                    continue
                entry = value_reservoir[ci]
                samples = entry['samples']
                if not samples:
                    continue
                arr = np.array(samples, dtype=np.int64)
                qs = [1, 5, 25, 50, 75, 95, 99]
                out.append({
                    'col': col_name_by_idx.get(ci, f'ci_{ci}'),
                    'seen': int(entry['seen']),
                    'reservoir_size': int(len(arr)),
                    'distinct_in_reservoir': int(np.unique(arr).size),
                    'mean': float(arr.mean()),
                    'min': int(arr.min()),
                    'max': int(arr.max()),
                    'quantiles': {
                        f'p{q}': float(np.percentile(arr, q)) for q in qs
                    },
                })
            return out

        return {
            'version': 2,
            'collected_at': 'end_of_training',
            'is_training_path': bool(self.is_training),
            'global': {
                'total_samples_observed': st['total_samples'],
                'total_pos': st['total_pos'],
                'global_cvr': global_cvr,
            },
            'q1_user_id': _freq_bucket_stats(st['user_id_counter']),
            'q1_item_id': _freq_bucket_stats(st['item_id_counter']),
            'q2_user_int_missing_top': _missing_summary(
                st['user_int_missing'], self._user_int_plan)[:20],
            'q2_item_int_missing_top': _missing_summary(
                st['item_int_missing'], self._item_int_plan)[:14],
            'q4_seq_length': {
                d: _quantiles(st['seq_length_reservoir'][d])
                for d in self.seq_domains
            },
            'q5_time_diff': {
                d: _quantiles(st['time_diff_reservoir'][d])
                for d in self.seq_domains
            },
            # Q6 + Q7 are populated by the test-EDA path. They stay
            # near-empty / zero on the train path (where abs_ts_reservoir
            # and *_value_reservoir are not updated).
            'q6_abs_ts': _quantiles(st.get(
                'abs_ts_reservoir', {'samples': [], 'seen': 0})),
            'q7_user_int_value_dist': _value_dist_summary(
                st.get('user_int_value_reservoir', {}), self._user_int_plan),
            'q7_item_int_value_dist': _value_dist_summary(
                st.get('item_int_value_reservoir', {}), self._item_int_plan),
        }

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        """Pad an Arrow ``ListArray`` of ints to shape ``[B, max_len]``.

        Values <= 0 are mapped to 0 (padding). Note: the raw data contains -1
        (missing); currently treated the same way as 0 (padding).

        Returns:
            A tuple ``(padded, lengths)`` where ``padded`` has shape
            ``[B, max_len]`` and ``lengths`` has shape ``[B]``.
        """
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            padded[i, :use_len] = values[start:start + use_len]
            lengths[i] = use_len

        padded[padded <= 0] = 0
        return padded, lengths

    # Backwards-compatible alias kept for bench_raw_dataset.py and other
    # external callers that pre-date the rename. New code should call
    # `_pad_varlen_int_column` directly.
    _pad_varlen_column = _pad_varlen_int_column

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """Pad an Arrow ``ListArray<float>`` to shape ``[B, max_dim]``."""
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_dim), dtype=np.float32)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            padded[i, :use_len] = values[start:start + use_len]

        return padded

    def _filter_result_rows(
        self,
        result: Dict[str, Any],
        keep_mask: "npt.NDArray[np.bool_]",
    ) -> Optional[Dict[str, Any]]:
        """Apply a row-level timestamp filter to every per-row field."""
        if keep_mask.all():
            return result
        if not keep_mask.any():
            return None

        keep_tensor = torch.from_numpy(keep_mask)
        out: Dict[str, Any] = {}
        for key, value in result.items():
            if isinstance(value, torch.Tensor) and value.shape[:1] == keep_tensor.shape:
                out[key] = value[keep_tensor]
            elif key == 'user_id':
                out[key] = [v for v, keep in zip(value, keep_mask) if keep]
            else:
                out[key] = value
        return out

    def _convert_batch(self, batch: "pa.RecordBatch") -> Optional[Dict[str, Any]]:
        """Convert an Arrow RecordBatch into a training-ready dict of tensors."""
        B = batch.num_rows

        # ---- meta ----
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
        keep_mask = np.ones(B, dtype=np.bool_)
        if self.timestamp_min is not None:
            keep_mask &= timestamps >= self.timestamp_min
        if self.timestamp_max is not None:
            keep_mask &= timestamps < self.timestamp_max
        raw_label_types = (
            batch.column(self._col_idx['label_type']).fill_null(0)
            .to_numpy(zero_copy_only=False).astype(np.int64)
            if self.is_training else np.zeros(B, dtype=np.int64)
        )
        if self.is_training:
            labels = (raw_label_types == 2).astype(np.int64)
            click_labels = (raw_label_types >= 1).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)
            click_labels = np.zeros(B, dtype=np.int64)
        sample_weight: Optional[np.ndarray] = None
        if self.is_training and self.enable_mature_negative_weighting:
            if self.negative_maturity_end_ts is None:
                raise ValueError("negative_maturity_end_ts was not set")
            sample_weight = np.ones(B, dtype=np.float32)
            negative_mask = raw_label_types != 2
            immature_mask = (
                negative_mask
                & ((int(self.negative_maturity_end_ts) - timestamps)
                   < int(self.negative_maturity_sec))
            )
            sample_weight[immature_mask] = float(self.immature_negative_weight)
        if (self.is_training
                and self.temporal_weight_alpha > 0.0
                and self.temporal_weight_ts_min is not None
                and self.temporal_weight_ts_max is not None):
            # Recency reweighting: samples closer to the test period get higher
            # weight. weight_i = exp(alpha * norm_ts_i) where norm_ts_i in [0,1].
            # alpha=0 → uniform; alpha=3 → latest 1/3 of the range is ~3× more
            # important than the earliest.
            ts_range = float(self.temporal_weight_ts_max - self.temporal_weight_ts_min)
            if ts_range > 0:
                norm_ts = (timestamps.astype(np.float32) - self.temporal_weight_ts_min) / ts_range
                norm_ts = np.clip(norm_ts, 0.0, 1.0)
                tw = np.exp(self.temporal_weight_alpha * norm_ts).astype(np.float32)
                # Normalize so mean weight ≈ 1 (keeps effective learning rate stable)
                tw /= tw.mean().clip(min=1e-6)
                if sample_weight is None:
                    sample_weight = tw
                else:
                    sample_weight = sample_weight * tw
        # 5/17 EXP-069 · Hour-aware reweight (Beijing time = UTC + 8h).
        # Composes multiplicatively with M41 temporal_weight_alpha if both
        # enabled. Upweights samples in test daypart hours (e.g. 北京 09~14).
        if (self.is_training
                and self.hour_weight_multiplier is not None
                and self.hour_weight_min is not None
                and self.hour_weight_max is not None):
            beijing_hour = ((timestamps + 8 * 3600) // 3600) % 24
            in_hour_window = (
                (beijing_hour >= self.hour_weight_min)
                & (beijing_hour <= self.hour_weight_max)
            )
            hw = np.ones(B, dtype=np.float32)
            hw[in_hour_window] = self.hour_weight_multiplier
            # Normalize so mean weight ≈ 1 (keeps effective LR stable)
            hw /= hw.mean().clip(min=1e-6)
            if sample_weight is None:
                sample_weight = hw
            else:
                sample_weight = sample_weight * hw
        # 5/18 M83 · Exact same-day overlap reweight. This targets a Beijing
        # date + hour range (e.g. 2026-03-04 08~09) and divides by a global
        # train-set normalizer instead of the current batch mean.
        if (self.is_training
                and self.target_day_hour_weight_day is not None
                and self.target_day_hour_weight_min is not None
                and self.target_day_hour_weight_max is not None
                and self.target_day_hour_weight_multiplier is not None):
            bj_ts = timestamps + 8 * 3600
            bj_day = bj_ts // 86400
            bj_hour = (bj_ts // 3600) % 24
            in_target_window = (
                (bj_day == self.target_day_hour_weight_day)
                & (bj_hour >= self.target_day_hour_weight_min)
                & (bj_hour <= self.target_day_hour_weight_max)
            )
            dhw = np.ones(B, dtype=np.float32)
            dhw[in_target_window] = self.target_day_hour_weight_multiplier
            norm = self.target_day_hour_weight_norm or float(dhw.mean())
            dhw /= max(float(norm), 1e-6)
            if sample_weight is None:
                sample_weight = dhw
            else:
                sample_weight = sample_weight * dhw
        user_ids = batch.column(self._col_idx['user_id']).to_pylist()

        # ---- user_int: write into pre-allocated buffer ----
        # Note: null -> 0 (via fill_null), -1 -> 0 (via arr<=0); missing values
        # are treated the same as padding. Features with vs==0 have no vocab
        # information and are forced to 0 on the dataset side so that the
        # model's 1-slot Embedding (created for vs=0) is never indexed out of
        # range.
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        # T31 · collect per-fid missing masks when enabled, for the
        # user_dense bitmap below. Keyed by fid so we don't depend on
        # plan iteration order when filling the bitmap.
        user_int_missing_masks: Optional[Dict[int, np.ndarray]] = (
            {} if self.enable_is_missing else None)
        for ci, dim, offset, vs, is_sel in self._user_int_plan_x:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                # T31 · derive missing mask BEFORE the <=0 → 0 collapse so
                # both null and <=0 are treated as "missing".
                if is_sel:
                    missing = arr <= 0
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('user_int', ci, arr, vs)
                else:
                    arr[:] = 0
                # T31 · route missing rows to the reserved top slot
                # (vs - 1 post-bump) and record the mask for the dense bitmap.
                if is_sel:
                    arr[missing] = vs - 1
                    fid = self._ci_to_user_int_fid.get(ci)
                    if fid is not None:
                        user_int_missing_masks[fid] = missing
                user_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('user_int', ci, padded, vs)
                else:
                    padded[:] = 0
                user_int[:, offset:offset + dim] = padded

        # ---- item_int ----
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        item_int_missing_masks: Optional[Dict[int, np.ndarray]] = (
            {} if self.enable_is_missing else None)
        for ci, dim, offset, vs, is_sel in self._item_int_plan_x:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                if is_sel:
                    missing = arr <= 0
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('item_int', ci, arr, vs)
                else:
                    arr[:] = 0
                if is_sel:
                    arr[missing] = vs - 1
                    fid = self._ci_to_item_int_fid.get(ci)
                    if fid is not None:
                        item_int_missing_masks[fid] = missing
                item_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('item_int', ci, padded, vs)
                else:
                    padded[:] = 0
                item_int[:, offset:offset + dim] = padded

        # ---- user_dense ----
        user_dense = self._buf_user_dense[:B]
        user_dense[:] = 0
        for ci, dim, offset, do_log1p in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            if do_log1p:
                # These columns contain raw count/frequency statistics
                # (magnitude 10^3–10^5). Apply log1p so they share the same
                # numeric scale as pre-normalised embedding columns (fid 61/87).
                padded = np.log1p(np.maximum(padded, 0.0))
            user_dense[:, offset:offset + dim] = padded

        dense_offset = sum(dim for _, dim in self._user_dense_cols)
        if self.enable_count_features:
            # Log-scale synthetic magnitudes so they stay in the same rough
            # numeric regime as the existing dense inputs.
            user_dense[:, dense_offset] = np.log1p(
                (user_int != 0).sum(axis=1).astype(np.float32))
            user_dense[:, dense_offset + 1] = np.log1p(
                (item_int != 0).sum(axis=1).astype(np.float32))
            dense_offset += 2

        result = {
            'user_int_feats': torch.from_numpy(user_int.copy()),
            'user_dense_feats': torch.from_numpy(user_dense.copy()),
            'item_int_feats': torch.from_numpy(item_int.copy()),
            'item_dense_feats': torch.zeros(B, 0, dtype=torch.float32),
            'label': torch.from_numpy(labels),
            'label_click': torch.from_numpy(click_labels),
            'timestamp': torch.from_numpy(timestamps),
            'user_id': user_ids,
            '_seq_domains': self.seq_domains,
        }
        if sample_weight is not None:
            result['sample_weight'] = torch.from_numpy(sample_weight)

        # ---- Sequence features: fused padding directly into the 3D buffer ----
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]

            # Write directly into the pre-allocated 3D buffer.
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            # Fused path: first collect (offsets, values, vocab_size, col_idx)
            # for every side-info column, then fill the buffer in a single pass.
            col_data = []
            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                col_data.append((col.offsets.to_numpy(), col.values.to_numpy(), vs, ci))

            for c, (offs, vals, vs, ci) in enumerate(col_data):
                for i in range(B):
                    s = int(offs[i])
                    e = int(offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    out[i, c, :ul] = vals[s:s + ul]
                    if ul > lengths[i]:
                        lengths[i] = ul

            # Values <= 0 -> 0.
            out[out <= 0] = 0

            # Check out-of-bound values per feature's vocab_size.
            # vs==0 means no vocab info; force the whole slice to 0 so that
            # the model's 1-slot Embedding is never indexed out of range.
            for c, (_, _, vs, ci) in enumerate(col_data):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                else:
                    slice_c[:] = 0

            # ---- T23 · Structural disable of requested sideinfo fids ----
            # Applied uniformly to every row (train/valid/inference). Does
            # NOT zero the entire time step (other side-info columns keep
            # their ids), so the ``lengths`` vector and key_padding_mask
            # derived from it stay correct and we never trigger the
            # all-padding SDPA NaN failure mode.
            if self.disable_seq_fids:
                sideinfo_fids_for_domain = self.sideinfo_fids[domain]
                for c, fid in enumerate(sideinfo_fids_for_domain):
                    if int(fid) in self.disable_seq_fids:
                        out[:, c, :] = 0

            # ---- P1 · Item-ID Random Mask (training path only) ----
            # Gate: require shuffle=True (training DataLoader) + is_training
            # + mask_prob > 0 + (all domains OR this domain explicitly
            # opted-in). Mask is token-level and applied uniformly across
            # all side-info columns of the same (row, token), so a single
            # sequence position is either fully kept or fully dropped
            # rather than losing only some of its features. Padding tokens
            # (out[i, c, t]==0 everywhere) are never selected because we
            # only draw from real non-padding positions, but the mask is
            # still applied on top of the raw data and any position that
            # happened to be padding stays 0. ``domain_seq_mask`` is
            # retained so the downstream time_bucket can share the same
            # keep/drop pattern (otherwise dropped tokens would still
            # expose their recency to the model).
            domain_seq_mask: Optional["npt.NDArray[np.bool_]"] = None
            if (self.is_training and self.shuffle
                    and self.id_mask_prob > 0.0
                    and (self.id_mask_seq_domains is None
                         or domain in self.id_mask_seq_domains)):
                # Only mask non-padding positions (lengths[i] gives the
                # valid prefix length); random uniform per (row, token).
                token_mask = (np.random.rand(B, max_len)
                              < self.id_mask_prob)
                length_mask = (
                    np.arange(max_len)[None, :]
                    < lengths[:, None]
                )
                drop_mask = token_mask & length_mask
                if drop_mask.any():
                    # Broadcast across all side-info columns so the
                    # dropped token's entire feature vector becomes
                    # padding (0), keeping shape/vocab invariants.
                    out[:, :, :][np.broadcast_to(
                        drop_mask[:, None, :], out.shape)] = 0
                    domain_seq_mask = drop_mask

            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())

            # Time bucketing.
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()
                # Pad timestamps into shape (B, max_len).
                ts_padded = np.zeros((B, max_len), dtype=np.int64)
                for i in range(B):
                    s = int(ts_offs[i])
                    e = int(ts_offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    ts_padded[i, :ul] = ts_vals[s:s + ul]

                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)
                # np.searchsorted returns values in [0, len(boundaries)].
                # After +1 the nominal range is [1, len(boundaries)+1];
                # the upper bound only appears when time_diff exceeds the
                # largest boundary (~1 year) and would index past
                # nn.Embedding(NUM_TIME_BUCKETS=len(boundaries)+1).
                # Clip raw result to [0, len(boundaries)-1] so the final
                # bucket id (after +1) stays within [1, len(boundaries)]
                # and is always a valid Embedding index. Time-diffs beyond the
                # largest boundary collapse into the last bucket.
                # T30: use per-run boundaries (defaults to DEFAULT_BUCKET_BOUNDARIES).
                # T38: select per-domain boundaries when self._bucket_boundaries
                # is a dict. Falls back to default ladder if domain not present.
                if isinstance(self._bucket_boundaries, dict):
                    boundaries = self._bucket_boundaries.get(
                        domain, DEFAULT_BUCKET_BOUNDARIES)
                else:
                    boundaries = self._bucket_boundaries
                raw_buckets = np.clip(
                    np.searchsorted(boundaries, time_diff.ravel()),
                    0, len(boundaries) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets

            # P1: also clear time_bucket at masked positions so the model
            # cannot recover the dropped token's recency through the time
            # channel.
            if domain_seq_mask is not None:
                time_bucket[domain_seq_mask] = 0

            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())

        if self.enable_seq_stats_features:
            for domain in self.seq_domains:
                lengths = np.log1p(
                    self._buf_seq_lens[domain][:B].astype(np.float32))
                user_dense[:, dense_offset] = lengths
                dense_offset += 1

                _, ts_ci = self._seq_plan[domain]
                ts_range = np.zeros(B, dtype=np.float32)
                if ts_ci is not None:
                    ts_col = batch.column(ts_ci)
                    ts_offs = ts_col.offsets.to_numpy()
                    ts_vals = ts_col.values.to_numpy()
                    max_len = self._seq_maxlen[domain]
                    for i in range(B):
                        s = int(ts_offs[i])
                        e = int(ts_offs[i + 1])
                        if e <= s:
                            continue
                        vals = ts_vals[s:s + min(e - s, max_len)]
                        vals = vals[vals > 0]
                        if len(vals) > 0:
                            ts_range[i] = float(vals.max() - vals.min())
                user_dense[:, dense_offset] = np.log1p(ts_range)
                dense_offset += 1
            result['user_dense_feats'] = torch.from_numpy(user_dense.copy())

        if self.enable_history_cvr_features:
            if self._history_store is None:
                raise ValueError("history CVR store was not loaded")
            hist_feats = self._history_store.lookup_batch(
                timestamps, item_int, self._history_item_offsets)
            dim = hist_feats.shape[1]
            user_dense[:, dense_offset:dense_offset + dim] = hist_feats
            dense_offset += dim
            result['user_dense_feats'] = torch.from_numpy(user_dense.copy())

        if self.enable_time_of_day_features:
            # Circular encoding of hour-of-day and day-of-week captures
            # diurnal / weekly periodicity without ordinal discontinuities.
            # Timestamps are Unix seconds (UTC); Beijing time = UTC+8, but
            # since the shift is a fixed offset it does not affect periodicity
            # — the model learns phase from data regardless.
            ts_float = timestamps.astype(np.float64)
            hours = ((ts_float // 3600) % 24).astype(np.float32)
            # Day-of-week: days since Unix epoch (1970-01-01 was Thursday=3)
            day_idx = ((ts_float // 86400).astype(np.int64) + 3) % 7
            dow = day_idx.astype(np.float32)
            tod = user_dense[:, dense_offset:dense_offset + TIME_OF_DAY_FEATURE_DIM]
            tod[:, 0] = np.sin(2.0 * np.pi * hours / 24.0)
            tod[:, 1] = np.cos(2.0 * np.pi * hours / 24.0)
            tod[:, 2] = np.sin(2.0 * np.pi * dow / 7.0)
            tod[:, 3] = np.cos(2.0 * np.pi * dow / 7.0)
            tod[:, 4] = ((dow >= 5).astype(np.float32))  # Sat=5, Sun=6
            dense_offset += TIME_OF_DAY_FEATURE_DIM
            result['user_dense_feats'] = torch.from_numpy(user_dense.copy())

        if self.enable_beijing_time_features:
            # Beijing-local DOW/weekend is not equivalent to UTC near midnight.
            bj_ts = timestamps.astype(np.float64) + 8.0 * 3600.0
            bj_hours = ((bj_ts // 3600) % 24).astype(np.float32)
            bj_day_idx = ((bj_ts // 86400).astype(np.int64) + 3) % 7
            bj_dow = bj_day_idx.astype(np.float32)
            bj_weekend = (bj_dow >= 5).astype(np.float32)
            bj = user_dense[:, dense_offset:dense_offset + BEIJING_TIME_FEATURE_DIM]
            bj[:, 0] = np.sin(2.0 * np.pi * bj_hours / 24.0)
            bj[:, 1] = np.cos(2.0 * np.pi * bj_hours / 24.0)
            bj[:, 2] = np.sin(2.0 * np.pi * bj_hours / 12.0)
            bj[:, 3] = np.cos(2.0 * np.pi * bj_hours / 12.0)
            bj[:, 4] = np.sin(2.0 * np.pi * bj_dow / 7.0)
            bj[:, 5] = np.cos(2.0 * np.pi * bj_dow / 7.0)
            bj[:, 6] = bj_weekend
            bj[:, 7] = 1.0 - bj_weekend
            bj[:, 8] = ((bj_hours >= 9) & (bj_hours <= 14)).astype(np.float32)
            bj[:, 9] = ((bj_hours >= 6) & (bj_hours < 9)).astype(np.float32)
            dense_offset += BEIJING_TIME_FEATURE_DIM
            result['user_dense_feats'] = torch.from_numpy(user_dense.copy())

        if self.enable_beijing_time_v2_features:
            bj_ts = timestamps.astype(np.float64) + 8.0 * 3600.0
            bj_hours = ((bj_ts // 3600) % 24).astype(np.float32)
            bj_day = (bj_ts // 86400).astype(np.int64)
            days_to_target = (
                self._beijing_time_v2_target_day - bj_day
            ).astype(np.float32)
            clipped_days = np.clip(days_to_target, -7.0, 7.0)
            hour_to_window = np.zeros_like(bj_hours, dtype=np.float32)
            before_window = bj_hours < 9
            after_window = bj_hours > 14
            hour_to_window[before_window] = 9.0 - bj_hours[before_window]
            hour_to_window[after_window] = bj_hours[after_window] - 14.0
            hour_center_delta = ((bj_hours - 11.5 + 12.0) % 24.0) - 12.0

            v2 = user_dense[
                :, dense_offset:dense_offset + BEIJING_TIME_V2_FEATURE_DIM
            ]
            v2[:, 0] = clipped_days / 7.0
            v2[:, 1] = np.abs(clipped_days) / 7.0
            v2[:, 2] = (bj_day == self._beijing_time_v2_festival_day).astype(np.float32)
            v2[:, 3] = (bj_day == self._beijing_time_v2_target_day).astype(np.float32)
            v2[:, 4] = (bj_day < self._beijing_time_v2_festival_day).astype(np.float32)
            v2[:, 5] = (bj_day > self._beijing_time_v2_target_day).astype(np.float32)
            v2[:, 6] = np.clip(hour_to_window, 0.0, 12.0) / 12.0
            v2[:, 7] = hour_center_delta / 12.0
            v2[:, 8] = (bj_hours < 6).astype(np.float32)
            v2[:, 9] = ((bj_hours >= 6) & (bj_hours < 9)).astype(np.float32)
            v2[:, 10] = ((bj_hours >= 9) & (bj_hours <= 14)).astype(np.float32)
            v2[:, 11] = ((bj_hours > 14) & (bj_hours <= 20)).astype(np.float32)
            v2[:, 12] = (bj_hours > 20).astype(np.float32)
            v2[:, 13] = v2[:, 3] * v2[:, 10]
            dense_offset += BEIJING_TIME_V2_FEATURE_DIM
            result['user_dense_feats'] = torch.from_numpy(user_dense.copy())

        if self.enable_hour_only_features:
            # Hour-of-day only (2 features): captures diurnal periodicity
            # without day-of-week bias. DOW is excluded because training data
            # is heavily skewed toward a single weekday (03-03, ~63%), which
            # causes the model to overfit to that weekday's patterns and hurts
            # generalization to the test set's temporal distribution.
            ts_float = timestamps.astype(np.float64)
            hours = ((ts_float // 3600) % 24).astype(np.float32)
            ho = user_dense[:, dense_offset:dense_offset + HOUR_ONLY_FEATURE_DIM]
            ho[:, 0] = np.sin(2.0 * np.pi * hours / 24.0)
            ho[:, 1] = np.cos(2.0 * np.pi * hours / 24.0)
            dense_offset += HOUR_ONLY_FEATURE_DIM
            result['user_dense_feats'] = torch.from_numpy(user_dense.copy())

        # T31 · is-missing bitmap. Written at _is_missing_dense_offset, which
        # was recorded at schema-build time to match user_dense_schema layout.
        # The bitmap ordering is (user_int_fids ascending) followed by
        # (item_int_fids ascending), matching _is_missing_user_int_plan /
        # _is_missing_item_int_plan sort order.
        if self.enable_is_missing and self._is_missing_dense_offset >= 0:
            bm_off = self._is_missing_dense_offset
            for _, _, fid in self._is_missing_user_int_plan:
                mask = user_int_missing_masks.get(fid) \
                    if user_int_missing_masks is not None else None
                if mask is not None:
                    user_dense[:, bm_off] = mask.astype(np.float32)
                bm_off += 1
            for _, _, fid in self._is_missing_item_int_plan:
                mask = item_int_missing_masks.get(fid) \
                    if item_int_missing_masks is not None else None
                if mask is not None:
                    user_dense[:, bm_off] = mask.astype(np.float32)
                bm_off += 1
            # Commit the updated buffer back to the result. The tensor was
            # already materialized in `result['user_dense_feats']` during the
            # core user_dense step; we overwrite it here so downstream code
            # (trainer._make_model_input) sees the bitmap values too.
            result['user_dense_feats'] = torch.from_numpy(user_dense.copy())

        # ---- Slice-diagnostic side-channel (training/valid only) ----
        # These underscore-prefixed fields are NOT read by trainer's
        # _make_model_input, so the training forward pass is unaffected.
        # They are also gated on is_training=True so the inference path
        # (infer.py with is_training=False) does NOT carry them, keeping
        # the inference contract unchanged.
        if self.is_training:
            result['_diag_user_int_nz'] = torch.from_numpy(
                (user_int != 0).sum(axis=1).astype(np.int32))
            result['_diag_item_int_nz'] = torch.from_numpy(
                (item_int != 0).sum(axis=1).astype(np.int32))
            seq_total_len = np.zeros(B, dtype=np.int32)
            for domain in self.seq_domains:
                seq_total_len += self._buf_seq_lens[domain][:B].astype(np.int32)
            result['_diag_seq_total_len'] = torch.from_numpy(seq_total_len)
            # Re-read raw label_type (0/1/2) for per-class AUC slicing; the
            # 'label' field was already collapsed to binary (label_type==2).
            result['_diag_label_type_raw'] = torch.from_numpy(
                raw_label_types.astype(np.int8))

            # EDA stats collection (runs only when enabled · training path only)
            # Uses the already-materialized user_int / item_int / timestamps /
            # labels buffers to avoid duplicate Arrow → numpy conversion.
            if self.enable_eda_dump:
                self._update_eda_stats(
                    batch=batch,
                    B=B,
                    labels=labels,
                    user_int=user_int,
                    item_int=item_int,
                    timestamps=timestamps,
                )
        else:
            # Inference path: still want EDA when explicitly asked. No labels
            # available so we use the label-free ``_update_test_eda_stats``.
            # Used to study train↔test covariate shift without consuming
            # extra evaluation quota.
            if self.enable_eda_dump:
                self._update_test_eda_stats(
                    batch=batch,
                    B=B,
                    user_int=user_int,
                    item_int=item_int,
                    timestamps=timestamps,
                )

        return self._filter_result_rows(result, keep_mask)


def _rg_ts_range(
    pf_cache: Dict[str, "pq.ParquetFile"],
    file_path: str,
    rg_idx: int,
) -> Optional[Tuple[int, int]]:
    """Return ``(min_ts, max_ts)`` for a single row group.

    Prefers the parquet column statistics (zero IO past the metadata that is
    already loaded); falls back to actually reading the ``timestamp`` column
    of that row group when statistics are missing. ``None`` if the row group
    has no ``timestamp`` column at all.
    """
    pf = pf_cache.get(file_path)
    if pf is None:
        pf = pq.ParquetFile(file_path)
        pf_cache[file_path] = pf
    rg_meta = pf.metadata.row_group(rg_idx)
    ts_col = None
    for c in range(rg_meta.num_columns):
        if rg_meta.column(c).path_in_schema == 'timestamp':
            ts_col = c
            break
    if ts_col is None:
        return None
    stats = rg_meta.column(ts_col).statistics
    if stats is not None and stats.has_min_max:
        return (int(stats.min), int(stats.max))
    arr = pf.read_row_group(rg_idx, columns=['timestamp']).column('timestamp').to_numpy()
    if len(arr) == 0:
        return None
    return (int(arr.min()), int(arr.max()))


def _row_ts_cutoff(
    row_groups: List[Tuple[str, int, int]],
    valid_ratio: float,
) -> Tuple[int, int, int, int, int]:
    """Return row-level timestamp cutoff and exact split counts.

    Returns:
        ``(cutoff_ts, total_rows, train_rows_after, valid_rows_after,
        target_valid_rows)``. Rows equal to the cutoff go to validation, so
        duplicate cutoff timestamps can make ``valid_rows_after`` larger than
        ``target_valid_rows``.
    """
    ts_parts = []
    pf_cache: Dict[str, "pq.ParquetFile"] = {}
    for file_path, rg_idx, _ in row_groups:
        pf = pf_cache.get(file_path)
        if pf is None:
            pf = pq.ParquetFile(file_path)
            pf_cache[file_path] = pf
        arr = pf.read_row_group(
            rg_idx, columns=['timestamp']).column('timestamp').to_numpy()
        if len(arr) > 0:
            ts_parts.append(arr.astype(np.int64, copy=False))
    if not ts_parts:
        raise ValueError("enable_row_time_cutoff found no timestamp rows")
    all_ts = np.concatenate(ts_parts)
    n_valid = max(1, int(len(all_ts) * valid_ratio))
    cutoff_idx = max(0, len(all_ts) - n_valid)
    cutoff = int(np.partition(all_ts, cutoff_idx)[cutoff_idx])
    train_rows_after = int((all_ts < cutoff).sum())
    valid_rows_after = int((all_ts >= cutoff).sum())
    return cutoff, int(len(all_ts)), train_rows_after, valid_rows_after, n_valid


def _row_ts_double_cutoff(
    row_groups: List[Tuple[str, int, int]],
    valid_ratio: float,
    gap_ratio: float,
) -> Tuple[int, int, int, int, int, int, int]:
    """Return two row-level timestamp cutoffs for train / gap / valid.

    Targets row-level quantiles so the resulting split counts match
    ``valid_ratio`` / ``gap_ratio`` exactly (down to ties at the cutoff
    timestamps). This is the row-level analogue of the RG-level gap
    that EXP-026 found ineffective: because each RG in this dataset
    spans ~4 days, sorting RGs by ``max(timestamp)`` does not isolate
    a clean time window; sorting every row by ``timestamp`` does.

    Layout on the time axis::

        [...... train (rows with ts < gap_start_ts) ......]
        [..... gap (gap_start_ts <= ts < valid_start_ts) .....]  (dropped)
        [..... valid (ts >= valid_start_ts) .....]

    Returns:
        Tuple of ``(gap_start_ts, valid_start_ts, total_rows,
        train_rows_after, gap_rows_after, valid_rows_after,
        target_valid_rows)``. Rows equal to ``gap_start_ts`` go to gap;
        rows equal to ``valid_start_ts`` go to valid. Duplicate
        timestamps at the boundaries therefore shift rows forward
        rather than back.
    """
    if valid_ratio <= 0 or gap_ratio <= 0:
        raise ValueError(
            f"_row_ts_double_cutoff requires valid_ratio > 0 and "
            f"gap_ratio > 0; got valid_ratio={valid_ratio}, "
            f"gap_ratio={gap_ratio}")
    ts_parts: List["npt.NDArray[np.int64]"] = []
    pf_cache: Dict[str, "pq.ParquetFile"] = {}
    for file_path, rg_idx, _ in row_groups:
        pf = pf_cache.get(file_path)
        if pf is None:
            pf = pq.ParquetFile(file_path)
            pf_cache[file_path] = pf
        arr = pf.read_row_group(
            rg_idx, columns=['timestamp']).column('timestamp').to_numpy()
        if len(arr) > 0:
            ts_parts.append(arr.astype(np.int64, copy=False))
    if not ts_parts:
        raise ValueError(
            "enable_row_valid_gap found no timestamp rows in the provided "
            "row_groups")
    all_ts = np.concatenate(ts_parts)
    n_total = int(len(all_ts))
    n_valid = max(1, int(n_total * valid_ratio))
    n_gap = max(1, int(n_total * gap_ratio))
    if n_valid + n_gap >= n_total:
        raise ValueError(
            f"valid_ratio ({valid_ratio}) + gap_ratio ({gap_ratio}) "
            f"consume all {n_total} rows; reduce them.")
    # Row-level quantiles: valid = rightmost n_valid rows;
    # gap = the n_gap rows immediately before valid; train = the rest.
    valid_start_idx = max(0, n_total - n_valid)
    gap_start_idx = max(0, valid_start_idx - n_gap)
    # np.partition is O(n); pick the two cutoff values by position.
    sorted_prefix = np.partition(all_ts, [gap_start_idx, valid_start_idx])
    gap_start_ts = int(sorted_prefix[gap_start_idx])
    valid_start_ts = int(sorted_prefix[valid_start_idx])
    if gap_start_ts >= valid_start_ts:
        # Tie case: many rows share the same ts at the quantile. Fall
        # back to the unique-ts fallback that still preserves strict
        # ordering of the three slices.
        uniq = np.unique(all_ts)
        if len(uniq) < 3:
            raise ValueError(
                "Not enough unique timestamps to form a 3-way split; "
                f"got {len(uniq)} unique ts values")
        # Map ratios to quantile indices on the unique-ts axis.
        gap_q = max(0, len(uniq) - max(2, int(len(uniq) * (valid_ratio + gap_ratio))))
        valid_q = max(gap_q + 1, len(uniq) - max(1, int(len(uniq) * valid_ratio)))
        valid_q = min(valid_q, len(uniq) - 1)
        gap_start_ts = int(uniq[gap_q])
        valid_start_ts = int(uniq[valid_q])
        if gap_start_ts >= valid_start_ts:
            raise ValueError(
                f"Could not form a clean 3-way time split; "
                f"gap_start_ts={gap_start_ts}, valid_start_ts={valid_start_ts}")
    train_rows_after = int((all_ts < gap_start_ts).sum())
    gap_rows_after = int(
        ((all_ts >= gap_start_ts) & (all_ts < valid_start_ts)).sum())
    valid_rows_after = int((all_ts >= valid_start_ts).sum())
    return (
        gap_start_ts,
        valid_start_ts,
        n_total,
        train_rows_after,
        gap_rows_after,
        valid_rows_after,
        n_valid,
    )


def _row_groups_ts_range(
    row_groups: List[Tuple[str, int, int]],
) -> Optional[Tuple[int, int]]:
    pf_cache: Dict[str, "pq.ParquetFile"] = {}
    ranges = []
    for file_path, rg_idx, _ in row_groups:
        ts_range = _rg_ts_range(pf_cache, file_path, rg_idx)
        if ts_range is not None:
            ranges.append(ts_range)
    if not ranges:
        return None
    return min(r[0] for r in ranges), max(r[1] for r in ranges)


def _beijing_date_hour_weight_fraction(
    row_groups: List[Tuple[str, int, int]],
    date: str,
    hour_min: int,
    hour_max: int,
    timestamp_min: Optional[int] = None,
    timestamp_max: Optional[int] = None,
) -> Tuple[int, int, float]:
    target_day = int(
        datetime.strptime(date, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        // 86400
    )
    total = 0
    matched = 0
    pf_cache: Dict[str, "pq.ParquetFile"] = {}
    for file_path, rg_idx, _ in row_groups:
        pf = pf_cache.get(file_path)
        if pf is None:
            pf = pq.ParquetFile(file_path)
            pf_cache[file_path] = pf
        ts = pf.read_row_group(rg_idx, columns=['timestamp']).column(
            'timestamp').to_numpy().astype(np.int64)
        if timestamp_min is not None:
            ts = ts[ts >= int(timestamp_min)]
        if timestamp_max is not None:
            ts = ts[ts < int(timestamp_max)]
        if ts.size == 0:
            continue
        bj_ts = ts + 8 * 3600
        bj_day = bj_ts // 86400
        bj_hour = (bj_ts // 3600) % 24
        mask = (
            (bj_day == target_day)
            & (bj_hour >= int(hour_min))
            & (bj_hour <= int(hour_max))
        )
        total += int(ts.size)
        matched += int(mask.sum())
    frac = matched / total if total else 0.0
    return total, matched, frac


def _parse_history_cvr_item_fids(
    raw: str,
    item_int_cols: List[List[int]],
) -> List[int]:
    scalar_fids = [int(fid) for fid, _, dim in item_int_cols if int(dim) == 1]
    if not raw or raw.strip().lower() in ('scalar', 'item_scalar'):
        return scalar_fids
    selected = [int(x.strip()) for x in raw.split(',') if x.strip()]
    scalar_set = set(scalar_fids)
    bad = [fid for fid in selected if fid not in scalar_set]
    if bad:
        raise ValueError(
            "history CVR currently supports scalar item_int fids only; "
            f"invalid fids={bad}, scalar_fids={scalar_fids}")
    return selected


def _table_column_numpy(
    table: "pa.Table",
    name: str,
    dtype: Any,
) -> "npt.NDArray[Any]":
    return (
        table.column(name).combine_chunks().fill_null(0)
        .to_numpy(zero_copy_only=False).astype(dtype, copy=False)
    )


def _accumulate_bin_stats(
    stats: Dict[int, List[float]],
    bins: "npt.NDArray[np.int64]",
    positives: "npt.NDArray[np.float32]",
) -> None:
    if len(bins) == 0:
        return
    uniq, inv = np.unique(bins, return_inverse=True)
    counts = np.bincount(inv).astype(np.float64)
    pos = np.bincount(inv, weights=positives.astype(np.float64))
    for bin_idx, cnt, p in zip(uniq, counts, pos):
        row = stats.setdefault(int(bin_idx), [0.0, 0.0])
        row[0] += float(cnt)
        row[1] += float(p)


def _accumulate_value_bin_stats(
    stats: Dict[Tuple[int, int], List[float]],
    values: "npt.NDArray[np.int64]",
    bins: "npt.NDArray[np.int64]",
    positives: "npt.NDArray[np.float32]",
) -> None:
    if len(values) == 0:
        return
    pairs = np.empty(len(values), dtype=[('value', '<i8'), ('bin', '<i8')])
    pairs['value'] = values
    pairs['bin'] = bins
    uniq, inv = np.unique(pairs, return_inverse=True)
    counts = np.bincount(inv).astype(np.float64)
    pos = np.bincount(inv, weights=positives.astype(np.float64))
    for pair, cnt, p in zip(uniq, counts, pos):
        key = (int(pair['value']), int(pair['bin']))
        row = stats.setdefault(key, [0.0, 0.0])
        row[0] += float(cnt)
        row[1] += float(p)


def _finalize_cumulative_rows(
    rows: List[Tuple[int, float, float]],
    bin_sec: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not rows:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    rows = sorted(rows, key=lambda x: x[0])
    bin_end_ts = np.array(
        [(int(b) + 1) * bin_sec - 1 for b, _, _ in rows],
        dtype=np.int64,
    )
    counts = np.cumsum(
        np.array([cnt for _, cnt, _ in rows], dtype=np.float64)
    ).astype(np.float32)
    positives = np.cumsum(
        np.array([pos for _, _, pos in rows], dtype=np.float64)
    ).astype(np.float32)
    return bin_end_ts, counts, positives


def _build_history_cvr_cache(
    row_groups: List[Tuple[str, int, int]],
    schema_path: str,
    cache_path: str,
    item_fids_raw: str = 'scalar',
    bin_sec: int = 3600,
    cutoff_sec: int = 86400,
    prior_strength: float = 20.0,
    time_mode: str = 'timestamp_cutoff',
    available_lag_sec: int = 0,
    negative_maturity_sec: int = 86400,
    timestamp_min: Optional[int] = None,
    timestamp_max: Optional[int] = None,
) -> List[int]:
    """Build a compact time-cumulative history-CVR sidecar.

    The sidecar includes only the provided training Row Groups. Validation and
    inference rows later perform time-ordered lookups against it, so labels
    from validation/test data are never used. In ``available`` mode, positives
    enter history at ``label_time`` and negatives enter after the maturity
    window, which models delayed feedback more faithfully than event-time bins.
    """
    if bin_sec <= 0:
        raise ValueError("history_cvr_bin_sec must be > 0")
    if cutoff_sec < 0:
        raise ValueError("history_cvr_cutoff_sec must be >= 0")
    if time_mode not in ('timestamp_cutoff', 'available'):
        raise ValueError(
            "history_cvr_time_mode must be 'timestamp_cutoff' or 'available'")
    if available_lag_sec < 0:
        raise ValueError("history_cvr_available_lag_sec must be >= 0")
    if negative_maturity_sec < 0:
        raise ValueError("negative_maturity_sec must be >= 0")

    with open(schema_path, 'r', encoding='utf-8') as f:
        raw_schema = json.load(f)
    selected_fids = _parse_history_cvr_item_fids(
        item_fids_raw, raw_schema['item_int'])
    if not selected_fids:
        raise ValueError("history CVR selected no item fids")

    columns = ['timestamp', 'label_type']
    if time_mode == 'available':
        columns.append('label_time')
    columns += [
        f'item_int_feats_{fid}' for fid in selected_fids
    ]
    global_stats: Dict[int, List[float]] = {}
    per_fid_stats: Dict[int, Dict[Tuple[int, int], List[float]]] = {
        fid: {} for fid in selected_fids
    }
    total_rows = 0
    total_pos = 0.0

    lookup_cutoff_sec = (
        int(available_lag_sec)
        if time_mode == 'available'
        else int(cutoff_sec)
    )
    logging.info(
        f"[P0-C] Building history CVR cache at {cache_path}: "
        f"fids={selected_fids}, bin_sec={bin_sec}, time_mode={time_mode}, "
        f"lookup_cutoff_sec={lookup_cutoff_sec}, "
        f"negative_maturity_sec={negative_maturity_sec}, "
        f"timestamp_min={timestamp_min}, timestamp_max={timestamp_max}, "
        f"source_row_groups={len(row_groups)}")
    pf_cache: Dict[str, "pq.ParquetFile"] = {}
    for file_path, rg_idx, _ in row_groups:
        pf = pf_cache.get(file_path)
        if pf is None:
            pf = pq.ParquetFile(file_path)
            pf_cache[file_path] = pf
        table = pf.read_row_group(rg_idx, columns=columns)
        timestamps = _table_column_numpy(table, 'timestamp', np.int64)
        raw_labels = _table_column_numpy(table, 'label_type', np.int64)
        keep_mask = np.ones(len(timestamps), dtype=np.bool_)
        if timestamp_min is not None:
            keep_mask &= timestamps >= int(timestamp_min)
        if timestamp_max is not None:
            keep_mask &= timestamps < int(timestamp_max)
        if not keep_mask.any():
            continue
        timestamps = timestamps[keep_mask]
        raw_labels = raw_labels[keep_mask]
        positives = (raw_labels == 2).astype(np.float32)
        if time_mode == 'available':
            label_times = _table_column_numpy(
                table, 'label_time', np.int64)[keep_mask]
            available_times = timestamps + int(negative_maturity_sec)
            positive_mask = raw_labels == 2
            positive_available = np.where(
                label_times > 0, label_times, timestamps)
            positive_available = np.maximum(positive_available, timestamps)
            available_times[positive_mask] = positive_available[positive_mask]
            bins = available_times // int(bin_sec)
        else:
            bins = timestamps // int(bin_sec)

        total_rows += int(len(timestamps))
        total_pos += float(positives.sum())
        _accumulate_bin_stats(global_stats, bins.astype(np.int64), positives)

        for fid in selected_fids:
            arr = _table_column_numpy(
                table, f'item_int_feats_{fid}', np.int64).copy()
            arr = arr[keep_mask].copy()
            arr[arr <= 0] = 0
            mask = arr > 0
            _accumulate_value_bin_stats(
                per_fid_stats[fid],
                arr[mask].astype(np.int64, copy=False),
                bins[mask].astype(np.int64, copy=False),
                positives[mask],
            )

    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)

    values: List[int] = []
    fid_offsets = [0]
    value_offsets = [0]
    bin_end_parts: List[np.ndarray] = []
    count_parts: List[np.ndarray] = []
    positive_parts: List[np.ndarray] = []
    for fid in selected_fids:
        grouped: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
        for (value, bin_idx), (cnt, pos) in per_fid_stats[fid].items():
            grouped[int(value)].append((int(bin_idx), float(cnt), float(pos)))
        for value in sorted(grouped):
            values.append(int(value))
            bin_end_ts, counts, positives = _finalize_cumulative_rows(
                grouped[value], bin_sec)
            bin_end_parts.append(bin_end_ts)
            count_parts.append(counts)
            positive_parts.append(positives)
            value_offsets.append(value_offsets[-1] + len(bin_end_ts))
        fid_offsets.append(len(values))

    global_rows = [
        (bin_idx, cnt, pos)
        for bin_idx, (cnt, pos) in global_stats.items()
    ]
    global_bin_end_ts, global_counts, global_positives = (
        _finalize_cumulative_rows(global_rows, bin_sec)
    )
    global_prior = float(total_pos / total_rows) if total_rows > 0 else 0.0

    np.savez_compressed(
        cache_path,
        version=np.array([1], dtype=np.int32),
        fids=np.array(selected_fids, dtype=np.int64),
        fid_offsets=np.array(fid_offsets, dtype=np.int64),
        values=np.array(values, dtype=np.int64),
        value_offsets=np.array(value_offsets, dtype=np.int64),
        bin_end_ts=(
            np.concatenate(bin_end_parts)
            if bin_end_parts else np.zeros(0, dtype=np.int64)
        ),
        counts=(
            np.concatenate(count_parts)
            if count_parts else np.zeros(0, dtype=np.float32)
        ),
        positives=(
            np.concatenate(positive_parts)
            if positive_parts else np.zeros(0, dtype=np.float32)
        ),
        global_bin_end_ts=global_bin_end_ts,
        global_counts=global_counts,
        global_positives=global_positives,
        global_prior=np.array([global_prior], dtype=np.float32),
        cutoff_sec=np.array([lookup_cutoff_sec], dtype=np.int64),
        prior_strength=np.array([prior_strength], dtype=np.float32),
        bin_sec=np.array([bin_sec], dtype=np.int64),
        time_mode=np.array([time_mode]),
        available_lag_sec=np.array([available_lag_sec], dtype=np.int64),
        negative_maturity_sec=np.array([negative_maturity_sec], dtype=np.int64),
        source_rows=np.array([total_rows], dtype=np.int64),
    )
    logging.info(
        f"[P0-C] History CVR cache ready: path={cache_path}, "
        f"source_rows={total_rows}, global_prior={global_prior:.6f}, "
        f"values={len(values)}, entries={value_offsets[-1]}")
    return selected_fids


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 16,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    valid_split_strategy: str = 'filename',
    limit_train_rgs: int = 0,
    limit_valid_rgs: int = 0,
    enable_row_time_cutoff: bool = False,
    enable_row_valid_gap: bool = False,
    fixed_valid_timestamp_min: Optional[int] = None,
    fixed_valid_timestamp_max: Optional[int] = None,
    fixed_train_timestamp_max: Optional[int] = None,
    aux_valid_windows: Optional[List[Tuple[str, int, int]]] = None,
    enable_count_features: bool = False,
    enable_seq_stats_features: bool = False,
    enable_history_cvr_features: bool = False,
    history_cvr_cache_path: Optional[str] = None,
    history_cvr_item_fids: str = 'scalar',
    history_cvr_bin_sec: int = 3600,
    history_cvr_cutoff_sec: int = 86400,
    history_cvr_time_mode: str = 'timestamp_cutoff',
    history_cvr_available_lag_sec: int = 0,
    history_cvr_prior_strength: float = 20.0,
    enable_mature_negative_weighting: bool = False,
    negative_maturity_sec: int = 86400,
    immature_negative_weight: float = 0.0,
    dense_log1p_fids: frozenset[int] | None = None,
    enable_time_of_day_features: bool = False,
    temporal_weight_alpha: float = 0.0,
    # 5/17 EXP-069 · hour-aware reweight (default disabled · None passes through)
    hour_weight_min: Optional[int] = None,
    hour_weight_max: Optional[int] = None,
    hour_weight_multiplier: Optional[float] = None,
    # 5/18 M83 · exact Beijing date+hour reweight with global normalizer.
    target_day_hour_weight_date: str = "",
    target_day_hour_weight_min: Optional[int] = None,
    target_day_hour_weight_max: Optional[int] = None,
    target_day_hour_weight_multiplier: Optional[float] = None,
    enable_beijing_time_features: bool = False,
    enable_beijing_time_v2_features: bool = False,
    enable_hour_only_features: bool = False,
    holdout_ratio: float = 0.0,
    valid_gap_ratio: float = 0.0,
    id_mask_prob: float = 0.0,
    id_mask_seq_domains: Optional[List[str]] = None,
    disable_seq_fids: Optional[List[int]] = None,
    enable_eda_dump: bool = False,
    # 5/16 EXP-056 · weekend-CVR-low daypart filter (M45+):
    # When set, train rows with timestamp < user_train_timestamp_min are
    # dropped (e.g. exclude 02-28/03-01 weekend with CVR 0.024-0.034 ·
    # 5x lower than weekday). Valid set unaffected (still uses tail RGs).
    user_train_timestamp_min: Optional[int] = None,
    eda_reservoir_size: int = 20000,
    bucket_boundaries: Optional[
        Union[np.ndarray, Dict[str, np.ndarray]]
    ] = None,
    # T38 · per-domain time bucket: when enabled, fit log-scale boundaries
    # per seq domain from a sample of train timestamps. Output is a Dict[
    # str, np.ndarray] passed to PCVRParquetDataset.
    enable_per_domain_buckets: bool = False,
    per_domain_bucket_sample_rgs: int = 8,
    enable_is_missing: bool = False,
    is_missing_user_int_fids: Optional[List[int]] = None,
    is_missing_item_int_fids: Optional[List[int]] = None,
    **kwargs: Any,
) -> Tuple[
    DataLoader,
    DataLoader,
    Optional[DataLoader],
    Dict[str, DataLoader],
    PCVRParquetDataset,
]:
    """Create train / valid DataLoaders from raw multi-column Parquet files.

    The validation split is taken as the last ``valid_ratio`` fraction of Row
    Groups. The order in which Row Groups are listed (and therefore which
    fraction becomes "the last") is controlled by ``valid_split_strategy``:

    - ``'filename'`` (default, behavior-preserving): keep Row Groups in
      ``sorted(glob)`` + RG-index order. This is what the original baseline
      does and assumes the parquet shards already happen to be in
      time-ascending order.
    - ``'time'``: read each Row Group's ``timestamp`` column statistics and
      sort Row Groups by ``max(timestamp)`` ascending before splitting. This
      is the right choice when the test set is guaranteed to be later than
      the training set (TAAC 2026 setup) and the parquet write order is not
      known to be time-ordered.

    Either way, a sanity log is emitted with the train/valid timestamp ranges
    and their overlap so the actual split alignment is visible without any
    extra job. When ``enable_row_time_cutoff`` is set, the validation minimum
    timestamp becomes a hard row-level cut: train keeps ``timestamp < cutoff``
    and valid keeps ``timestamp >= cutoff``.

    When ``holdout_ratio > 0`` (ADR-005, requires ``valid_split_strategy=
    'time'``), a *third* disjoint split is carved from the latest Row Groups
    as a secondary validation set. The time-axis layout becomes
    ``[ train | dev (valid_ratio) | holdout (holdout_ratio) ]`` with holdout
    being the closest slice to the future-dated test set. Holdout never
    contributes to ``train_loader`` or the primary ``valid_loader``; the
    caller can pass the returned ``holdout_loader`` to the trainer to obtain
    per-epoch holdout AUC/LogLoss as an independent generalization signal.
    ``enable_row_time_cutoff`` is currently **not** supported in combination
    with ``holdout_ratio > 0`` (would require extending ``_row_ts_cutoff`` to
    emit two cut points; out of scope for this iteration).

    ``fixed_valid_timestamp_min/max`` enables a hand-picked row-level
    validation window. Row Groups that touch the window are loaded, then
    filtered exactly at row level. This is intended for pseudo-public
    backtests such as "train before 2026-03-03 09:49:51, validate on
    2026-03-03 09:49:51~14:49:07" and is mutually exclusive with the
    ratio-based row cutoffs.

    ``aux_valid_windows`` adds one or more extra read-only validation windows
    evaluated by the trainer after primary validation. These loaders never
    drive best-model selection. A window is "clean" only if the current
    training rows end at or before its start; otherwise the metric is logged as
    a leaky diagnostic.

    Returns:
        A tuple ``(train_loader, valid_loader, holdout_loader,
        aux_valid_loaders, train_dataset)``.
        ``holdout_loader`` is ``None`` unless ``holdout_ratio > 0``. The
        ``aux_valid_loaders`` dict is empty unless ``aux_valid_windows`` is
        non-empty. The fifth element is returned so the caller can access the feature
        schema (``user_int_schema``, ``item_int_schema``, ...) needed to
        construct the model.
    """
    random.seed(seed)

    if limit_train_rgs < 0:
        raise ValueError("limit_train_rgs must be >= 0")
    if limit_valid_rgs < 0:
        raise ValueError("limit_valid_rgs must be >= 0")
    if holdout_ratio < 0.0:
        raise ValueError("holdout_ratio must be >= 0")
    if valid_gap_ratio < 0.0:
        raise ValueError("valid_gap_ratio must be >= 0")
    if not 0.0 <= id_mask_prob <= 1.0:
        raise ValueError(
            f"id_mask_prob must be in [0, 1]; got {id_mask_prob}")
    fixed_valid_enabled = (
        fixed_valid_timestamp_min is not None
        or fixed_valid_timestamp_max is not None
        or fixed_train_timestamp_max is not None
    )
    if fixed_valid_enabled:
        if fixed_valid_timestamp_min is None or fixed_valid_timestamp_max is None:
            raise ValueError(
                "fixed validation window requires both "
                "fixed_valid_timestamp_min and fixed_valid_timestamp_max")
        if int(fixed_valid_timestamp_max) <= int(fixed_valid_timestamp_min):
            raise ValueError(
                "fixed_valid_timestamp_max must be greater than "
                "fixed_valid_timestamp_min")
        fixed_train_max = (
            int(fixed_train_timestamp_max)
            if fixed_train_timestamp_max is not None
            else int(fixed_valid_timestamp_min)
        )
        if fixed_train_max > int(fixed_valid_timestamp_min):
            raise ValueError(
                "fixed_train_timestamp_max must be <= "
                "fixed_valid_timestamp_min to avoid train/valid leakage")
        if valid_split_strategy != 'time':
            raise ValueError(
                "fixed validation window requires valid_split_strategy='time' "
                "so Row Group timestamp ranges are available.")
        if enable_row_time_cutoff or enable_row_valid_gap:
            raise ValueError(
                "fixed validation window is mutually exclusive with "
                "enable_row_time_cutoff and enable_row_valid_gap")
        if holdout_ratio > 0.0 or valid_gap_ratio > 0.0:
            raise ValueError(
                "fixed validation window is mutually exclusive with "
                "holdout_ratio and valid_gap_ratio")
    if aux_valid_windows:
        if valid_split_strategy != 'time':
            raise ValueError(
                "aux_valid_windows requires valid_split_strategy='time' so "
                "Row Group timestamp ranges are available.")
        seen_aux_names: set[str] = set()
        for name, start_ts, end_ts in aux_valid_windows:
            if not name:
                raise ValueError("aux validation window name must be non-empty")
            if name in seen_aux_names:
                raise ValueError(
                    f"duplicate aux validation window name: {name}")
            seen_aux_names.add(name)
            if int(end_ts) <= int(start_ts):
                raise ValueError(
                    f"aux validation window {name!r} has end <= start: "
                    f"{start_ts}..{end_ts}")
    if valid_gap_ratio > 0.0 and valid_split_strategy != 'time':
        raise ValueError(
            "valid_gap_ratio > 0 requires valid_split_strategy='time' "
            "(Funnel Redesign P3 / ADR-005 alt C): carving a time-axis "
            "gap between train and valid is only meaningful when Row "
            "Groups are ordered by timestamp.")
    if valid_gap_ratio > 0.0 and enable_row_time_cutoff:
        raise ValueError(
            "valid_gap_ratio > 0 with enable_row_time_cutoff=True is "
            "not supported (two row-level cut points would collide). "
            "Use enable_row_valid_gap=True instead to get the double "
            "row-level cutoff designed for this case.")
    if enable_row_valid_gap:
        if not enable_row_time_cutoff and valid_gap_ratio <= 0.0:
            raise ValueError(
                "enable_row_valid_gap=True requires valid_gap_ratio > 0 "
                "(C6 / T22-fix row-level gap). The gap ratio controls "
                "how many rows between the train tail and dev head are "
                "dropped entirely.")
        if enable_row_time_cutoff:
            raise ValueError(
                "enable_row_valid_gap and enable_row_time_cutoff are "
                "mutually exclusive; enable_row_valid_gap already "
                "subsumes the single-cutoff row-level split.")
        if valid_split_strategy != 'time':
            raise ValueError(
                "enable_row_valid_gap=True requires "
                "valid_split_strategy='time'.")
        if holdout_ratio > 0.0:
            raise ValueError(
                "enable_row_valid_gap=True is not yet compatible with "
                "holdout_ratio > 0 (would require a third row-level "
                "cut). Disable one of them.")
    if holdout_ratio > 0.0:
        if valid_split_strategy != 'time':
            raise ValueError(
                "holdout_ratio > 0 requires valid_split_strategy='time' "
                "(ADR-005): a future-most holdout only makes sense on a "
                "time-sorted RG view.")
        if enable_row_time_cutoff:
            raise ValueError(
                "holdout_ratio > 0 is not yet compatible with "
                "enable_row_time_cutoff (would need a second row-level cut "
                "point). Disable one of them.")
        if valid_ratio + holdout_ratio >= 1.0:
            raise ValueError(
                f"valid_ratio ({valid_ratio}) + holdout_ratio "
                f"({holdout_ratio}) must be < 1.0 to leave room for train.")
    if (valid_ratio + holdout_ratio + valid_gap_ratio) >= 1.0:
        raise ValueError(
            f"valid_ratio ({valid_ratio}) + holdout_ratio "
            f"({holdout_ratio}) + valid_gap_ratio ({valid_gap_ratio}) "
            "must be < 1.0 to leave room for train.")
    if history_cvr_time_mode not in ('timestamp_cutoff', 'available'):
        raise ValueError(
            "history_cvr_time_mode must be 'timestamp_cutoff' or 'available'")
    if history_cvr_bin_sec <= 0:
        raise ValueError("history_cvr_bin_sec must be > 0")
    if history_cvr_cutoff_sec < 0:
        raise ValueError("history_cvr_cutoff_sec must be >= 0")
    if history_cvr_available_lag_sec < 0:
        raise ValueError("history_cvr_available_lag_sec must be >= 0")
    if negative_maturity_sec < 0:
        raise ValueError("negative_maturity_sec must be >= 0")
    if immature_negative_weight < 0:
        raise ValueError("immature_negative_weight must be >= 0")

    import glob as _glob
    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))

    rg_info: List[Tuple[str, int, int]] = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)

    if valid_split_strategy not in ('filename', 'time'):
        raise ValueError(
            f"Unknown valid_split_strategy={valid_split_strategy!r}; "
            "expected 'filename' or 'time'.")

    # Per-RG timestamp range cache, populated lazily for sanity logging and
    # eagerly for `valid_split_strategy='time'`.
    rg_ts_ranges: List[Optional[Tuple[int, int]]] = [None] * total_rgs
    pf_cache: Dict[str, "pq.ParquetFile"] = {}

    if valid_split_strategy == 'time':
        # Cost: at most one metadata lookup + (rare) one column read per RG;
        # for 1000 RGs with statistics this is sub-second.
        for i, (f, rg_idx, _) in enumerate(rg_info):
            rg_ts_ranges[i] = _rg_ts_range(pf_cache, f, rg_idx)
        n_missing = sum(r is None for r in rg_ts_ranges)
        if n_missing:
            raise ValueError(
                "valid_split_strategy='time' requires every row group to "
                "expose a 'timestamp' column with parseable min/max; got "
                f"{n_missing}/{total_rgs} RGs without a timestamp range.")
        order = sorted(
            range(total_rgs),
            key=lambda k: (rg_ts_ranges[k][1], rg_ts_ranges[k][0]),
        )
        rg_info = [rg_info[k] for k in order]
        rg_ts_ranges = [rg_ts_ranges[k] for k in order]
        logging.info(
            f"[T8] valid_split_strategy=time: sorted {total_rgs} RGs "
            f"by max(timestamp) ASC")

    # ADR-005 holdout split: carve the latest RGs as a third disjoint set
    # *before* computing the primary train/valid split. Layout on the time
    # axis (when valid_split_strategy='time' + holdout_ratio>0):
    #   rg_info = [...  train  ...][ dev (valid_ratio) ][ holdout ]
    # Holdout is the closest slice to the future-dated test set and is meant
    # to serve as a covariate-shift-aware generalization probe (EXP-018
    # showed local dev AUC is not a reliable LB proxy).
    n_holdout_rgs = 0
    if holdout_ratio > 0.0:
        n_holdout_rgs = max(1, int(total_rgs * holdout_ratio))
        if n_holdout_rgs >= total_rgs:
            raise ValueError(
                f"holdout_ratio={holdout_ratio} would consume all "
                f"{total_rgs} row groups; reduce it.")

    post_holdout_total = total_rgs - n_holdout_rgs
    base_n_valid_rgs = max(1, int(post_holdout_total * valid_ratio))
    # Funnel Redesign P3: carve an optional time-axis gap between train
    # and dev. Layout with valid_split_strategy='time':
    #   [..... train .....][ gap (dropped) ][ dev (valid_ratio) ][ holdout ]
    # ``gap`` is never loaded by any loader; it exists only to push dev
    # away from the train tail so the "dev vs LB" covariate-shift pattern
    # observed in EXP-024 (dev↑ / LB↓) has a proxy we can measure locally.
    n_gap_rgs = 0
    if valid_gap_ratio > 0.0 and not enable_row_valid_gap:
        n_gap_rgs = max(1, int(post_holdout_total * valid_gap_ratio))
        if n_gap_rgs + base_n_valid_rgs >= post_holdout_total:
            raise ValueError(
                f"valid_gap_ratio={valid_gap_ratio} leaves no room for "
                f"train after reserving {base_n_valid_rgs} dev RGs "
                f"from {post_holdout_total} post-holdout RGs.")
    base_n_train_rgs = post_holdout_total - base_n_valid_rgs - n_gap_rgs

    # train_ratio: use only the first N% of the training Row Groups.
    if train_ratio < 1.0:
        base_n_train_rgs = max(1, int(base_n_train_rgs * train_ratio))
        logging.info(
            f"train_ratio={train_ratio}: using {base_n_train_rgs} "
            "train Row Groups")
    # Note: when valid_gap_ratio>0 the arithmetic below reserves slack
    # (gap + dev) on the right; we only recompute valid when gap==0 to
    # preserve the legacy "train + valid = post_holdout_total" invariant.
    if n_gap_rgs == 0:
        base_n_valid_rgs = post_holdout_total - base_n_train_rgs

    train_start_idx = 0
    train_end_idx = base_n_train_rgs
    gap_start_idx = train_end_idx
    gap_end_idx = train_end_idx + n_gap_rgs
    valid_start_idx = gap_end_idx
    valid_end_idx = post_holdout_total
    holdout_start_idx = post_holdout_total
    holdout_end_idx = total_rgs

    split_train_rows = sum(r[2] for r in rg_info[train_start_idx:train_end_idx])
    split_valid_rows = sum(r[2] for r in rg_info[valid_start_idx:valid_end_idx])
    split_holdout_rows = sum(
        r[2] for r in rg_info[holdout_start_idx:holdout_end_idx])
    split_gap_rows = sum(
        r[2] for r in rg_info[gap_start_idx:gap_end_idx])

    logging.info(f"Row Group split ({valid_split_strategy}): "
                 f"{base_n_train_rgs} train ({split_train_rows} rows), "
                 + (f"{n_gap_rgs} gap (dropped, {split_gap_rows} rows), "
                    if n_gap_rgs else "")
                 + f"{base_n_valid_rgs} valid ({split_valid_rows} rows)"
                 + (f", {n_holdout_rgs} holdout "
                    f"({split_holdout_rows} rows)" if n_holdout_rgs else ""))

    if limit_train_rgs > 0:
        available = train_end_idx - train_start_idx
        if limit_train_rgs < available:
            train_start_idx = train_end_idx - limit_train_rgs
            logging.info(
                f"limit_train_rgs={limit_train_rgs}: using last "
                f"{limit_train_rgs}/{available} train Row Groups closest to "
                "the validation cut")
        else:
            logging.info(
                f"limit_train_rgs={limit_train_rgs}: no-op because only "
                f"{available} train Row Groups are available")

    if limit_valid_rgs > 0:
        available = valid_end_idx - valid_start_idx
        if limit_valid_rgs < available:
            valid_end_idx = valid_start_idx + limit_valid_rgs
            logging.info(
                f"limit_valid_rgs={limit_valid_rgs}: using first "
                f"{limit_valid_rgs}/{available} validation Row Groups")
        else:
            logging.info(
                f"limit_valid_rgs={limit_valid_rgs}: no-op because only "
                f"{available} validation Row Groups are available")

    train_rgs = rg_info[train_start_idx:train_end_idx]
    valid_rgs = rg_info[valid_start_idx:valid_end_idx]
    holdout_rgs = rg_info[holdout_start_idx:holdout_end_idx]
    if not train_rgs:
        raise ValueError(
            "Train Row Group split is empty; adjust valid_ratio, "
            "train_ratio, or limit_train_rgs")
    if not valid_rgs:
        raise ValueError(
            "Validation Row Group split is empty; adjust valid_ratio or "
            "limit_valid_rgs")
    if holdout_ratio > 0.0 and not holdout_rgs:
        raise ValueError(
            "Holdout Row Group split is empty; adjust holdout_ratio")
    train_rows = sum(r[2] for r in train_rgs)
    valid_rows = sum(r[2] for r in valid_rgs)
    holdout_rows = sum(r[2] for r in holdout_rgs)

    if limit_train_rgs > 0 or limit_valid_rgs > 0:
        logging.info(
            "Fast proxy Row Group view: "
            f"{len(train_rgs)} train ({train_rows} rows), "
            f"{len(valid_rgs)} valid ({valid_rows} rows)")

    # ---- T8: train/valid timestamp range sanity log (both strategies) ----
    # For 'filename' we only need 4 boundary RGs to assess the cut alignment,
    # avoiding scanning all 1000 RGs in the default path.
    if valid_split_strategy == 'filename' and not enable_row_time_cutoff:
        sample_idx = sorted({
            train_start_idx,
            max(train_start_idx, train_end_idx - 1),
            valid_start_idx,
            max(valid_start_idx, valid_end_idx - 1),
        })
        for i in sample_idx:
            if 0 <= i < total_rgs and rg_ts_ranges[i] is None:
                f, rg_idx, _ = rg_info[i]
                rg_ts_ranges[i] = _rg_ts_range(pf_cache, f, rg_idx)
    if enable_row_time_cutoff:
        for i in range(train_start_idx, valid_end_idx):
            if rg_ts_ranges[i] is None:
                f, rg_idx, _ = rg_info[i]
                rg_ts_ranges[i] = _rg_ts_range(pf_cache, f, rg_idx)
    train_ts = [
        r for r in rg_ts_ranges[train_start_idx:train_end_idx]
        if r is not None
    ]
    valid_ts = [
        r for r in rg_ts_ranges[valid_start_idx:valid_end_idx]
        if r is not None
    ]
    if train_ts and valid_ts:
        train_min, train_max = min(r[0] for r in train_ts), max(r[1] for r in train_ts)
        valid_min, valid_max = min(r[0] for r in valid_ts), max(r[1] for r in valid_ts)
        # Positive overlap_sec = train extends past valid_min => leakage of
        # future-relative training rows into the valid window. For a clean
        # time-aware split this should be <= 0.
        overlap_sec = train_max - valid_min
        logging.info(
            f"[T8] train ts: [{train_min}, {train_max}]  "
            f"valid ts: [{valid_min}, {valid_max}]  "
            f"overlap_sec (train_max - valid_min, <=0 means clean split): "
            f"{overlap_sec}")
        if overlap_sec > 0 and valid_split_strategy == 'filename':
            logging.warning(
                "[T8] valid_split_strategy='filename' yields ts overlap with "
                "train; consider --valid_split_strategy time for a "
                "time-aware split (TAAC 2026 test set is strictly later "
                "than training).")
        # Funnel P3: expose the gap region's timestamp range so the
        # effective "train→valid distance" can be confirmed in platform
        # logs without re-scanning parquet metadata.
        if n_gap_rgs > 0:
            for i in range(gap_start_idx, gap_end_idx):
                if rg_ts_ranges[i] is None:
                    f, rg_idx, _ = rg_info[i]
                    rg_ts_ranges[i] = _rg_ts_range(pf_cache, f, rg_idx)
            gap_ts = [
                r for r in rg_ts_ranges[gap_start_idx:gap_end_idx]
                if r is not None
            ]
            if gap_ts:
                gap_min = min(r[0] for r in gap_ts)
                gap_max = max(r[1] for r in gap_ts)
                # Expected: train_max < gap_min <= gap_max < valid_min.
                logging.info(
                    f"[P3] gap ts (dropped): [{gap_min}, {gap_max}]  "
                    f"effective train→valid distance (valid_min - train_max) "
                    f"= {valid_min - train_max} sec")

    # ADR-005: log holdout timestamp range + disjointness vs dev/train.
    if holdout_ratio > 0.0:
        for i in range(holdout_start_idx, holdout_end_idx):
            if rg_ts_ranges[i] is None:
                f, rg_idx, _ = rg_info[i]
                rg_ts_ranges[i] = _rg_ts_range(pf_cache, f, rg_idx)
        holdout_ts = [
            r for r in rg_ts_ranges[holdout_start_idx:holdout_end_idx]
            if r is not None
        ]
        if holdout_ts and valid_ts:
            holdout_min = min(r[0] for r in holdout_ts)
            holdout_max = max(r[1] for r in holdout_ts)
            v_max = max(r[1] for r in valid_ts)
            # dev ends at valid_max; holdout starts at holdout_min. Ideally
            # holdout_min >= valid_max (strictly later). Overlap > 0 means
            # the two sets share some RGs' ts ranges and the signal gets
            # contaminated.
            dev_holdout_overlap = v_max - holdout_min
            logging.info(
                f"[ADR-005] holdout ts: [{holdout_min}, {holdout_max}]  "
                f"dev_end={v_max}  "
                f"dev_holdout_overlap_sec (<=0 is clean): "
                f"{dev_holdout_overlap}")
            if dev_holdout_overlap > 0:
                logging.warning(
                    "[ADR-005] dev/holdout timestamp ranges overlap; "
                    "holdout signal may be contaminated. This should not "
                    "happen under valid_split_strategy='time'; check RG "
                    "sorting.")

    train_timestamp_min = None
    train_timestamp_max = None
    valid_timestamp_min = None
    valid_timestamp_max = None
    # 5/16 EXP-056 · merge user-supplied timestamp_min (weekend filter) into
    # train_timestamp_min. enable_row_time_cutoff / valid_gap will overwrite
    # train_timestamp_max / valid_timestamp_min later · but train_timestamp_min
    # only set here (no internal mechanism produces it).
    if user_train_timestamp_min is not None:
        train_timestamp_min = int(user_train_timestamp_min)
    cutoff_train_rows = None
    cutoff_valid_rows = None

    if fixed_valid_enabled:
        train_timestamp_max = (
            int(fixed_train_timestamp_max)
            if fixed_train_timestamp_max is not None
            else int(fixed_valid_timestamp_min)
        )
        valid_timestamp_min = int(fixed_valid_timestamp_min)
        valid_timestamp_max = int(fixed_valid_timestamp_max)
        train_rgs = [
            rg for rg, ts in zip(rg_info, rg_ts_ranges)
            if ts is not None
            and ts[0] < train_timestamp_max
            and (train_timestamp_min is None or ts[1] >= train_timestamp_min)
        ]
        valid_rgs = [
            rg for rg, ts in zip(rg_info, rg_ts_ranges)
            if ts is not None
            and ts[1] >= valid_timestamp_min
            and ts[0] < valid_timestamp_max
        ]
        if not train_rgs or not valid_rgs:
            raise ValueError(
                "Fixed validation window produced an empty train or valid "
                "Row Group view; check timestamp bounds.")
        train_rows = sum(r[2] for r in train_rgs)
        valid_rows = sum(r[2] for r in valid_rgs)
        logging.info(
            "[fixed_valid_window] enabled: "
            f"train_timestamp_min={train_timestamp_min}, "
            f"train_timestamp_max={train_timestamp_max}, "
            f"valid_timestamp_min={valid_timestamp_min}, "
            f"valid_timestamp_max={valid_timestamp_max}")
        logging.info(
            "[fixed_valid_window] Row Groups touching row-level windows: "
            f"{len(train_rgs)} train ({train_rows} pre-filter rows), "
            f"{len(valid_rgs)} valid ({valid_rows} pre-filter rows)")

    if enable_row_time_cutoff:
        candidate_start_idx = train_start_idx
        candidate_end_idx = valid_end_idx
        candidate_rgs = rg_info[candidate_start_idx:candidate_end_idx]
        (
            row_cutoff_ts,
            cutoff_total_rows,
            cutoff_train_rows,
            cutoff_valid_rows,
            cutoff_target_valid_rows,
        ) = _row_ts_cutoff(candidate_rgs, valid_ratio)
        train_timestamp_max = row_cutoff_ts
        valid_timestamp_min = row_cutoff_ts

        train_rgs = [
            rg for rg, ts in zip(candidate_rgs, rg_ts_ranges[candidate_start_idx:candidate_end_idx])
            if ts is not None and ts[0] < row_cutoff_ts
        ]
        valid_rgs = [
            rg for rg, ts in zip(candidate_rgs, rg_ts_ranges[candidate_start_idx:candidate_end_idx])
            if ts is not None and ts[1] >= row_cutoff_ts
        ]
        if not train_rgs or not valid_rgs:
            raise ValueError(
                "Row-level timestamp cutoff produced an empty train or valid "
                "Row Group view; adjust valid_ratio or disable "
                "enable_row_time_cutoff")
        train_rows = sum(r[2] for r in train_rgs)
        valid_rows = sum(r[2] for r in valid_rgs)
        logging.info(
            f"[T8.5] row-level timestamp cutoff enabled: cutoff={row_cutoff_ts}; "
            "train keeps timestamp < cutoff, valid keeps timestamp >= cutoff")
        logging.info(
            "[T8.5] exact row-level target after filtering: "
            f"candidate_rows={cutoff_total_rows}, "
            f"train_rows={cutoff_train_rows}, "
            f"valid_rows={cutoff_valid_rows} "
            f"(target_valid_rows={cutoff_target_valid_rows})")
        logging.info(
            "[T8.5] Row Groups touching row-level split: "
            f"{len(train_rgs)} train ({train_rows} pre-filter rows), "
            f"{len(valid_rgs)} valid ({valid_rows} pre-filter rows)")

    if enable_row_valid_gap:
        # T22-fix / C6 · row-level double cutoff. Unlike RG-level gap
        # (EXP-026 showed it is ineffective because each RG spans ~4
        # days), this walks every row in the candidate range, sorts by
        # timestamp, and picks two cutoffs that exactly slice
        # [train | gap (dropped) | dev] by row count.
        #
        # Layout:
        #   ts < gap_start_ts       → train
        #   gap_start_ts <= ts < valid_start_ts → gap (dropped, no loader)
        #   ts >= valid_start_ts    → dev
        #
        # Gap rows are excluded by the combination of
        # train_timestamp_max = gap_start_ts and valid_timestamp_min =
        # valid_start_ts — the interval between them is never loaded
        # by any DataLoader.
        candidate_start_idx = train_start_idx
        candidate_end_idx = valid_end_idx
        candidate_rgs = rg_info[candidate_start_idx:candidate_end_idx]
        (
            gap_start_ts,
            valid_start_ts,
            cutoff_total_rows,
            cutoff_train_rows,
            cutoff_gap_rows,
            cutoff_valid_rows,
            cutoff_target_valid_rows,
        ) = _row_ts_double_cutoff(
            candidate_rgs, valid_ratio, valid_gap_ratio)
        train_timestamp_max = gap_start_ts
        valid_timestamp_min = valid_start_ts

        train_rgs = [
            rg for rg, ts in zip(
                candidate_rgs,
                rg_ts_ranges[candidate_start_idx:candidate_end_idx])
            if ts is not None and ts[0] < gap_start_ts
        ]
        valid_rgs = [
            rg for rg, ts in zip(
                candidate_rgs,
                rg_ts_ranges[candidate_start_idx:candidate_end_idx])
            if ts is not None and ts[1] >= valid_start_ts
        ]
        if not train_rgs or not valid_rgs:
            raise ValueError(
                "Row-level valid gap cutoff produced an empty train or "
                "valid Row Group view; adjust valid_ratio / "
                "valid_gap_ratio or disable enable_row_valid_gap.")
        train_rows = sum(r[2] for r in train_rgs)
        valid_rows = sum(r[2] for r in valid_rgs)
        logging.info(
            "[C6] row-level valid gap enabled: "
            f"gap_start_ts={gap_start_ts}, valid_start_ts={valid_start_ts}; "
            "train keeps timestamp < gap_start_ts, "
            "gap (dropped) keeps gap_start_ts <= timestamp < valid_start_ts, "
            "valid keeps timestamp >= valid_start_ts")
        logging.info(
            "[C6] exact row-level target after filtering: "
            f"candidate_rows={cutoff_total_rows}, "
            f"train_rows={cutoff_train_rows}, "
            f"gap_rows={cutoff_gap_rows}, "
            f"valid_rows={cutoff_valid_rows} "
            f"(target_valid_rows={cutoff_target_valid_rows})")
        logging.info(
            "[C6] effective train→valid time distance (valid_start_ts - "
            f"gap_start_ts) = {valid_start_ts - gap_start_ts} sec "
            "(> 0 is clean, matches real future gap)")
        logging.info(
            "[C6] Row Groups touching row-level split: "
            f"{len(train_rgs)} train ({train_rows} pre-filter rows), "
            f"{len(valid_rgs)} valid ({valid_rows} pre-filter rows)")

    negative_maturity_end_ts = None
    train_ts_range: Optional[Tuple[int, int]] = None
    if enable_mature_negative_weighting:
        if train_timestamp_max is not None:
            negative_maturity_end_ts = int(train_timestamp_max)
        else:
            train_ts_range = _row_groups_ts_range(train_rgs)
            if train_ts_range is None:
                raise ValueError(
                    "enable_mature_negative_weighting could not determine "
                    "the training timestamp range")
            negative_maturity_end_ts = int(train_ts_range[1])
        logging.info(
            "[DF] mature-negative weighting enabled: "
            f"observation_end_ts={negative_maturity_end_ts}, "
            f"negative_maturity_sec={negative_maturity_sec}, "
            f"immature_negative_weight={immature_negative_weight}")

    if temporal_weight_alpha > 0.0 and train_ts_range is None:
        train_ts_range = _row_groups_ts_range(train_rgs)
        if train_ts_range is None:
            logging.warning(
                "[temporal_weight] could not determine training timestamp "
                "range; temporal weighting disabled")
            temporal_weight_alpha = 0.0
        else:
            logging.info(
                f"[temporal_weight] alpha={temporal_weight_alpha}, "
                f"ts_range=[{train_ts_range[0]}, {train_ts_range[1]}]")

    target_day_hour_weight_norm = None
    if (target_day_hour_weight_date
            and target_day_hour_weight_min is not None
            and target_day_hour_weight_max is not None
            and target_day_hour_weight_multiplier is not None
            and target_day_hour_weight_multiplier > 0):
        dh_total, dh_match, dh_frac = _beijing_date_hour_weight_fraction(
            train_rgs,
            target_day_hour_weight_date,
            int(target_day_hour_weight_min),
            int(target_day_hour_weight_max),
            timestamp_min=train_timestamp_min,
            timestamp_max=train_timestamp_max,
        )
        target_day_hour_weight_norm = (
            1.0 + (float(target_day_hour_weight_multiplier) - 1.0) * dh_frac
        )
        logging.info(
            "[target_day_hour_weight] date=%s hour=[%s,%s] multiplier=%.3f "
            "matched=%d/%d frac=%.6f global_norm=%.6f",
            target_day_hour_weight_date,
            target_day_hour_weight_min,
            target_day_hour_weight_max,
            float(target_day_hour_weight_multiplier),
            dh_match,
            dh_total,
            dh_frac,
            target_day_hour_weight_norm,
        )

    if enable_history_cvr_features:
        if not history_cvr_cache_path:
            raise ValueError(
                "enable_history_cvr_features requires history_cvr_cache_path")
        selected_fids = _build_history_cvr_cache(
            row_groups=train_rgs,
            schema_path=schema_path,
            cache_path=history_cvr_cache_path,
            item_fids_raw=history_cvr_item_fids,
            bin_sec=history_cvr_bin_sec,
            cutoff_sec=history_cvr_cutoff_sec,
            time_mode=history_cvr_time_mode,
            available_lag_sec=history_cvr_available_lag_sec,
            negative_maturity_sec=negative_maturity_sec,
            timestamp_min=train_timestamp_min,
            timestamp_max=train_timestamp_max,
            prior_strength=history_cvr_prior_strength,
        )
        logging.info(
            f"[P0-C] History CVR selected item fids: {selected_fids}")

    # T38 · per-domain time bucket boundaries fitting. When enabled, scan a
    # sample of train Row Groups, gather per-domain time_diff (current ts -
    # seq ts), and fit log-scale boundaries to each domain's distribution.
    # The result is a Dict[str, np.ndarray] passed to PCVRParquetDataset
    # via bucket_boundaries. This addresses EXP-058 §Error C: 4 seq domains
    # have time_diff scales spanning 11x (seq_d p99=56d vs seq_c p99=629d).
    if enable_per_domain_buckets and bucket_boundaries is None:
        per_domain_diffs = _scan_per_domain_time_diffs(
            train_rgs,
            schema_path,
            sample_rgs=per_domain_bucket_sample_rgs,
            timestamp_min=train_timestamp_min,
            timestamp_max=train_timestamp_max,
        )
        fitted_boundaries: Dict[str, np.ndarray] = {}
        for domain, diffs in per_domain_diffs.items():
            fitted = fit_log_scale_boundaries(diffs)
            fitted_boundaries[domain] = fitted
            logging.info(
                f"[T38] per-domain bucket fit · domain={domain} "
                f"n_samples={diffs.size} "
                f"p50={int(np.quantile(diffs[diffs > 0], 0.5)) if (diffs > 0).any() else 0}s "
                f"p99={int(np.quantile(diffs[diffs > 0], 0.99)) if (diffs > 0).any() else 0}s "
                f"first_5_boundaries={fitted[:5].tolist()} "
                f"last_5_boundaries={fitted[-5:].tolist()}")
        if sum(int(diffs.size) for diffs in per_domain_diffs.values()) == 0:
            raise ValueError(
                "[T38] enable_per_domain_buckets=True but all domains got "
                "n_samples=0. Refusing to silently fall back to default "
                "bucket boundaries; check uploaded dataset.py/schema_path and "
                "look for '[T38] resolved timestamp columns from schema'.")
        bucket_boundaries = fitted_boundaries
    elif enable_per_domain_buckets and bucket_boundaries is not None:
        logging.warning(
            "[T38] enable_per_domain_buckets=True but bucket_boundaries "
            "already supplied (likely from --time_bucket_boundaries CLI). "
            "Skipping per-domain fit; using supplied boundaries.")

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_groups=train_rgs,
        clip_vocab=clip_vocab,
        timestamp_min=train_timestamp_min,
        timestamp_max=train_timestamp_max,
        enable_count_features=enable_count_features,
        enable_seq_stats_features=enable_seq_stats_features,
        enable_history_cvr_features=enable_history_cvr_features,
        history_cvr_cache_path=history_cvr_cache_path,
        history_cvr_time_mode=history_cvr_time_mode,
        history_cvr_cutoff_sec=history_cvr_cutoff_sec,
        history_cvr_available_lag_sec=history_cvr_available_lag_sec,
        history_cvr_prior_strength=history_cvr_prior_strength,
        enable_mature_negative_weighting=enable_mature_negative_weighting,
        negative_maturity_sec=negative_maturity_sec,
        immature_negative_weight=immature_negative_weight,
        negative_maturity_end_ts=negative_maturity_end_ts,
        dense_log1p_fids=dense_log1p_fids,
        enable_time_of_day_features=enable_time_of_day_features,
        enable_beijing_time_features=enable_beijing_time_features,
        enable_beijing_time_v2_features=enable_beijing_time_v2_features,
        temporal_weight_alpha=temporal_weight_alpha,
        temporal_weight_ts_min=int(train_ts_range[0]) if temporal_weight_alpha > 0.0 and train_ts_range else None,
        temporal_weight_ts_max=int(train_ts_range[1]) if temporal_weight_alpha > 0.0 and train_ts_range else None,
        # 5/17 EXP-069 · hour-aware reweight (None disables · only train)
        hour_weight_min=hour_weight_min,
        hour_weight_max=hour_weight_max,
        hour_weight_multiplier=hour_weight_multiplier,
        target_day_hour_weight_date=target_day_hour_weight_date,
        target_day_hour_weight_min=target_day_hour_weight_min,
        target_day_hour_weight_max=target_day_hour_weight_max,
        target_day_hour_weight_multiplier=target_day_hour_weight_multiplier,
        target_day_hour_weight_norm=target_day_hour_weight_norm,
        enable_hour_only_features=enable_hour_only_features,
        id_mask_prob=id_mask_prob,
        id_mask_seq_domains=id_mask_seq_domains,
        disable_seq_fids=disable_seq_fids,
        enable_eda_dump=enable_eda_dump,
        eda_reservoir_size=eda_reservoir_size,
        bucket_boundaries=bucket_boundaries,
        enable_is_missing=enable_is_missing,
        is_missing_user_int_fids=is_missing_user_int_fids,
        is_missing_item_int_fids=is_missing_item_int_fids,
    )
    use_cuda = torch.cuda.is_available()
    # EDA accumulation only works when the dataset object is shared with the
    # trainer process (main process can call dataset.finalize_eda()).
    # With num_workers>0 the dataloader workers hold dataset *copies* so the
    # main process would see an empty _eda_state. Force num_workers=0 for EDA
    # runs to keep it simple and robust (throughput cost is acceptable for a
    # one-off diagnostic run).
    if enable_eda_dump and num_workers > 0:
        logging.warning(
            f"[EDA] enable_eda_dump=True forces num_workers 0 "
            f"(was {num_workers}); dataset._eda_state is only updated in the "
            "main process. Expect ~20% throughput hit vs multi-worker run."
        )
        num_workers = 0
    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_cuda, **_train_kw,
    )

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_groups=valid_rgs,
        clip_vocab=clip_vocab,
        timestamp_min=valid_timestamp_min,
        timestamp_max=valid_timestamp_max,
        enable_count_features=enable_count_features,
        enable_seq_stats_features=enable_seq_stats_features,
        enable_history_cvr_features=enable_history_cvr_features,
        history_cvr_cache_path=history_cvr_cache_path,
        history_cvr_time_mode=history_cvr_time_mode,
        history_cvr_cutoff_sec=history_cvr_cutoff_sec,
        history_cvr_available_lag_sec=history_cvr_available_lag_sec,
        history_cvr_prior_strength=history_cvr_prior_strength,
        dense_log1p_fids=dense_log1p_fids,
        enable_time_of_day_features=enable_time_of_day_features,
        enable_beijing_time_features=enable_beijing_time_features,
        enable_beijing_time_v2_features=enable_beijing_time_v2_features,
        enable_hour_only_features=enable_hour_only_features,
        disable_seq_fids=disable_seq_fids,
        bucket_boundaries=bucket_boundaries,
        enable_is_missing=enable_is_missing,
        is_missing_user_int_fids=is_missing_user_int_fids,
        is_missing_item_int_fids=is_missing_item_int_fids,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=0, pin_memory=use_cuda,
    )

    # ADR-005: holdout loader. Disjoint from train/valid; read-only, never
    # shuffled, never drives any weight update. Mirrors valid_dataset
    # construction (same feature flags) so metrics are directly comparable.
    holdout_loader: Optional[DataLoader] = None
    if holdout_ratio > 0.0 and holdout_rgs:
        holdout_dataset = PCVRParquetDataset(
            parquet_path=data_dir,
            schema_path=schema_path,
            batch_size=batch_size,
            seq_max_lens=seq_max_lens,
            shuffle=False,
            buffer_batches=0,
            row_groups=holdout_rgs,
            clip_vocab=clip_vocab,
            timestamp_min=None,
            timestamp_max=None,
            enable_count_features=enable_count_features,
            enable_seq_stats_features=enable_seq_stats_features,
            enable_history_cvr_features=enable_history_cvr_features,
            history_cvr_cache_path=history_cvr_cache_path,
            history_cvr_time_mode=history_cvr_time_mode,
            history_cvr_cutoff_sec=history_cvr_cutoff_sec,
            history_cvr_available_lag_sec=history_cvr_available_lag_sec,
            history_cvr_prior_strength=history_cvr_prior_strength,
            dense_log1p_fids=dense_log1p_fids,
            enable_time_of_day_features=enable_time_of_day_features,
            enable_beijing_time_features=enable_beijing_time_features,
            enable_beijing_time_v2_features=enable_beijing_time_v2_features,
            enable_hour_only_features=enable_hour_only_features,
            disable_seq_fids=disable_seq_fids,
            bucket_boundaries=bucket_boundaries,
            enable_is_missing=enable_is_missing,
            is_missing_user_int_fids=is_missing_user_int_fids,
            is_missing_item_int_fids=is_missing_item_int_fids,
        )
        holdout_loader = DataLoader(
            holdout_dataset, batch_size=None,
            num_workers=0, pin_memory=use_cuda,
        )

    aux_valid_loaders: Dict[str, DataLoader] = {}
    if aux_valid_windows:
        # Prefer the effective row-level cutoff when present. The Row Groups
        # that touch a cutoff can still contain later rows, so their metadata
        # max would falsely mark a genuinely clean aux window as leaky.
        effective_train_max = (
            int(train_timestamp_max)
            if train_timestamp_max is not None
            else None
        )
        if effective_train_max is None:
            actual_train_ts_range = _row_groups_ts_range(train_rgs)
            effective_train_max = (
                int(actual_train_ts_range[1])
                if actual_train_ts_range is not None else None
            )
        for name, start_ts, end_ts in aux_valid_windows:
            aux_rgs = [
                rg for rg, ts in zip(rg_info, rg_ts_ranges)
                if ts is not None
                and ts[1] >= int(start_ts)
                and ts[0] < int(end_ts)
            ]
            if not aux_rgs:
                logging.warning(
                    f"[aux_valid/{name}] window [{start_ts}, {end_ts}) "
                    "has no touching Row Groups; skipping")
                continue
            aux_rows = sum(r[2] for r in aux_rgs)
            clean_flag = (
                effective_train_max is not None
                and effective_train_max <= int(start_ts)
            )
            logging.info(
                f"[aux_valid/{name}] configured: window=[{start_ts}, "
                f"{end_ts}), train_max={effective_train_max}, "
                f"status={'clean' if clean_flag else 'leaky'}, "
                f"touching_rgs={len(aux_rgs)}, pre_filter_rows={aux_rows}")
            aux_dataset = PCVRParquetDataset(
                parquet_path=data_dir,
                schema_path=schema_path,
                batch_size=batch_size,
                seq_max_lens=seq_max_lens,
                shuffle=False,
                buffer_batches=0,
                row_groups=aux_rgs,
                clip_vocab=clip_vocab,
                timestamp_min=int(start_ts),
                timestamp_max=int(end_ts),
                enable_count_features=enable_count_features,
                enable_seq_stats_features=enable_seq_stats_features,
                enable_history_cvr_features=enable_history_cvr_features,
                history_cvr_cache_path=history_cvr_cache_path,
                history_cvr_time_mode=history_cvr_time_mode,
                history_cvr_cutoff_sec=history_cvr_cutoff_sec,
                history_cvr_available_lag_sec=history_cvr_available_lag_sec,
                history_cvr_prior_strength=history_cvr_prior_strength,
                dense_log1p_fids=dense_log1p_fids,
                enable_time_of_day_features=enable_time_of_day_features,
                enable_beijing_time_features=enable_beijing_time_features,
                enable_beijing_time_v2_features=enable_beijing_time_v2_features,
                enable_hour_only_features=enable_hour_only_features,
                disable_seq_fids=disable_seq_fids,
                bucket_boundaries=bucket_boundaries,
                enable_is_missing=enable_is_missing,
                is_missing_user_int_fids=is_missing_user_int_fids,
                is_missing_item_int_fids=is_missing_item_int_fids,
            )
            aux_valid_loaders[name] = DataLoader(
                aux_dataset, batch_size=None,
                num_workers=0, pin_memory=use_cuda,
            )

    if enable_row_time_cutoff:
        logging.info(
            f"Parquet train: {train_rows} pre-filter rows "
            f"(expected {cutoff_train_rows} after row filter), "
            f"valid: {valid_rows} pre-filter rows "
            f"(expected {cutoff_valid_rows} after row filter), "
            f"batch_size={batch_size}, buffer_batches={buffer_batches}")
    else:
        logging.info(
            f"Parquet train: {train_rows} rows, valid: {valid_rows} rows"
            + (f", holdout: {holdout_rows} rows" if holdout_ratio > 0 else "")
            + (f", aux_valid={list(aux_valid_loaders)}"
               if aux_valid_loaders else "")
            + f", batch_size={batch_size}, buffer_batches={buffer_batches}")

    return train_loader, valid_loader, holdout_loader, aux_valid_loaders, train_dataset
