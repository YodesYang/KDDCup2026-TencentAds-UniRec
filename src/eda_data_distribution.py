"""src/eda_data_distribution.py · 5/17 EXP-067 · Train-side EDA for A2/F3/G2.

Goal: collect train-side data needed for cross-reference with test-side EDA
(emitted by src/infer.py::_TestEdaState in next evaluation):

- A2 · seq item id frequency per (domain, fid) for high-vocab fids (vocab > 1M)
       · top-K item ids by frequency (default 50000 per fid)
       · purpose: enable downstream offline analysis (tools/analyze_a2_bucket
         _purity.py) computing the **hash-bucket-level signal/noise** for the
         seq encoder pathway · NOT raw item id overlap (model uses
         seq_hash_vocab=500000 modulo · raw overlap rate is mechanistically
         meaningless in 100% user cold-start scenario)
- F3 · user_id distinct count (bounded in-memory set, no identifier output)
       · purpose: verify "100% user cold start" assumption · compute
         test↔train user_id overlap rate (sanity check · model does not use
         user_id directly · low actionability)
- G2 v2 (5/17 13:00) · per (fid, value) → CVR for low-vocab int fids
       · expands EXP-049's "fid → label MI" to fine-grained (fid, value)
         granularity · interpretable cluster identification
       · purpose: detect strong CVR cluster signals at single-fid level
         (multi-fid interactions covered by EXP-049 user×user CV)

Run as a platform training-style job (uses platform parquet path):
    python src/eda_data_distribution.py \\
        --data_dir <train parquet dir> \\
        --schema_path <schema.json> \\
        --output_dir <result dir> \\
        --max_files 0  # full scan (default for production)

Output: <output_dir>/eda_data_distribution.json containing:
- a2_train_seq_item_freq: Dict[domain_fid, List[(item_id, freq)]]
- a2_train_seq_item_distinct_per_fid: Dict[domain_fid, int]
- f3_train_user_id_distinct: int (bounded distinct count)
- f3_user_id_cap: int
- g2_top_fid_values_by_signal: List[Dict] · top (fid, value) pairs by
  (count × |cvr_gap|), each row interpretable as "section: fid: value: cvr"
- meta: scan stats (total_rows, total_files, wall_time_sec)

Memory bound: ~1-2GB (bounded user ids + 4 x 250k seq item ids + ~10k fid_value
pairs). Wall time: ~30 min on full train (1044 RGs).

Downstream offline scripts:
- tools/analyze_a2_bucket_purity.py: cross-reference A2 train + A2 test
  blobs · compute hash-bucket-level signal/noise breakdown · output
  per-(domain, fid) "pure_signal_rate" + "bucket_collision_noise_rate".
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("eda_data_distribution")


# ============================================================================
# Schema parsing helpers
# ============================================================================

def _load_schema(schema_path: str) -> Dict[str, Any]:
    """Parse schema.json · same format as src/dataset.py."""
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_seq_item_id_plan(
    schema: Dict[str, Any],
    min_vocab: int = 1_000_000,
) -> Dict[str, List[Tuple[int, int, str]]]:
    """For each seq domain, list (fid, vocab, column_name) for high-vocab fids.

    column_name = f"{prefix}_{fid}" (matches platform parquet column naming).
    Mirrors src/infer.py::_TestEdaState seq_item_fid_plan structure but adds
    the actual parquet column name (which is what we read here · vs the
    batch tensor slot index used downstream).
    """
    out: Dict[str, List[Tuple[int, int, str]]] = {}
    seq_cfg = schema.get("seq", {}) or {}
    for domain in ("seq_a", "seq_b", "seq_c", "seq_d"):
        cfg = seq_cfg.get(domain)
        if not cfg:
            continue
        prefix = cfg.get("prefix")
        ts_fid = cfg.get("ts_fid")
        features = cfg.get("features", [])
        plan_list: List[Tuple[int, int, str]] = []
        for fid, vocab in features:
            if fid == ts_fid:
                continue  # skip timestamp fid
            if int(vocab) > min_vocab:
                col_name = f"{prefix}_{fid}"
                plan_list.append((int(fid), int(vocab), col_name))
        if plan_list:
            out[domain] = plan_list
    return out


def _build_low_vocab_int_plan(
    schema: Dict[str, Any],
    section: str,  # "user_int" or "item_int"
    max_vocab: int = 256,
) -> List[Tuple[int, str, int]]:
    """For G2: pick low-vocab int fids (vocab ≤ max_vocab) for clustering.

    Returns list of (fid, column_name, vocab). Low-vocab fids are good
    cluster anchors because their values map to interpretable categories
    (gender / device / region / etc.) without high cardinality blowup.
    """
    out: List[Tuple[int, str, int]] = []
    items = schema.get(section, [])
    for entry in items:
        if len(entry) < 2:
            continue
        fid, vocab = int(entry[0]), int(entry[1])
        if vocab <= max_vocab:
            col_name = f"{section}_feats_{fid}"
            out.append((fid, col_name, vocab))
    return out


# ============================================================================
# Streaming accumulators
# ============================================================================

class _A2Accumulator:
    """A2 · per (domain, fid) item id frequency for high-vocab fids."""

    def __init__(
        self,
        seq_item_id_plan: Dict[str, List[Tuple[int, int, str]]],
        top_k: int = 50_000,
    ) -> None:
        self.plan = seq_item_id_plan
        self.top_k = top_k
        # (domain, fid) → Counter (item_id → freq)
        self.counter: Dict[Tuple[str, int], "Counter[int]"] = defaultdict(Counter)

    def update(self, table: "pq.Table") -> None:
        for domain, fid_list in self.plan.items():
            for fid, _vocab, col_name in fid_list:
                if col_name not in table.schema.names:
                    continue
                col = table.column(col_name)
                # ListArray: combine chunks for uniform access
                if hasattr(col, "combine_chunks"):
                    col = col.combine_chunks()
                vals = col.values.to_numpy().astype(np.int64)
                vals = vals[vals > 0]  # drop padding
                if vals.size == 0:
                    continue
                key = (domain, fid)
                # Bound memory: cap unique values at 5x top_k
                hard_cap = self.top_k * 5
                counter = self.counter[key]
                # bulk update via numpy unique
                uniq, counts = np.unique(vals, return_counts=True)
                for u, c in zip(uniq, counts):
                    u_int = int(u)
                    if u_int in counter:
                        counter[u_int] += int(c)
                    elif len(counter) < hard_cap:
                        counter[u_int] = int(c)

    def finalize(self) -> Dict[str, Any]:
        out_freq: Dict[str, List[List[int]]] = {}
        out_distinct: Dict[str, int] = {}
        for (domain, fid), counter in self.counter.items():
            top = counter.most_common(self.top_k)
            key = f"{domain}_fid_{fid}"
            out_freq[key] = [[int(item_id), int(freq)] for item_id, freq in top]
            out_distinct[key] = len(counter)
        return {
            "a2_train_seq_item_freq": out_freq,
            "a2_train_seq_item_distinct_per_fid": out_distinct,
        }


class _F3Accumulator:
    """F3 · user_id distinct count without publishing identifiers."""

    def __init__(self, user_id_cap: int = 500_000) -> None:
        self.user_id_cap = user_id_cap
        self.user_id_set: set = set()
        self.distinct_count = 0  # incremented BEFORE cap check, for true count

    def update(self, table: "pq.Table") -> None:
        if "user_id" not in table.schema.names:
            return
        uids = table.column("user_id").to_pylist()
        for uid in uids:
            uid_str = str(uid)
            if uid_str not in self.user_id_set:
                self.distinct_count += 1
                if len(self.user_id_set) < self.user_id_cap:
                    self.user_id_set.add(uid_str)

    def finalize(self) -> Dict[str, Any]:
        return {
            "f3_train_user_id_distinct": self.distinct_count,
            "f3_user_id_cap": self.user_id_cap,
        }


class _G2Accumulator:
    """G2 v2 (5/17 13:00) · per (fid, value) → CVR for low-vocab int fids.

    Replaces the original hash-bucket cluster approach (5/17 09:30 v1) which
    produced uninterpretable cluster ids. v2 instead expands EXP-049's "fid →
    label MI" to fine-grained "(fid, value) → CVR" granularity:

    - For each low-vocab fid (vocab ≤ max_vocab) in user_int + item_int
    - For each unique value seen in that fid
    - Track (count, label_sum) and compute cvr · cvr_gap_vs_global · signal

    Output: top N (fid, value) pairs by signal volume (count × |cvr_gap|).
    Each row is interpretable: e.g. "user_int_feats_50 = 1 has CVR 0.18 vs
    global 0.05 · count 1.2M · signal 0.157" → strong actionable cluster.

    Compared to original v1 hash-bucket approach:
    - Pros: cluster id = (fid, value) tuple is human-readable
    - Cons: only single-fid; misses multi-fid interactions (but EXP-049 user×
      user CV correlation already covers user-side; item-side covered by EDA
      heuristics)
    """

    def __init__(
        self,
        user_int_plan: List[Tuple[int, str, int]],
        item_int_plan: List[Tuple[int, str, int]],
        top_n: int = 200,
        min_count: int = 100,
    ) -> None:
        self.user_int_plan = user_int_plan
        self.item_int_plan = item_int_plan
        self.top_n = top_n
        self.min_count = min_count
        # ('user_int', fid, value) → [count, label_sum]
        self.fid_value_stats: Dict[
            Tuple[str, int, int], List[int]
        ] = defaultdict(lambda: [0, 0])
        self.global_count = 0
        self.global_label_sum = 0

    def _update_section(
        self,
        table: "pq.Table",
        plan: List[Tuple[int, str, int]],
        section: str,
        binary_labels: np.ndarray,
    ) -> None:
        """Update fid_value_stats for one section (user_int or item_int)."""
        n = table.num_rows
        for fid, col_name, _vocab in plan:
            if col_name not in table.schema.names:
                continue
            col = table.column(col_name)
            if hasattr(col, "combine_chunks"):
                col = col.combine_chunks()
            try:
                vals_arr = col.to_numpy().astype(np.int64)
            except (ValueError, AttributeError):
                continue
            if vals_arr.ndim == 1:
                fid_vals = vals_arr
            else:
                # Multi-valued: take first nonzero (matches Q7 logic)
                fid_vals = np.zeros(n, dtype=np.int64)
                for i in range(n):
                    row = vals_arr[i] if i < len(vals_arr) else None
                    if row is not None and len(row) > 0:
                        nz = row[row != 0]
                        if len(nz) > 0:
                            fid_vals[i] = nz[0]
            # Bulk update with numpy unique to avoid per-row dict ops
            uniq, inv_idx = np.unique(fid_vals, return_inverse=True)
            for u_idx, u_val in enumerate(uniq):
                mask = inv_idx == u_idx
                cnt = int(mask.sum())
                ls = int(binary_labels[mask].sum())
                key = (section, int(fid), int(u_val))
                entry = self.fid_value_stats[key]
                entry[0] += cnt
                entry[1] += ls

    def update(self, table: "pq.Table") -> None:
        if "label_type" not in table.schema.names:
            return
        n = table.num_rows
        if n == 0:
            return
        labels = table.column("label_type").to_numpy().astype(np.int64)
        binary_labels = (labels == 2).astype(np.int64)

        self.global_count += n
        self.global_label_sum += int(binary_labels.sum())

        self._update_section(
            table, self.user_int_plan, "user_int", binary_labels)
        self._update_section(
            table, self.item_int_plan, "item_int", binary_labels)

    def finalize(self) -> Dict[str, Any]:
        global_cvr = (self.global_label_sum / self.global_count
                      if self.global_count else 0.0)
        # For each (section, fid, value): compute CVR + gap + signal
        scored: List[Tuple[str, int, int, float, int, int, float]] = []
        for (section, fid, val), (cnt, ls) in self.fid_value_stats.items():
            if cnt < self.min_count:
                continue
            cvr = ls / cnt
            gap = cvr - global_cvr
            signal = cnt * abs(gap)
            scored.append((section, fid, val, cvr, cnt, ls, signal))
        scored.sort(key=lambda x: -x[6])
        top = scored[: self.top_n]

        return {
            "g2_global_cvr": float(global_cvr),
            "g2_global_count": int(self.global_count),
            "g2_global_label_sum": int(self.global_label_sum),
            "g2_total_fid_value_pairs": len(self.fid_value_stats),
            "g2_pairs_above_min_count": len(scored),
            "g2_top_fid_values_by_signal": [
                {
                    "section": section,
                    "fid": int(fid),
                    "value": int(val),
                    "cvr": float(cvr),
                    "count": int(cnt),
                    "label_sum": int(ls),
                    "signal": float(sig),
                    "cvr_gap_vs_global": float(cvr - global_cvr),
                }
                for section, fid, val, cvr, cnt, ls, sig in top
            ],
            "g2_user_int_plan_fids": [fid for fid, _, _ in self.user_int_plan],
            "g2_item_int_plan_fids": [fid for fid, _, _ in self.item_int_plan],
        }


# ============================================================================
# Driver
# ============================================================================

def _find_parquet_files(data_dir: Path) -> List[Path]:
    files = sorted(data_dir.rglob("*.parquet"))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train-side EDA for A2/F3/G2 (5/17 EXP-067)"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing train parquet files",
    )
    parser.add_argument(
        "--schema_path", type=str, required=True,
        help="Path to schema.json",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/tmp/eda_data_distribution",
        help="Output directory for eda_data_distribution.json",
    )
    parser.add_argument(
        "--max_files", type=int, default=0,
        help="Max parquet files to scan (0 = all, ~1044 in production)",
    )
    parser.add_argument(
        "--max_rows_per_file", type=int, default=0,
        help="Max rows per file (0 = all)",
    )
    parser.add_argument(
        "--a2_top_k", type=int, default=50_000,
        help="Top-K item ids per (domain, fid) to retain in A2 output",
    )
    parser.add_argument(
        "--f3_user_id_cap", type=int, default=500_000,
        help="Cap on F3 user_id set size",
    )
    parser.add_argument(
        "--g2_user_max_vocab", type=int, default=256,
        help="G2: include user_int fids with vocab ≤ this",
    )
    parser.add_argument(
        "--g2_item_max_vocab", type=int, default=256,
        help="G2: include item_int fids with vocab ≤ this",
    )
    parser.add_argument(
        "--g2_top_n", type=int, default=100,
        help="G2: top N clusters by signal to emit",
    )
    parser.add_argument(
        "--g2_min_count", type=int, default=100,
        help="G2: min cluster count to consider",
    )
    parser.add_argument(
        "--seq_item_min_vocab", type=int, default=1_000_000,
        help="A2: include seq fids with vocab > this (item id-like)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = _load_schema(args.schema_path)

    seq_item_id_plan = _build_seq_item_id_plan(
        schema, min_vocab=args.seq_item_min_vocab)
    user_int_plan = _build_low_vocab_int_plan(
        schema, "user_int", max_vocab=args.g2_user_max_vocab)
    item_int_plan = _build_low_vocab_int_plan(
        schema, "item_int", max_vocab=args.g2_item_max_vocab)

    logger.info(
        "[plan] seq_item_id_plan = %s",
        {d: [(fid, vocab) for fid, vocab, _ in lst]
         for d, lst in seq_item_id_plan.items()},
    )
    logger.info(
        "[plan] user_int_low_vocab fids (≤%d): %s",
        args.g2_user_max_vocab,
        [(fid, vocab) for fid, _, vocab in user_int_plan],
    )
    logger.info(
        "[plan] item_int_low_vocab fids (≤%d): %s",
        args.g2_item_max_vocab,
        [(fid, vocab) for fid, _, vocab in item_int_plan],
    )

    a2 = _A2Accumulator(seq_item_id_plan, top_k=args.a2_top_k)
    f3 = _F3Accumulator(user_id_cap=args.f3_user_id_cap)
    g2 = _G2Accumulator(
        user_int_plan, item_int_plan,
        top_n=args.g2_top_n, min_count=args.g2_min_count,
    )

    files = _find_parquet_files(data_dir)
    if args.max_files > 0 and len(files) > args.max_files:
        # Even-spaced subsampling for temporal coverage
        step = len(files) // args.max_files
        files = [files[i * step] for i in range(args.max_files)]
    logger.info("[scan] %d parquet files in %s", len(files), data_dir)

    t0 = time.time()
    total_rows = 0
    files_scanned = 0
    for fp in files:
        try:
            pf = pq.ParquetFile(str(fp))
        except (OSError, ValueError) as e:
            logger.warning("[scan] skip %s: %r", fp, e)
            continue
        for rg_idx in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg_idx)
            if tbl.num_rows == 0:
                continue
            if args.max_rows_per_file > 0:
                row_budget = args.max_rows_per_file - (
                    total_rows % max(1, args.max_rows_per_file))
                if row_budget <= 0:
                    break
                if tbl.num_rows > row_budget:
                    tbl = tbl.slice(0, row_budget)
            a2.update(tbl)
            f3.update(tbl)
            g2.update(tbl)
            total_rows += tbl.num_rows
            del tbl
        files_scanned += 1
        if files_scanned % 50 == 0 or files_scanned == len(files):
            elapsed = time.time() - t0
            logger.info(
                "[scan] %d/%d files · %d rows · %.1fs · "
                "a2_keys=%s · f3_distinct=%d · g2_pairs=%d",
                files_scanned, len(files), total_rows, elapsed,
                {f"{d}_{fid}": len(c) for (d, fid), c in a2.counter.items()},
                f3.distinct_count, len(g2.fid_value_stats),
            )
        gc.collect()

    elapsed = time.time() - t0
    logger.info(
        "[scan] done · %d files · %d rows · %.1fs",
        files_scanned, total_rows, elapsed,
    )

    out: Dict[str, Any] = {
        "meta": {
            "data_dir": str(data_dir),
            "schema_path": args.schema_path,
            "files_scanned": files_scanned,
            "total_rows": total_rows,
            "wall_time_sec": float(elapsed),
            "a2_top_k": args.a2_top_k,
            "f3_user_id_cap": args.f3_user_id_cap,
            "g2_user_max_vocab": args.g2_user_max_vocab,
            "g2_item_max_vocab": args.g2_item_max_vocab,
            "g2_top_n": args.g2_top_n,
            "g2_min_count": args.g2_min_count,
            "seq_item_min_vocab": args.seq_item_min_vocab,
        },
    }
    out.update(a2.finalize())
    out.update(f3.finalize())
    out.update(g2.finalize())

    out_path = out_dir / "eda_data_distribution.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info("[done] wrote %s (%d bytes)", out_path, out_path.stat().st_size)

    # Pretty top-N to stdout for terminal review (platform UI Logs visible)
    print('\n========== A2 · TRAIN SEQ ITEM FREQUENCY (per fid · top 10) ==========')
    a2_distinct = out.get('a2_train_seq_item_distinct_per_fid', {})
    a2_freq = out.get('a2_train_seq_item_freq', {})
    for key, distinct in a2_distinct.items():
        top10 = a2_freq.get(key, [])[:10]
        print(f"  {key:25s} distinct={distinct:>10d}  top10={top10}")

    print('\n========== F3 · USER_ID DISTINCT TRAIN ==========')
    f3_distinct = out.get('f3_train_user_id_distinct', 0)
    print(f"  f3_train_user_id_distinct = {f3_distinct:,}")
    print(f"  f3_user_id_cap = {out.get('f3_user_id_cap', 0):,}")

    print('\n========== G2 · TOP CVR SIGNALS BY (fid, value) · top 30 ==========')
    g2_global_cvr = out.get('g2_global_cvr', 0.0)
    g2_global_count = out.get('g2_global_count', 0)
    g2_top = out.get('g2_top_fid_values_by_signal', [])[:30]
    print(f"  g2_global_cvr = {g2_global_cvr:.6f} ({g2_global_count:,} samples)")
    print(f"  total_pairs = {out.get('g2_total_fid_value_pairs', 0)} · "
          f"above_min_count = {out.get('g2_pairs_above_min_count', 0)}")
    for r in g2_top:
        sec = r.get('section', '?')
        fid = r.get('fid', '?')
        val = r.get('value', '?')
        cnt = r.get('count', 0)
        cvr = r.get('cvr', 0.0)
        gap = r.get('cvr_gap_vs_global', 0.0)
        sig = r.get('signal', 0.0)
        print(f"  [{sec:4s}] fid={fid:<4} val={str(val)[:8]:8s} "
              f"cnt={cnt:>8} cvr={cvr:.4f} gap={gap:+.4f} signal={sig:>10.2f}")

    # Platform log recovery: emit base64+gzip blob as the very last stdout
    # so tools/decode_eda_blob.py can recover the full report when running
    # on the platform (UI Logs only shows tail 1000 lines · file is NOT
    # downloadable). Mirror the proven pattern from eda_correlation.py.
    #
    # ⚠️ Size-bounded subset: full `out` dict is too large (raw 18.9MB ·
    # gzip ~3MB · base64 ~4MB · ~33k blob lines · platform UI tail 1000
    # lines truncates blob). Build compact `out_blob` containing only the
    # essential data points we actually use downstream, capped to keep
    # blob ≤ 100 KB raw JSON (≈ 30 KB gzipped · ≈ 40 KB base64 · ≈ 350
    # blob lines · safely under platform UI tail limit).
    #
    # What we keep:
    # - A2: per-fid top 500 high-freq item ids (vs default top 50k) ·
    #   plus distinct count (full)
    # - F3: distinct count only (no user_id samples)
    # - G2: full top_n list (already small ≤ 200 entries)
    # - meta: full
    out_blob = {
        "meta": out.get("meta", {}),
        "a2_train_seq_item_distinct_per_fid": a2_distinct,
        "a2_train_seq_item_freq_top500": {
            key: a2_freq.get(key, [])[:500] for key in a2_distinct
        },
        "f3_train_user_id_distinct": f3_distinct,
        "f3_user_id_cap": out.get("f3_user_id_cap", 0),
        "g2_global_cvr": g2_global_cvr,
        "g2_global_count": g2_global_count,
        "g2_total_fid_value_pairs": out.get("g2_total_fid_value_pairs", 0),
        "g2_pairs_above_min_count": out.get("g2_pairs_above_min_count", 0),
        "g2_top_fid_values_by_signal": out.get(
            "g2_top_fid_values_by_signal", []
        ),
        "g2_user_int_plan_fids": out.get("g2_user_int_plan_fids", []),
        "g2_item_int_plan_fids": out.get("g2_item_int_plan_fids", []),
    }
    import base64
    import gzip
    payload = json.dumps(
        out_blob, sort_keys=True, ensure_ascii=False
    ).encode('utf-8')
    gz = gzip.compress(payload, compresslevel=9)
    b64 = base64.b64encode(gz).decode('ascii')
    chunk_size = 120
    chunks = [b64[i:i + chunk_size] for i in range(0, len(b64), chunk_size)]
    sys.stdout.flush()
    print()
    print('<<<EDA_BLOB_START>>>')
    for ch in chunks:
        print(ch)
    print('<<<EDA_BLOB_END>>>')
    sys.stdout.flush()
    logger.info(
        "[blob] emitted %d chars / %d chunks / raw %d bytes "
        "(size-bounded subset)",
        len(b64), len(chunks), len(payload),
    )


if __name__ == "__main__":
    main()
