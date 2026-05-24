"""Train-set feature-to-feature correlation EDA for TAAC 2026.

Independent script (no torch). Computes:

  Q-A  user_int × user_int  Cramér's V + mutual information
  Q-B  user_int × item_int  Cramér's V + mutual information
  Q-C  fid → label          mutual information (fid importance baseline)

All metrics work on **categorical** values (we use Cramér's V because
Pearson is invalid on int categorical fids). Multi-hot fids are reduced
to their first non-zero slot (a stable proxy for the fid's "primary
value"). High-cardinality fids (vocab > 256) are hash-bucketed to 256
buckets to bound contingency-table memory; this preserves correlation
ranking even though absolute values become slightly conservative.

Targets DECEM Trick #4 (ns_groups by correlation) and #5 (FM/DCN
cross features) — once we know which fid pairs carry the most
mutual information, we can:

  1. Re-cut ns_groups so high-MI fids share a group (RankMixerNS local
     mixing) and low-MI fids are spread across groups
  2. Pick the top-K user×item MI pairs as candidates for FM 2-way
     cross or DCN explicit cross terms

Usage::

  python src/eda_correlation.py \\
    --parquet_dir ./data/demo_rg100 \\
    --schema_path ./data/demo_rg100/schema.json \\
    --out ./eda_correlation.json \\
    [--max_row_groups 0]    # 0 = all
    [--top_k 50]            # how many top pairs to retain in output
    [--hash_buckets 256]    # cap for high-cardinality fids
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
LOG = logging.getLogger('eda-corr')
print = functools.partial(print, flush=True)


# ───────── Config ─────────

DEFAULT_HASH_BUCKETS = 256
DEFAULT_TOP_K = 50


# ───────── Helpers ─────────

def _load_schema(schema_path: str) -> Dict[str, Any]:
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _list_parquet_files(parquet_dir: str) -> List[str]:
    p = Path(parquet_dir)
    if p.is_file():
        return [str(p)]
    files = sorted(glob.glob(str(p / '*.parquet')))
    if not files:
        raise FileNotFoundError(f'No .parquet files under {parquet_dir}')
    return files


def _first_nonzero_per_row(arr: np.ndarray) -> np.ndarray:
    """Reduce a multi-hot per-fid block (B, dim) → (B,) primary value.

    Single-column fids (dim=1) flatten directly. Multi-hot picks the first
    column with the row's any-nonzero mask; rows that are entirely zero
    map to 0 (treated as the "missing" anchor — same convention as the
    is-missing path).
    """
    if arr.ndim == 1:
        return arr.copy()
    if arr.shape[1] == 1:
        return arr[:, 0].copy()
    nonzero_mask = (arr != 0).any(axis=1)
    out = np.zeros(arr.shape[0], dtype=arr.dtype)
    if nonzero_mask.any():
        out[nonzero_mask] = arr[nonzero_mask, 0]
    return out


def _hash_bucket(values: np.ndarray, n_buckets: int) -> np.ndarray:
    """Modulo hash to ``n_buckets``. Preserves 0 (missing sentinel) → 0.

    We keep 0 → 0 explicitly so the "missing" anchor stays distinguishable
    from the rest of the bucketed space; the remaining values are spread
    across buckets [1, n_buckets - 1] via ``(v - 1) % (n_buckets - 1) + 1``.
    """
    out = np.zeros_like(values)
    nonzero = values != 0
    if nonzero.any():
        rest = n_buckets - 1
        out[nonzero] = ((values[nonzero] - 1) % rest) + 1
    return out


# ───────── Schema parsing (matches dataset.FeatureSchema layout) ─────────

def _build_int_plan(
    schema: Dict[str, Any],
    side: str,  # 'user' or 'item'
) -> List[Tuple[str, int, int]]:
    """Return list of (col_name, dim, vocab_size) for user_int / item_int fids.

    Schema entries follow the project convention ``[fid, vocab_size, dim]``;
    the parquet column name is ``f'{side}_int_feats_{fid}'``.
    """
    feats_key = f'{side}_int'
    if feats_key not in schema:
        return []
    feats = schema[feats_key]
    plan: List[Tuple[str, int, int]] = []
    for entry in feats:
        if isinstance(entry, dict):
            fid = int(entry.get('fid', entry.get('id')))
            vocab = int(entry.get('vocab_size', 0))
            dim = int(entry.get('dim', 1))
        else:
            fid, vocab, dim = int(entry[0]), int(entry[1]), int(entry[2])
        col = f'{side}_int_feats_{fid}'
        plan.append((col, dim, vocab))
    return plan


# ───────── Categorical correlation core ─────────

def _build_contingency(
    a: np.ndarray, b: np.ndarray, va: int, vb: int,
) -> np.ndarray:
    """Return a (va × vb) contingency table.

    a / b must already be non-negative ints in [0, va) / [0, vb).
    Uses np.add.at for vectorized accumulation.
    """
    table = np.zeros((va, vb), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    return table


def _cramers_v(table: np.ndarray) -> float:
    """Cramér's V from a contingency table.

    Returns 0 when table degenerates (single row/col or zero total).
    """
    n = table.sum()
    if n == 0:
        return 0.0
    rows = table.sum(axis=1, keepdims=True)
    cols = table.sum(axis=0, keepdims=True)
    if (rows == 0).all() or (cols == 0).all():
        return 0.0
    expected = rows @ cols / n
    nonzero_exp = expected > 0
    chi2 = ((table[nonzero_exp] - expected[nonzero_exp]) ** 2 /
            expected[nonzero_exp]).sum()
    r, c = table.shape
    denom = n * max(min(r, c) - 1, 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(chi2 / denom))


def _mutual_info_vec(table: np.ndarray) -> float:
    """Vectorized mutual information ``I(A;B)`` in nats.

    ``MI = sum_{a,b} p(a,b) log [p(a,b) / (p(a) * p(b))]``.
    Returns 0 when the table is empty or fully degenerate.
    """
    n = table.sum()
    if n == 0:
        return 0.0
    pab = table.astype(np.float64) / n
    pa = pab.sum(axis=1, keepdims=True)  # (R, 1)
    pb = pab.sum(axis=0, keepdims=True)  # (1, C)
    denom = pa * pb
    valid = (pab > 0) & (denom > 0)
    if not valid.any():
        return 0.0
    return float((pab[valid] * np.log(pab[valid] / denom[valid])).sum())


# ───────── Per-fid value extraction (streaming) ─────────

def _extract_fid_values(
    parquet_files: List[str],
    plan: List[Tuple[str, int, int]],
    side_label: str,
    max_row_groups: int = 0,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Stream parquet files; for each fid in ``plan`` collect primary-value
    np.array of length N. Also collects the binary label vector.

    Multi-hot fids are reduced via ``_first_nonzero_per_row``. Values are
    kept as int64 (no clipping yet — bucketing happens later in
    ``_prepare_for_table``).

    Returns ``(per_fid_values, labels)``.
    """
    per_fid: Dict[str, List[np.ndarray]] = {col: [] for col, _, _ in plan}
    labels_list: List[np.ndarray] = []
    rg_count = 0
    rows_total = 0
    t0 = time.time()
    for fpath in parquet_files:
        pf = pq.ParquetFile(fpath)
        for rg_idx in range(pf.num_row_groups):
            if max_row_groups > 0 and rg_count >= max_row_groups:
                break
            tbl = pf.read_row_group(rg_idx, use_threads=True)
            tbl = tbl.combine_chunks()
            n = tbl.num_rows

            label_col = tbl.column('label_type') \
                if 'label_type' in tbl.schema.names else None
            if label_col is not None:
                arr = label_col.to_numpy(zero_copy_only=False)
                arr = np.where(arr == None, 0, arr).astype(np.int64)  # noqa: E711
                labels_list.append((arr == 2).astype(np.int8))
            else:
                labels_list.append(np.zeros(n, dtype=np.int8))

            for col, dim, _vs in plan:
                if col not in tbl.schema.names:
                    per_fid[col].append(np.zeros(n, dtype=np.int64))
                    continue
                arr = tbl.column(col)
                # combine_chunks() above guarantees ChunkedArray with one
                # chunk; .chunk(0) gives the underlying Array.
                if hasattr(arr, 'chunk'):
                    arr = arr.chunk(0)
                if dim == 1:
                    vals = arr.to_numpy(zero_copy_only=False)
                    if vals.dtype == object:
                        vals = np.array(
                            [0 if v is None else int(v) for v in vals],
                            dtype=np.int64)
                    else:
                        vals = vals.astype(np.int64, copy=False)
                    vals = np.where(vals < 0, 0, vals)
                    per_fid[col].append(vals)
                else:
                    flat = arr.values.to_numpy(zero_copy_only=False)
                    if flat.dtype == object:
                        flat = np.array(
                            [0 if v is None else int(v) for v in flat],
                            dtype=np.int64)
                    else:
                        flat = flat.astype(np.int64, copy=False)
                    flat = np.where(flat < 0, 0, flat)
                    offs = arr.offsets.to_numpy().astype(np.int64)
                    block = np.zeros((n, dim), dtype=np.int64)
                    for i in range(n):
                        s = int(offs[i])
                        e = int(offs[i + 1])
                        take = min(e - s, dim)
                        if take > 0:
                            block[i, :take] = flat[s:s + take]
                    per_fid[col].append(_first_nonzero_per_row(block))
            rg_count += 1
            rows_total += n
            if rg_count % 50 == 0:
                LOG.info(
                    f'[{side_label}] scanned {rg_count} RGs · '
                    f'{rows_total:,} rows · '
                    f'elapsed {time.time() - t0:.1f}s')
        if max_row_groups > 0 and rg_count >= max_row_groups:
            break
    LOG.info(
        f'[{side_label}] DONE scan: {rg_count} RGs · {rows_total:,} rows · '
        f'elapsed {time.time() - t0:.1f}s')

    out_arrays = {col: np.concatenate(chunks) for col, chunks in per_fid.items()}
    labels_arr = np.concatenate(labels_list) if labels_list \
        else np.zeros(0, dtype=np.int8)
    return out_arrays, labels_arr


def _prepare_for_table(
    values: np.ndarray, vocab_cap: int,
) -> Tuple[np.ndarray, int]:
    """Map raw int64 values to dense [0, V) ids, bucketing if vocab > cap.

    Returns ``(prepared, V_effective)``. We always reserve id=0 for the
    "missing" anchor so downstream metrics distinguish missing rows.
    """
    distinct = np.unique(values)
    if distinct.size <= vocab_cap:
        # Dense remap: build a lookup
        remap = {int(v): i for i, v in enumerate(distinct.tolist())}
        out = np.array([remap[int(v)] for v in values], dtype=np.int64)
        return out, distinct.size
    return _hash_bucket(values, vocab_cap), vocab_cap


# ───────── Main pipeline ─────────

def _compute_pair_metrics(
    a: np.ndarray, va: int, b: np.ndarray, vb: int,
) -> Tuple[float, float]:
    table = _build_contingency(a, b, va, vb)
    return _cramers_v(table), _mutual_info_vec(table)


def _compute_label_mi(
    fid_vals: np.ndarray, vfid: int, labels: np.ndarray,
) -> float:
    """MI between a fid (already prepared) and binary label."""
    if labels.size == 0:
        return 0.0
    table = _build_contingency(
        fid_vals.astype(np.int64),
        labels.astype(np.int64),
        vfid, 2,
    )
    return _mutual_info_vec(table)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet_dir', required=True)
    parser.add_argument('--schema_path', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max_row_groups', type=int, default=0,
                        help='0 = all RGs; small N for local smoke')
    parser.add_argument('--top_k', type=int, default=DEFAULT_TOP_K,
                        help='retain top-K pairs by |Cramér V|')
    parser.add_argument('--hash_buckets', type=int,
                        default=DEFAULT_HASH_BUCKETS,
                        help='vocab cap for high-cardinality fids')
    args = parser.parse_args()

    schema = _load_schema(args.schema_path)
    user_plan = _build_int_plan(schema, 'user')
    item_plan = _build_int_plan(schema, 'item')
    LOG.info(
        f'plan: {len(user_plan)} user_int fids, {len(item_plan)} item_int fids')

    parquet_files = _list_parquet_files(args.parquet_dir)
    LOG.info(f'parquet files: {len(parquet_files)}')

    user_vals, labels = _extract_fid_values(
        parquet_files, user_plan, 'user_int', args.max_row_groups)
    item_vals, labels2 = _extract_fid_values(
        parquet_files, item_plan, 'item_int', args.max_row_groups)
    if labels.size != labels2.size:
        LOG.warning(
            f'label size mismatch user-pass={labels.size} '
            f'item-pass={labels2.size}; using user-pass labels')

    LOG.info('preparing categorical encodings (bucket high-cardinality fids)')
    user_prep: Dict[str, Tuple[np.ndarray, int]] = {}
    item_prep: Dict[str, Tuple[np.ndarray, int]] = {}
    for col, _dim, _vs in user_plan:
        user_prep[col] = _prepare_for_table(user_vals[col], args.hash_buckets)
    for col, _dim, _vs in item_plan:
        item_prep[col] = _prepare_for_table(item_vals[col], args.hash_buckets)

    # Q-A user × user (upper triangle only, exclude self)
    LOG.info('computing user × user pairwise metrics')
    user_pairs: List[Dict[str, Any]] = []
    user_cols = [c for c, _, _ in user_plan]
    for i, ca in enumerate(user_cols):
        a, va = user_prep[ca]
        for cb in user_cols[i + 1:]:
            b, vb = user_prep[cb]
            cv, mi = _compute_pair_metrics(a, va, b, vb)
            user_pairs.append({'a': ca, 'b': cb, 'va': int(va), 'vb': int(vb),
                               'cramer_v': cv, 'mi': mi})
    user_pairs.sort(key=lambda x: -x['cramer_v'])
    LOG.info(f'  {len(user_pairs)} user×user pairs computed')

    # Q-B user × item (full cross)
    LOG.info('computing user × item pairwise metrics')
    cross_pairs: List[Dict[str, Any]] = []
    item_cols = [c for c, _, _ in item_plan]
    for cu in user_cols:
        u, vu = user_prep[cu]
        for ci in item_cols:
            it, vi = item_prep[ci]
            cv, mi = _compute_pair_metrics(u, vu, it, vi)
            cross_pairs.append({'u': cu, 'i': ci, 'vu': int(vu), 'vi': int(vi),
                                'cramer_v': cv, 'mi': mi})
    cross_pairs.sort(key=lambda x: -x['cramer_v'])
    LOG.info(f'  {len(cross_pairs)} user×item pairs computed')

    # Q-C fid → label MI
    LOG.info('computing fid → label MI')
    fid_label: List[Dict[str, Any]] = []
    for col in user_cols:
        a, va = user_prep[col]
        mi = _compute_label_mi(a, va, labels)
        fid_label.append({'col': col, 'side': 'user', 'mi': mi, 'v': int(va)})
    for col in item_cols:
        a, va = item_prep[col]
        mi = _compute_label_mi(a, va, labels)
        fid_label.append({'col': col, 'side': 'item', 'mi': mi, 'v': int(va)})
    fid_label.sort(key=lambda x: -x['mi'])

    summary = {
        'version': 1,
        'scan': {
            'parquet_dir': args.parquet_dir,
            'rows': int(labels.size),
            'user_fids': len(user_plan),
            'item_fids': len(item_plan),
            'hash_buckets': args.hash_buckets,
            'top_k': args.top_k,
        },
        'q_a_user_user_top': user_pairs[:args.top_k],
        'q_b_user_item_top': cross_pairs[:args.top_k],
        'q_c_fid_label_mi': fid_label,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    LOG.info(f'wrote {out_path}')

    # Pretty top-N for terminal review
    print('\n========== TOP USER × USER (Cramér V desc) ==========')
    for r in user_pairs[:20]:
        print(f"  {r['a']:30s}  ↔  {r['b']:30s}  CV={r['cramer_v']:.4f}  MI={r['mi']:.4f}")
    print('\n========== TOP USER × ITEM (Cramér V desc) ==========')
    for r in cross_pairs[:20]:
        print(f"  {r['u']:30s}  ×  {r['i']:30s}  CV={r['cramer_v']:.4f}  MI={r['mi']:.4f}")
    print('\n========== FID → LABEL MI (desc) ==========')
    for r in fid_label[:25]:
        print(f"  [{r['side']:4s}]  {r['col']:30s}  MI={r['mi']:.6f}  V={r['v']}")

    # Platform log recovery: emit base64+gzip blob as the very last stdout
    # so tools/decode_eda_blob.py can recover the full report when running
    # on the platform (UI Logs only shows tail 1000 lines).
    import base64
    import gzip
    payload = json.dumps(summary, sort_keys=True).encode('utf-8')
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
    LOG.info(
        f'blob emitted: {len(b64)} chars, {len(chunks)} chunks, '
        f'raw {len(payload)} bytes')


if __name__ == '__main__':
    main()
