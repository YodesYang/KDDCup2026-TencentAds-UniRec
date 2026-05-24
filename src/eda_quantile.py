"""Independent EDA scanner for TAAC 2026 pcvr parquet data.

Runs outside training loop -- pure pyarrow + numpy, no torch. Designed to:
- Finish in ~5-10 min on platform data (1.9M rows, 1044 RGs)
- Produce the same 4 EDA questions as the in-training EDA collector, but
  fully vectorized (no Python per-row loops, no dict counters on hot path)
- **Recover full report from the platform Logs UI (tail 1000 rows only)**
  via base64(gzip(JSON)) blob printed as the ABSOLUTE LAST lines of stdout,
  flanked by sentinels <<<EDA_BLOB_START>>> / <<<EDA_BLOB_END>>>.

EDA Questions:
  Q1  user_id / item_id frequency distribution
  Q2  per-fid (user_int + item_int) null rate × CVR gap
  Q4  per-domain sequence true length quantiles
  Q5  per-domain time_diff quantiles (timestamp_ref - seq_ts)

Performance design:
- Scan parquet row group by row group (streaming, bounded memory)
- For Q1: accumulate uid/iid via numpy.unique per RG, merge final counts
- For Q2: per-fid null mask = (arr <= 0 | isnull) computed as numpy vectorized
- For Q5: arrow ListArray -> flat numpy values + offsets, single concat, one
          sort per reservoir merge (no per-row Python loop)

Platform log recovery protocol (why this works when eda_platform.py v5 failed):
- Platform Logs UI shows ONLY tail 1000 rows of stdout; early output is lost.
- eda_platform.py v5 put base64 blob in the middle then kept printing pretty
  tables after, pushing the blob out of the 1000-row window.
- This script prints the blob AS THE VERY LAST output (no lines after
  <<<EDA_BLOB_END>>>). Progress logs (~25 lines for 1044 RGs) + pretty
  summary (~30 lines) + sentinel/blob (~500 lines for full data) = well
  below 1000-row window.
- User copies entire Logs UI output, pipes start/end sentinel extract
  through `base64 -d | gunzip | jq .` to recover full JSON.

Usage::

  python src/eda_quantile.py \\
    --parquet_dir ./data/demo_rg100 \\
    --schema_path ./data/demo_rg100/schema.json \\
    --out ./eda_summary.json \\
    [--max_row_groups 0]  # 0 = all RGs; set small for local smoke

Platform usage (training job, run.sh invokes):

  python eda_quantile.py \\
    --parquet_dir $TRAIN_DATA_PATH \\
    --schema_path $TRAIN_DATA_PATH/schema.json \\
    --out $TRAIN_LOG_PATH/eda_summary.json

"""

from __future__ import annotations

import argparse
import base64
import functools
import glob
import gzip
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
LOG = logging.getLogger('eda')
print = functools.partial(print, flush=True)  # platform log safety


# Sentinels — keep exact literal strings; extractor relies on them.
BLOB_START = '<<<EDA_BLOB_START>>>'
BLOB_END = '<<<EDA_BLOB_END>>>'
EOF_MARKER = '<<<EDA_EOF>>>'


def _rss_mb() -> float:
    """Best-effort resident set size in MB; 0.0 when unreadable (non-Linux)."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except OSError:
        pass
    return 0.0


# ───── Helpers ─────

def _reservoir_merge(
    buf: List[np.ndarray],
    new: np.ndarray,
    max_size: int,
    rng: np.random.Generator,
) -> None:
    """Mutate buf in place: append new data, subsample in-flight when buf grows.

    Memory bound: ≤ 2 * max_size int64 per buf (amortized O(N) total).

    Subsample strategy: when the total size in buf exceeds `2 * max_size`,
    concat everything, uniformly downsample to `max_size`, and replace buf
    with a single chunk. This is NOT strict Algorithm R, but for quantile
    estimation the uniform downsample is acceptable and cheap.

    This matters because platform RGs can contribute ~50K-200K seq entries
    per domain per RG; without in-flight cropping, a full scan would buffer
    ~300M int64 entries (= 2.4GB) per domain → OOM at ~RG 700/1044.
    """
    if new.size == 0:
        return
    buf.append(new)
    cur = sum(a.size for a in buf)
    if cur > 2 * max_size:
        arr = np.concatenate(buf)
        idx = rng.choice(arr.size, size=max_size, replace=False)
        buf.clear()
        buf.append(arr[idx])


def _reservoir_finalize(
    buf: List[np.ndarray],
    max_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if not buf:
        return np.empty(0, dtype=np.int64)
    arr = np.concatenate(buf)
    if len(arr) > max_size:
        idx = rng.choice(len(arr), size=max_size, replace=False)
        arr = arr[idx]
    return arr


def _quantiles(arr: np.ndarray) -> Dict[str, Any]:
    if arr.size == 0:
        return {'n': 0}
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    return {
        'n': int(arr.size),
        'mean': float(arr.mean()),
        'min': int(arr.min()),
        'max': int(arr.max()),
        'quantiles': {f'p{q}': float(np.percentile(arr, q)) for q in qs},
    }


def _pa_list_to_numpy(col: pa.ChunkedArray) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate a ChunkedArray of list<int64> into (values, offsets).

    Offsets are re-computed relative to the concatenated values buffer.
    """
    values_parts: List[np.ndarray] = []
    offsets_parts: List[np.ndarray] = []
    cum = 0
    for chunk in col.chunks:
        vals = chunk.values.to_numpy(zero_copy_only=False).astype(np.int64,
                                                                   copy=False)
        offs = chunk.offsets.to_numpy(zero_copy_only=False).astype(np.int64,
                                                                   copy=False)
        if values_parts:
            offs = offs + cum
            offs = offs[1:]
        values_parts.append(vals)
        offsets_parts.append(offs)
        cum += len(vals)
    values = np.concatenate(values_parts) if values_parts else np.empty(
        0, dtype=np.int64)
    offsets = np.concatenate(offsets_parts) if offsets_parts else np.array(
        [0], dtype=np.int64)
    return values, offsets


def _scalar_to_numpy(col: pa.ChunkedArray) -> np.ndarray:
    """Safe scalar int cast (nulls become 0, following dataset.py convention)."""
    arr = col.fill_null(0).to_numpy(zero_copy_only=False)
    return np.asarray(arr, dtype=np.int64)


def _list_any_scalar_null(col: pa.ChunkedArray) -> np.ndarray:
    """For a list<int64> column, return per-row bool of 'effectively null'.

    A row is deemed null iff its list is either empty, all entries <= 0, or
    Arrow-null at the row level. Matches dataset.py collapse rule.
    """
    masks: List[np.ndarray] = []
    for chunk in col.chunks:
        vals = chunk.values.to_numpy(zero_copy_only=False).astype(np.int64,
                                                                   copy=False)
        offs = chunk.offsets.to_numpy(zero_copy_only=False).astype(np.int64,
                                                                   copy=False)
        n = len(offs) - 1
        out = np.ones(n, dtype=bool)
        # Vectorized positive-entry count per row via bincount + gt0 mask
        gt0 = vals > 0
        if len(vals) > 0 and n > 0:
            row_id = np.repeat(np.arange(n, dtype=np.int64),
                               offs[1:] - offs[:-1])
            if row_id.size > 0:
                pos_per_row = np.bincount(row_id[gt0], minlength=n) if gt0.any() \
                    else np.zeros(n, dtype=np.int64)
                out = pos_per_row == 0
        # also mark arrow-null rows as null
        if chunk.null_count > 0:
            valid = np.asarray(chunk.is_valid())
            out |= ~valid
        masks.append(out)
    return np.concatenate(masks) if masks else np.empty(0, dtype=bool)


# ───── Main ─────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet_dir', type=str, required=True)
    ap.add_argument('--schema_path', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--max_row_groups', type=int, default=0,
                    help='Cap RGs scanned for smoke; 0 = all')
    ap.add_argument('--reservoir_size', type=int, default=200000,
                    help='Per-domain reservoir size for Q4/Q5')
    ap.add_argument('--id_counter_cap', type=int, default=0,
                    help='Cap distinct uids/iids tracked; 0 = no cap')
    ap.add_argument('--stdout_blob', dest='stdout_blob',
                    action='store_true', default=True,
                    help='Print base64(gzip(JSON)) blob as the last stdout '
                         'output for platform Logs recovery (default: on)')
    ap.add_argument('--no_stdout_blob', dest='stdout_blob',
                    action='store_false',
                    help='Suppress blob (local runs that only need --out)')
    ap.add_argument('--blob_chunk_size', type=int, default=120,
                    help='Chars per blob line (default 120)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load schema
    with open(args.schema_path) as f:
        schema = json.load(f)
    user_int_plan: List[Tuple[int, int, int]] = [  # (fid, vs, dim)
        tuple(e) for e in schema['user_int']
    ]
    item_int_plan: List[Tuple[int, int, int]] = [
        tuple(e) for e in schema['item_int']
    ]
    seq_cfg: Dict[str, Dict[str, Any]] = schema['seq']

    # Enumerate parquet files
    parquet_files = sorted(glob.glob(os.path.join(args.parquet_dir,
                                                  '*.parquet')))
    if not parquet_files:
        raise SystemExit(f'No *.parquet under {args.parquet_dir}')

    LOG.info('Parquet files: %d · schema: %d user_int / %d item_int / %d seq',
             len(parquet_files), len(user_int_plan), len(item_int_plan),
             len(seq_cfg))

    # Accumulators
    total_rows = 0
    total_pos = 0
    # Q1
    uid_counter: Dict[int, int] = {}
    iid_counter: Dict[int, int] = {}
    # Q2: per-fid [n_null, n_total, pos_null, pos_notnull]
    user_int_acc = {fid: [0, 0, 0, 0] for fid, _, _ in user_int_plan}
    item_int_acc = {fid: [0, 0, 0, 0] for fid, _, _ in item_int_plan}
    # Q4/Q5: reservoirs
    seq_len_buf: Dict[str, List[np.ndarray]] = {d: [] for d in seq_cfg}
    time_diff_buf: Dict[str, List[np.ndarray]] = {d: [] for d in seq_cfg}
    # Q6/Q7 · CVR by absolute time bucket
    # Q6: day-of-year bucketing (UTC days since epoch). Each entry stores
    #     [n_rows, n_pos]. Captures train weekday/lunar-festival CVR shift.
    # Q7: hour-of-day bucketing (UTC hour 0-23). Each entry stores
    #     [n_rows, n_pos]. Captures daypart shift relevant to test (北京
    #     03-04 09:00~14:00 = UTC 01:00~06:00).
    #
    # Both use plain dict[int -> [int, int]] for O(1) update + O(B/24)
    # finalization. Memory bounded (≤ 30 entries each).
    cvr_by_day: Dict[int, List[int]] = {}
    cvr_by_hour: Dict[int, List[int]] = {}
    # seq domain prefix / ts_fid
    dom_prefix = {d: c['prefix'] for d, c in seq_cfg.items()}
    dom_ts_fid = {d: c.get('ts_fid') for d, c in seq_cfg.items()}

    rg_seen = 0
    t_start = time.time()

    for fpath in parquet_files:
        pf = pq.ParquetFile(fpath)
        for rg_idx in range(pf.num_row_groups):
            if args.max_row_groups and rg_seen >= args.max_row_groups:
                break
            rg = pf.read_row_group(rg_idx)
            n = rg.num_rows
            total_rows += n

            # Labels: label_type == 2 → positive (mirrors dataset.py)
            label_type = _scalar_to_numpy(rg.column('label_type'))
            labels = (label_type == 2).astype(np.int64)
            total_pos += int(labels.sum())

            # Q1 user_id / item_id counter
            uids = _scalar_to_numpy(rg.column('user_id'))
            iids = _scalar_to_numpy(rg.column('item_id'))
            u_uniq, u_counts = np.unique(uids, return_counts=True)
            i_uniq, i_counts = np.unique(iids, return_counts=True)
            for u, c in zip(u_uniq.tolist(), u_counts.tolist()):
                if args.id_counter_cap and len(uid_counter) \
                        >= args.id_counter_cap and u not in uid_counter:
                    continue
                uid_counter[u] = uid_counter.get(u, 0) + c
            for u, c in zip(i_uniq.tolist(), i_counts.tolist()):
                if args.id_counter_cap and len(iid_counter) \
                        >= args.id_counter_cap and u not in iid_counter:
                    continue
                iid_counter[u] = iid_counter.get(u, 0) + c

            # Q2 · user_int per-fid missing
            for fid, vs, dim in user_int_plan:
                col_name = f'user_int_feats_{fid}'
                if col_name not in rg.schema.names:
                    continue
                col = rg.column(col_name)
                if pa.types.is_list(col.type):
                    is_null = _list_any_scalar_null(col)
                else:
                    arr = _scalar_to_numpy(col)
                    is_null = arr <= 0
                n_null = int(is_null.sum())
                acc = user_int_acc[fid]
                acc[0] += n_null
                acc[1] += n
                acc[2] += int(labels[is_null].sum())
                acc[3] += int(labels[~is_null].sum())
            for fid, vs, dim in item_int_plan:
                col_name = f'item_int_feats_{fid}'
                if col_name not in rg.schema.names:
                    continue
                col = rg.column(col_name)
                if pa.types.is_list(col.type):
                    is_null = _list_any_scalar_null(col)
                else:
                    arr = _scalar_to_numpy(col)
                    is_null = arr <= 0
                n_null = int(is_null.sum())
                acc = item_int_acc[fid]
                acc[0] += n_null
                acc[1] += n
                acc[2] += int(labels[is_null].sum())
                acc[3] += int(labels[~is_null].sum())

            # Q4 / Q5 · per-domain seq length + time_diff
            ref_ts = _scalar_to_numpy(rg.column('timestamp'))

            # Q6 · CVR by day · group ref_ts // 86400 · vectorized bincount
            day_idx = (ref_ts.astype(np.int64) // 86400).astype(np.int64)
            day_min = int(day_idx.min()) if day_idx.size > 0 else 0
            day_max = int(day_idx.max()) if day_idx.size > 0 else -1
            for d in range(day_min, day_max + 1):
                mask = (day_idx == d)
                cnt = int(mask.sum())
                if cnt == 0:
                    continue
                pos = int(labels[mask].sum())
                acc = cvr_by_day.get(d)
                if acc is None:
                    cvr_by_day[d] = [cnt, pos]
                else:
                    acc[0] += cnt
                    acc[1] += pos

            # Q7 · CVR by hour-of-day · UTC hour 0-23
            hour_of_day = ((ref_ts.astype(np.int64) // 3600) % 24).astype(np.int64)
            for h in range(24):
                mask = (hour_of_day == h)
                cnt = int(mask.sum())
                if cnt == 0:
                    continue
                pos = int(labels[mask].sum())
                acc = cvr_by_hour.get(h)
                if acc is None:
                    cvr_by_hour[h] = [cnt, pos]
                else:
                    acc[0] += cnt
                    acc[1] += pos

            for dom, cfg in seq_cfg.items():
                ts_fid = dom_ts_fid.get(dom)
                if ts_fid is None:
                    continue
                ts_col_name = f'{dom_prefix[dom]}_{ts_fid}'
                if ts_col_name not in rg.schema.names:
                    continue
                ts_col = rg.column(ts_col_name)
                values, offsets = _pa_list_to_numpy(ts_col)
                # Per-row valid count = bincount of row_id for positive entries
                gt0 = values > 0
                row_lens = offsets[1:] - offsets[:-1]
                # Vectorized true length per row
                if values.size > 0:
                    row_id = np.repeat(np.arange(n, dtype=np.int64), row_lens)
                    if gt0.any():
                        true_len = np.bincount(row_id[gt0], minlength=n)
                    else:
                        true_len = np.zeros(n, dtype=np.int64)
                else:
                    true_len = np.zeros(n, dtype=np.int64)
                _reservoir_merge(seq_len_buf[dom], true_len,
                                 args.reservoir_size, rng)

                # Time diff: broadcast ref_ts over rows' valid entries
                if values.size > 0 and gt0.any():
                    valid_values = values[gt0]
                    valid_row = row_id[gt0]
                    diffs = ref_ts[valid_row] - valid_values
                    diffs = diffs[diffs >= 0]
                    if diffs.size > 0:
                        _reservoir_merge(time_diff_buf[dom], diffs,
                                         args.reservoir_size, rng)

            rg_seen += 1
            if rg_seen % 50 == 0:
                elapsed = time.time() - t_start
                rate = rg_seen / max(elapsed, 1e-6)
                rss_mb = _rss_mb()
                LOG.info('RG %d · rows=%d · pos=%d · %.1f RG/s · rss=%.0fMB',
                         rg_seen, total_rows, total_pos, rate, rss_mb)

        if args.max_row_groups and rg_seen >= args.max_row_groups:
            break

    elapsed = time.time() - t_start
    LOG.info('Scan complete · %d RG · %d rows · %d pos · %.1fs',
             rg_seen, total_rows, total_pos, elapsed)

    # ───── Finalize ─────

    def _freq_stats(counter: Dict[int, int]) -> Dict[str, Any]:
        if not counter:
            return {'distinct': 0}
        freqs = np.array(list(counter.values()), dtype=np.int64)
        return {
            'distinct': int(len(freqs)),
            'freq_1': float((freqs == 1).sum() / len(freqs)),
            'freq_2_10': float(((freqs >= 2) & (freqs <= 10)).sum()
                               / len(freqs)),
            'freq_gt10': float((freqs > 10).sum() / len(freqs)),
            'max_freq': int(freqs.max()),
            'mean_freq': float(freqs.mean()),
            'total_observations': int(freqs.sum()),
        }

    def _missing_summary(
        acc: Dict[int, List[int]],
        plan: List[Tuple[int, int, int]],
        prefix: str,
    ) -> List[Dict[str, Any]]:
        out = []
        for fid, vs, dim in plan:
            if fid not in acc:
                continue
            n_null, n_total, pos_null, pos_notnull = acc[fid]
            if n_total == 0:
                continue
            n_notnull = n_total - n_null
            cvr_null = (pos_null / n_null) if n_null > 0 else 0.0
            cvr_notnull = (pos_notnull / n_notnull) if n_notnull > 0 else 0.0
            out.append({
                'col': f'{prefix}_{fid}',
                'fid': fid,
                'vocab_size': vs,
                'dim': dim,
                'null_rate': float(n_null / n_total),
                'cvr_null': float(cvr_null),
                'cvr_notnull': float(cvr_notnull),
                'cvr_gap': float(cvr_null - cvr_notnull),
                'n_null': int(n_null),
                'n_total': int(n_total),
            })
        out.sort(key=lambda x: -abs(x['cvr_gap']))
        return out

    def _cvr_bucket_summary(
        acc: Dict[int, List[int]],
        time_unit: str,
    ) -> List[Dict[str, Any]]:
        """Q6/Q7 · CVR per time bucket (day / hour-of-day).

        Returns sorted list with absolute time, count, pos, cvr, and
        human-readable date/hour. ``time_unit`` chooses formatting:
          - 'day': bucket = days since Unix epoch · adds date_utc + date_beijing
          - 'hour': bucket = hour-of-day 0-23 (UTC) · adds hour_beijing (UTC+8)
        """
        out = []
        for bucket in sorted(acc.keys()):
            cnt, pos = acc[bucket]
            cvr = (pos / cnt) if cnt > 0 else 0.0
            entry: Dict[str, Any] = {
                'bucket': int(bucket),
                'count': int(cnt),
                'pos': int(pos),
                'cvr': float(cvr),
                'count_pct': float(cnt / total_rows) if total_rows else 0.0,
            }
            if time_unit == 'day':
                # bucket = days since 1970-01-01 (Thursday) UTC
                # Convert via stdlib for human-readable strings
                from datetime import datetime, timezone, timedelta
                ts_utc = datetime(1970, 1, 1, tzinfo=timezone.utc) + \
                         timedelta(days=bucket)
                ts_beijing = ts_utc + timedelta(hours=8)
                entry['date_utc'] = ts_utc.strftime('%Y-%m-%d')
                entry['date_beijing'] = ts_beijing.strftime('%Y-%m-%d')
                # Day-of-week: Mon=0 ... Sun=6
                entry['weekday'] = ts_utc.weekday()
            elif time_unit == 'hour':
                entry['hour_utc'] = int(bucket)
                entry['hour_beijing'] = int((bucket + 8) % 24)
            out.append(entry)
        return out

    summary = {
        'version': 1,
        'source': {
            'parquet_dir': args.parquet_dir,
            'num_row_groups_scanned': rg_seen,
            'total_rows': total_rows,
            'elapsed_sec': round(elapsed, 2),
            'max_row_groups': args.max_row_groups,
            'reservoir_size': args.reservoir_size,
        },
        'global': {
            'total_samples_observed': total_rows,
            'total_pos': total_pos,
            'global_cvr': (total_pos / total_rows) if total_rows else 0.0,
        },
        'q1_user_id': _freq_stats(uid_counter),
        'q1_item_id': _freq_stats(iid_counter),
        'q2_user_int_missing_top': _missing_summary(user_int_acc,
                                                     user_int_plan,
                                                     'user_int_feats'),
        'q2_item_int_missing_top': _missing_summary(item_int_acc,
                                                     item_int_plan,
                                                     'item_int_feats'),
        'q4_seq_length': {
            d: _quantiles(_reservoir_finalize(seq_len_buf[d],
                                              args.reservoir_size, rng))
            for d in seq_cfg
        },
        'q5_time_diff': {
            d: _quantiles(_reservoir_finalize(time_diff_buf[d],
                                              args.reservoir_size, rng))
            for d in seq_cfg
        },
        'q6_cvr_by_day': _cvr_bucket_summary(cvr_by_day, time_unit='day'),
        'q7_cvr_by_hour': _cvr_bucket_summary(cvr_by_hour, time_unit='hour'),
    }

    # -------------------------------------------------------------------
    # Output protocol — order matters for platform log recovery.
    #
    # Platform Logs UI tail-truncates at 1000 rows. All later output pushes
    # earlier output out of the window. Therefore the base64(gzip(JSON))
    # blob MUST be the absolute last thing we print.
    #
    # Order:
    #   1. Persist JSON to disk (best-effort; Ckpt/Log paths not downloadable
    #      per 2026-05-13 user confirmation — but no cost to write them).
    #   2. Pretty summary to stdout (~30 lines, human-readable tail of log).
    #   3. Blob metadata line (raw / gz / b64 / lines — size budget guard).
    #   4. <<<EDA_BLOB_START>>> + base64 chunks + <<<EDA_BLOB_END>>>.
    #   5. <<<EDA_EOF>>> (single-line final marker — if UI shows this,
    #      the entire blob survived truncation).
    # -------------------------------------------------------------------

    # (1) Persist JSON — local `--out` always, platform paths if env set.
    _persist_json(summary, args.out)

    # (2) Pretty summary (writes to stdout via plain print, 20-40 lines).
    _print_pretty_summary(summary)

    # (3/4/5) Blob + sentinels + EOF marker — ABSOLUTE LAST output.
    if args.stdout_blob:
        _emit_stdout_blob(summary, chunk_size=args.blob_chunk_size)


def _persist_json(summary: Dict[str, Any], out_path: str) -> None:
    """Write `eda_summary.json` to --out, $TRAIN_LOG_PATH, $TRAIN_CKPT_PATH.

    All paths best-effort; failures are logged but don't abort (blob stdout
    path is the primary recovery channel). The platform UI does not currently
    allow browsing these directories (confirmed 2026-05-13), but future
    platform updates may expose them — writing costs ~0.
    """
    for target in _resolve_persist_targets(out_path):
        try:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            with open(target, 'w') as f:
                json.dump(summary, f, indent=2)
            LOG.info('Wrote %s', target)
        except OSError as e:
            LOG.warning('Could not write %s: %s', target, e)


def _resolve_persist_targets(out_path: str) -> List[str]:
    targets = [out_path]
    for env_name, fname in [('TRAIN_LOG_PATH', 'eda_summary.json'),
                             ('TRAIN_CKPT_PATH', 'eda_summary.json')]:
        env_val = os.environ.get(env_name)
        if env_val:
            cand = os.path.join(env_val, fname)
            if cand not in targets:
                targets.append(cand)
    return targets


def _print_pretty_summary(summary: Dict[str, Any]) -> None:
    """Concise human-readable summary — 20-40 lines, platform Logs safe.

    Goes BEFORE the blob so if blob is truncated the user still gets signal.
    Uses plain print (not LOG) so format is stable for grep/diff.
    """
    src = summary['source']
    glb = summary['global']
    print('')
    print('#' * 72)
    print('## EDA SUMMARY · scan=%d RG · rows=%d · pos=%d (cvr=%.5f) · %.1fs'
          % (src['num_row_groups_scanned'], src['total_rows'],
             glb['total_pos'], glb['global_cvr'], src['elapsed_sec']))
    print('#' * 72)

    q1u = summary['q1_user_id']; q1i = summary['q1_item_id']
    if q1u.get('distinct'):
        print('# Q1 user_id: distinct=%d freq1=%.3f freq2-10=%.3f max=%d'
              % (q1u['distinct'], q1u['freq_1'], q1u['freq_2_10'],
                 q1u['max_freq']))
    if q1i.get('distinct'):
        print('# Q1 item_id: distinct=%d freq1=%.3f freq2-10=%.3f max=%d'
              % (q1i['distinct'], q1i['freq_1'], q1i['freq_2_10'],
                 q1i['max_freq']))

    print('# Q2 user_int top-5 by |cvr_gap|:')
    for item in summary['q2_user_int_missing_top'][:5]:
        print('  %s null=%.4f cvr_null=%.5f cvr_notnull=%.5f gap=%+.5f'
              % (item['col'], item['null_rate'], item['cvr_null'],
                 item['cvr_notnull'], item['cvr_gap']))
    print('# Q2 item_int top-5 by |cvr_gap|:')
    for item in summary['q2_item_int_missing_top'][:5]:
        print('  %s null=%.4f cvr_null=%.5f cvr_notnull=%.5f gap=%+.5f'
              % (item['col'], item['null_rate'], item['cvr_null'],
                 item['cvr_notnull'], item['cvr_gap']))

    print('# Q4 seq length (p50/p95/p99):')
    for dom, q in summary['q4_seq_length'].items():
        if q.get('n', 0) == 0:
            print('  %s: empty' % dom); continue
        qs = q['quantiles']
        print('  %s: n=%d p50=%.0f p95=%.0f p99=%.0f p99.9=%.0f max=%d'
              % (dom, q['n'], qs['p50'], qs['p95'], qs['p99'], qs['p99.9'],
                 q['max']))
    print('# Q5 time_diff (seconds; p25/p50/p75/p95):')
    for dom, q in summary['q5_time_diff'].items():
        if q.get('n', 0) == 0:
            print('  %s: empty' % dom); continue
        qs = q['quantiles']
        print('  %s: n=%d p25=%.0f p50=%.0f p75=%.0f p95=%.0f p99=%.0f'
              % (dom, q['n'], qs['p25'], qs['p50'], qs['p75'], qs['p95'],
                 qs['p99']))

    # Q6 · CVR by day (UTC days since epoch · with 北京 date for 元宵节 audit)
    q6 = summary.get('q6_cvr_by_day', [])
    if q6:
        print('# Q6 CVR by day (UTC date · 北京 date · count_pct · cvr):')
        for entry in q6:
            print('  %s · 北京 %s · weekday=%d · cnt=%d (%.1f%%) · pos=%d · cvr=%.5f'
                  % (entry['date_utc'], entry['date_beijing'],
                     entry['weekday'], entry['count'],
                     entry['count_pct'] * 100, entry['pos'], entry['cvr']))

    # Q7 · CVR by hour-of-day (UTC + 北京 hour for daypart audit)
    q7 = summary.get('q7_cvr_by_hour', [])
    if q7:
        print('# Q7 CVR by hour-of-day (UTC · 北京 hour · count_pct · cvr):')
        for entry in q7:
            print('  UTC %02d · 北京 %02d · cnt=%d (%.1f%%) · pos=%d · cvr=%.5f'
                  % (entry['hour_utc'], entry['hour_beijing'],
                     entry['count'], entry['count_pct'] * 100,
                     entry['pos'], entry['cvr']))

    print('#' * 72)
    sys.stdout.flush()


def _emit_stdout_blob(summary: Dict[str, Any], chunk_size: int = 120) -> None:
    """Final stdout output: blob sentinels + chunks + EOF marker.

    NOTHING may print after EOF_MARKER or the blob may get truncated by the
    platform Logs UI. Returns without any trailing print().
    """
    raw = json.dumps(summary, separators=(',', ':'), default=str).encode('utf-8')
    gz = gzip.compress(raw, compresslevel=9)
    b64 = base64.b64encode(gz).decode('ascii')
    n_chunks = (len(b64) + chunk_size - 1) // chunk_size
    # Approx lines for sizing: sentinels + n_chunks + eof + meta ≈ n_chunks+4.
    approx_lines = n_chunks + 4
    print('## BLOB: raw=%dB gz=%dB b64=%dchars chunk=%d lines=%d'
          % (len(raw), len(gz), len(b64), chunk_size, approx_lines))
    if approx_lines > 800:
        print('## WARN: blob lines=%d > 800 (platform tail 1000). '
              'Decode tail might miss progress logs.' % approx_lines)
    print(BLOB_START)
    for i in range(0, len(b64), chunk_size):
        print(b64[i:i + chunk_size])
    print(BLOB_END)
    print(EOF_MARKER)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
