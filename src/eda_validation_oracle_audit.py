"""Log-only audit for validation-window design.

EDA-71 showed that the public set is a narrow future window:
Beijing 2026-03-04 09:49:51 to 14:49:07. This script compares candidate
training-time validation windows against that public timestamp / sequence
shape, then prints a compact scoreboard for Logs UI copy-paste.
"""

from __future__ import annotations

import argparse
import functools
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
LOG = logging.getLogger("eda_validation_oracle_audit")
print = functools.partial(print, flush=True)


SEQ_TS_COLS = {
    "seq_a": "domain_a_seq_39",
    "seq_b": "domain_b_seq_67",
    "seq_c": "domain_c_seq_27",
    "seq_d": "domain_d_seq_26",
}
SEQ_MAX_LENS = {
    "seq_a": 256,
    "seq_b": 256,
    "seq_c": 512,
    "seq_d": 512,
}

PUBLIC_START_BJ = "2026-03-04 09:49:51"
PUBLIC_END_BJ = "2026-03-04 14:49:07"
PUBLIC_TS_Q = {
    "p01": "2026-03-04 10:27:19",
    "p05": "2026-03-04 11:29:18",
    "p25": "2026-03-04 12:14:59",
    "p50": "2026-03-04 12:57:28",
    "p75": "2026-03-04 13:43:41",
    "p95": "2026-03-04 14:28:16",
    "p99": "2026-03-04 14:41:39",
}
PUBLIC_SEQ = {
    "seq_a": {"mean": 218.2, "p50": 256.0, "p95": 256.0},
    "seq_b": {"mean": 199.0, "p50": 256.0, "p95": 256.0},
    "seq_c": {"mean": 348.7, "p50": 373.0, "p95": 512.0},
    "seq_d": {"mean": 455.8, "p50": 512.0, "p95": 512.0},
}


def _find_parquet_files(data_dir: Path) -> List[Path]:
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")
    return files


def _to_numpy(col) -> np.ndarray:
    return col.to_numpy(zero_copy_only=False)


def _bj_epoch(local_time: str) -> int:
    dt = datetime.strptime(local_time, "%Y-%m-%d %H:%M:%S")
    return int(dt.replace(tzinfo=timezone.utc).timestamp()) - 8 * 3600


def _fmt_bj(ts: int) -> str:
    return datetime.fromtimestamp(
        int(ts) + 8 * 3600, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_utc(ts: int) -> str:
    return datetime.fromtimestamp(
        int(ts), tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")


def _list_lengths(col: "pa.ChunkedArray") -> np.ndarray:
    merged = col.combine_chunks()
    if not pa.types.is_list(merged.type):
        return np.zeros(len(merged), dtype=np.int64)
    offsets = merged.offsets.to_numpy(zero_copy_only=False)
    return np.diff(offsets).astype(np.int64, copy=False)


class Reservoir:
    def __init__(self, cap: int, seed: int) -> None:
        self.cap = int(cap)
        self.rng = np.random.default_rng(seed)
        self.n_seen = 0
        self.values: List[float] = []

    def update(self, arr: np.ndarray) -> None:
        if arr.size == 0 or self.cap <= 0:
            self.n_seen += int(arr.size)
            return
        arr = np.asarray(arr).reshape(-1)
        free = max(0, self.cap - len(self.values))
        if free:
            take = min(free, int(arr.size))
            self.values.extend(arr[:take].astype(float).tolist())
            self.n_seen += take
            arr = arr[take:]
        for x in arr:
            self.n_seen += 1
            j = int(self.rng.integers(0, self.n_seen))
            if j < self.cap:
                self.values[j] = float(x)

    def array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64)

    def q(self, pct: float) -> float:
        arr = self.array()
        if arr.size == 0:
            return float("nan")
        return float(np.percentile(arr, pct))


@dataclass(frozen=True)
class WindowSpec:
    name: str
    start_ts: int
    end_ts: int
    note: str

    def mask(self, ts: np.ndarray) -> np.ndarray:
        return (ts >= self.start_ts) & (ts < self.end_ts)


class WindowStats:
    def __init__(self, spec: WindowSpec, reservoir_cap: int, seed: int) -> None:
        self.spec = spec
        self.rows = 0
        self.pos = 0
        self.ts_min: Optional[int] = None
        self.ts_max: Optional[int] = None
        self.time_norm = Reservoir(reservoir_cap, seed)
        self.user_counts: Dict[int, int] = {}
        self.item_counts: Dict[int, int] = {}
        self.seq_len: Dict[str, Dict[str, object]] = {
            d: {
                "n": 0,
                "sum": 0.0,
                "max": 0,
                "empty": 0,
                "sample": Reservoir(reservoir_cap, seed + i + 10),
            }
            for i, d in enumerate(SEQ_TS_COLS)
        }

    def update(
        self,
        ts: np.ndarray,
        labels: np.ndarray,
        user_id: Optional[np.ndarray],
        item_id: Optional[np.ndarray],
        seq_lengths: Dict[str, np.ndarray],
    ) -> None:
        mask = self.spec.mask(ts)
        if not mask.any():
            return
        idx = np.flatnonzero(mask)
        w_ts = ts[idx].astype(np.int64)
        n = int(w_ts.size)
        self.rows += n
        self.pos += int(labels[idx].sum())
        mn = int(w_ts.min())
        mx = int(w_ts.max())
        self.ts_min = mn if self.ts_min is None else min(self.ts_min, mn)
        self.ts_max = mx if self.ts_max is None else max(self.ts_max, mx)

        denom = max(1, int(self.spec.end_ts) - int(self.spec.start_ts))
        self.time_norm.update((w_ts - int(self.spec.start_ts)) / denom)

        if user_id is not None:
            uniq, cnt = np.unique(user_id[idx].astype(np.int64), return_counts=True)
            for v, c in zip(uniq.tolist(), cnt.tolist()):
                self.user_counts[int(v)] = self.user_counts.get(int(v), 0) + int(c)
        if item_id is not None:
            uniq, cnt = np.unique(item_id[idx].astype(np.int64), return_counts=True)
            for v, c in zip(uniq.tolist(), cnt.tolist()):
                self.item_counts[int(v)] = self.item_counts.get(int(v), 0) + int(c)

        for domain, lengths in seq_lengths.items():
            vals = lengths[idx].astype(np.float64)
            # Public sequence summaries come from the model/infer path after
            # max-length truncation, so compare train windows on the same scale.
            vals = np.minimum(vals, float(SEQ_MAX_LENS.get(domain, vals.max(initial=0))))
            st = self.seq_len[domain]
            st["n"] = int(st["n"]) + int(vals.size)
            st["sum"] = float(st["sum"]) + float(vals.sum())
            st["max"] = max(int(st["max"]), int(vals.max()) if vals.size else 0)
            st["empty"] = int(st["empty"]) + int((vals == 0).sum())
            sample = st["sample"]
            assert isinstance(sample, Reservoir)
            sample.update(vals)

    def _count_dist(self, counts: Dict[int, int]) -> Tuple[int, float, int]:
        if not counts:
            return 0, float("nan"), 0
        vals = np.asarray(list(counts.values()), dtype=np.int64)
        return len(counts), float((vals == 1).mean()), int(vals.max())

    def time_shape_l1(self, public_norm_q: Dict[str, float]) -> float:
        arr = self.time_norm.array()
        if arr.size == 0:
            return float("inf")
        diffs = []
        for name, target in public_norm_q.items():
            pct = float(name[1:])
            diffs.append(abs(float(np.percentile(arr, pct)) - target))
        return float(np.mean(diffs))

    def seq_distance(self) -> Tuple[float, float]:
        mean_diffs: List[float] = []
        p50_diffs: List[float] = []
        for domain, target in PUBLIC_SEQ.items():
            st = self.seq_len[domain]
            n = int(st["n"])
            if n <= 0:
                continue
            mean = float(st["sum"]) / n
            sample = st["sample"]
            assert isinstance(sample, Reservoir)
            p50 = sample.q(50)
            mean_diffs.append(abs(mean - target["mean"]) / max(target["mean"], 1.0))
            p50_diffs.append(abs(p50 - target["p50"]) / max(target["p50"], 1.0))
        if not mean_diffs:
            return float("inf"), float("inf")
        return float(np.mean(mean_diffs)), float(np.mean(p50_diffs))

    def row(self, public_norm_q: Dict[str, float]) -> Dict[str, object]:
        cvr = self.pos / self.rows if self.rows else float("nan")
        u_unique, u_singleton, u_max = self._count_dist(self.user_counts)
        i_unique, _i_singleton, i_max = self._count_dist(self.item_counts)
        seq_mean_dist, seq_p50_dist = self.seq_distance()
        time_dist = self.time_shape_l1(public_norm_q)
        user_penalty = 0.0 if math.isnan(u_singleton) else max(0.0, 0.95 - u_singleton)
        oracle_distance = time_dist + seq_mean_dist + 0.25 * seq_p50_dist + 0.10 * user_penalty
        return {
            "name": self.spec.name,
            "rows": self.rows,
            "pos": self.pos,
            "cvr": cvr,
            "start_bj": _fmt_bj(self.spec.start_ts),
            "end_bj": _fmt_bj(self.spec.end_ts),
            "time_q_l1": time_dist,
            "seq_mean_rel_l1": seq_mean_dist,
            "seq_p50_rel_l1": seq_p50_dist,
            "user_unique": u_unique,
            "user_singleton_frac": u_singleton,
            "user_max_freq": u_max,
            "item_unique": i_unique,
            "item_max_freq": i_max,
            "oracle_distance": oracle_distance,
            "note": self.spec.note,
        }

    def print_details(self) -> None:
        print("")
        print(f"WINDOW_DETAIL name={self.spec.name}")
        print(f"range_bj={_fmt_bj(self.spec.start_ts)} -> {_fmt_bj(self.spec.end_ts)} note={self.spec.note}")
        print(f"rows={self.rows} pos={self.pos} cvr={self.pos / self.rows if self.rows else float('nan'):.6f}")
        if self.rows:
            qs = self.time_norm.array()
            print(
                "time_norm_quantiles "
                f"p01={np.percentile(qs, 1):.4f} p05={np.percentile(qs, 5):.4f} "
                f"p25={np.percentile(qs, 25):.4f} p50={np.percentile(qs, 50):.4f} "
                f"p75={np.percentile(qs, 75):.4f} p95={np.percentile(qs, 95):.4f} "
                f"p99={np.percentile(qs, 99):.4f}"
            )
        print("seq_len_summary domain rows mean p50 p95 max empty_frac public_mean public_p50")
        for domain, target in PUBLIC_SEQ.items():
            st = self.seq_len[domain]
            n = int(st["n"])
            sample = st["sample"]
            assert isinstance(sample, Reservoir)
            if n <= 0:
                print(f"{domain:>6} {0:>8} nan nan nan 0 nan {target['mean']:.1f} {target['p50']:.1f}")
                continue
            mean = float(st["sum"]) / n
            print(
                f"{domain:>6} {n:>8} {mean:7.2f} {sample.q(50):7.2f} "
                f"{sample.q(95):7.2f} {int(st['max']):>4} "
                f"{int(st['empty']) / n:9.6f} {target['mean']:10.1f} {target['p50']:10.1f}"
            )


def _candidate_windows(public_start: int, public_end: int) -> List[WindowSpec]:
    start_clock = PUBLIC_START_BJ.split()[1]
    end_clock = PUBLIC_END_BJ.split()[1]

    def same_clock(date: str, name: str, note: str) -> WindowSpec:
        return WindowSpec(
            name=name,
            start_ts=_bj_epoch(f"{date} {start_clock}"),
            end_ts=_bj_epoch(f"{date} {end_clock}"),
            note=note,
        )

    return [
        same_clock("2026-03-03", "clk0303_m91", "current M91 clean pseudo-public"),
        same_clock("2026-03-02", "clk0302", "same-clock stability candidate"),
        same_clock("2026-03-01", "clk0301", "same-clock low-row stress candidate"),
        WindowSpec(
            "p5p95_0303",
            _bj_epoch("2026-03-03 11:29:18"),
            _bj_epoch("2026-03-03 14:28:16"),
            "same date as M91 but public p05-p95 clock mass",
        ),
        WindowSpec(
            "p25p75_0303",
            _bj_epoch("2026-03-03 12:14:59"),
            _bj_epoch("2026-03-03 13:43:41"),
            "same date as M91 but public interquartile clock mass",
        ),
        WindowSpec(
            "pre0304_all",
            _bj_epoch("2026-03-04 00:00:00"),
            public_start,
            "target day rows before public start",
        ),
        WindowSpec(
            "pre0304_last60m",
            public_start - 3600,
            public_start,
            "closest same-day hour before public start",
        ),
        WindowSpec(
            "pre0304_last30m",
            public_start - 1800,
            public_start,
            "closest same-day 30 minutes before public start",
        ),
    ]


def _public_norm_quantiles(public_start: int, public_end: int) -> Dict[str, float]:
    denom = max(1, public_end - public_start)
    return {
        name: (_bj_epoch(ts) - public_start) / denom
        for name, ts in PUBLIC_TS_Q.items()
    }


def _print_public_reference(public_start: int, public_end: int, public_norm_q: Dict[str, float]) -> None:
    print("")
    print("PUBLIC_REFERENCE")
    print(f"public_window_bj={PUBLIC_START_BJ} -> {PUBLIC_END_BJ}")
    print(f"public_window_utc=[{public_start},{public_end}] {_fmt_utc(public_start)} -> {_fmt_utc(public_end)}")
    print("public_time_norm_quantiles " + " ".join(f"{k}={v:.4f}" for k, v in public_norm_q.items()))
    print("public_seq_summary domain mean p50 p95")
    for d, st in PUBLIC_SEQ.items():
        print(f"{d:>6} {st['mean']:7.1f} {st['p50']:7.1f} {st['p95']:7.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="EDA-72 validation oracle audit")
    ap.add_argument("--data_dir", default=os.environ.get("TRAIN_DATA_PATH", "data"))
    ap.add_argument("--max_row_groups", type=int, default=0, help="0 means full scan")
    ap.add_argument("--reservoir_cap", type=int, default=500000)
    ap.add_argument("--min_window_rows", type=int, default=10000)
    ap.add_argument("--print_window_details", action="store_true", default=False)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = _find_parquet_files(data_dir)
    public_start = _bj_epoch(PUBLIC_START_BJ)
    public_end = _bj_epoch(PUBLIC_END_BJ)
    public_norm_q = _public_norm_quantiles(public_start, public_end)
    windows = _candidate_windows(public_start, public_end)
    stats = [
        WindowStats(spec, reservoir_cap=args.reservoir_cap, seed=20260519 + i * 101)
        for i, spec in enumerate(windows)
    ]

    columns = ["timestamp", "label_type", "user_id", "item_id"] + list(SEQ_TS_COLS.values())
    LOG.info(
        "[plan] files=%d data_dir=%s max_row_groups=%d windows=%s",
        len(files), data_dir, args.max_row_groups, ",".join(w.name for w in windows),
    )

    total_rows = 0
    total_pos = 0
    ts_min: Optional[int] = None
    ts_max: Optional[int] = None
    rg_seen = 0
    t0 = time.time()
    missing_cols: set[str] = set()
    for fp_i, fp in enumerate(files, start=1):
        pf = pq.ParquetFile(str(fp))
        available = set(pf.schema_arrow.names)
        read_cols = [c for c in columns if c in available]
        missing_cols.update(c for c in columns if c not in available)
        for rg_idx in range(pf.num_row_groups):
            if args.max_row_groups and rg_seen >= args.max_row_groups:
                break
            tbl = pf.read_row_group(rg_idx, columns=read_cols)
            ts = _to_numpy(tbl.column("timestamp")).astype(np.int64)
            labels = (_to_numpy(tbl.column("label_type")).astype(np.int64) == 2).astype(np.int8)
            user_id = (
                _to_numpy(tbl.column("user_id")).astype(np.int64)
                if "user_id" in tbl.column_names else None
            )
            item_id = (
                _to_numpy(tbl.column("item_id")).astype(np.int64)
                if "item_id" in tbl.column_names else None
            )
            seq_lengths = {
                domain: _list_lengths(tbl.column(col))
                for domain, col in SEQ_TS_COLS.items()
                if col in tbl.column_names
            }

            if ts.size:
                mn = int(ts.min())
                mx = int(ts.max())
                ts_min = mn if ts_min is None else min(ts_min, mn)
                ts_max = mx if ts_max is None else max(ts_max, mx)
            total_rows += int(ts.size)
            total_pos += int(labels.sum())
            for st in stats:
                st.update(ts, labels, user_id, item_id, seq_lengths)
            rg_seen += 1

        if fp_i % 50 == 0 or fp_i == len(files) or (args.max_row_groups and rg_seen >= args.max_row_groups):
            LOG.info("[scan] files=%d/%d rg=%d rows=%d wall=%.1fs", fp_i, len(files), rg_seen, total_rows, time.time() - t0)
        if args.max_row_groups and rg_seen >= args.max_row_groups:
            break

    if ts_min is None or ts_max is None:
        raise RuntimeError("No rows scanned")

    print("")
    print("========== EDA-72 · VALIDATION ORACLE AUDIT ==========")
    print(f"TOTAL_ROWS={total_rows:,} TOTAL_POS={total_pos:,} CVR={total_pos / max(total_rows, 1):.6f}")
    print(f"TRAIN_RANGE_BJ={_fmt_bj(ts_min)} -> {_fmt_bj(ts_max)}")
    print(f"TRAIN_MAX_TO_PUBLIC_START_SEC={public_start - ts_max}")
    if missing_cols:
        print("MISSING_COLUMNS=" + ",".join(sorted(missing_cols)))

    _print_public_reference(public_start, public_end, public_norm_q)

    rows = [st.row(public_norm_q) for st in stats]
    rows_by_score = sorted(rows, key=lambda r: (r["rows"] < args.min_window_rows, float(r["oracle_distance"])))

    print("")
    print("WINDOW_SCOREBOARD sorted_by=oracle_distance")
    header = (
        "name rows cvr time_q_l1 seq_mean_l1 seq_p50_l1 "
        "user_uniq user_single user_max item_uniq item_max oracle_dist note"
    )
    print(header)
    for r in rows_by_score:
        print(
            f"{str(r['name']):>16} {int(r['rows']):>8} {float(r['cvr']):.6f} "
            f"{float(r['time_q_l1']):.5f} {float(r['seq_mean_rel_l1']):.5f} "
            f"{float(r['seq_p50_rel_l1']):.5f} {int(r['user_unique']):>8} "
            f"{float(r['user_singleton_frac']):.6f} {int(r['user_max_freq']):>6} "
            f"{int(r['item_unique']):>8} {int(r['item_max_freq']):>6} "
            f"{float(r['oracle_distance']):.5f} {r['note']}"
        )

    print("")
    print("WINDOW_TIMESTAMP_ARGS")
    for st in stats:
        print(
            f"{st.spec.name}: start={st.spec.start_ts} end={st.spec.end_ts} "
            f"bj='{_fmt_bj(st.spec.start_ts)} -> {_fmt_bj(st.spec.end_ts)}'"
        )

    viable = [r for r in rows if int(r["rows"]) >= args.min_window_rows]
    same_clock_viable = [r for r in viable if str(r["name"]).startswith("clk")]
    same_day_viable = [r for r in viable if str(r["name"]).startswith("pre0304")]
    best_same_clock = min(same_clock_viable, key=lambda r: float(r["oracle_distance"])) if same_clock_viable else None
    best_same_day = min(same_day_viable, key=lambda r: float(r["oracle_distance"])) if same_day_viable else None

    print("")
    print("FINAL_RECOMMENDATION")
    print(f"min_window_rows={args.min_window_rows}")
    print(f"viable_window_count={len(viable)}")
    print(f"best_same_clock={best_same_clock['name'] if best_same_clock else 'NONE'}")
    print(f"best_same_day_pre_public={best_same_day['name'] if best_same_day else 'NONE'}")
    print(f"recommend_multi_window_gate={str(len(viable) >= 3 and best_same_clock is not None).lower()}")
    print("recommended_policy=Use same-clock clean windows as negative filters; use same-day pre-public windows as distribution sanity checks, not label oracles.")

    if args.print_window_details:
        for st in stats:
            st.print_details()

    print("<<<EDA_72_EOF>>>")


if __name__ == "__main__":
    main()
