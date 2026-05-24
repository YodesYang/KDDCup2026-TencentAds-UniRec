"""Log-only time-window backtest audit.

This EDA targets the public test clock window observed from eval logs:
Beijing 2026-03-04 09:49:51 to 14:49:07. It scans train timestamps and
labels, then prints compact tables that can be copied from the platform Logs
UI. No files are written.
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pyarrow.parquet as pq


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
LOG = logging.getLogger("eda_time_window_backtest")
print = functools.partial(print, flush=True)


Stat = List[float]  # rows, positives


def _find_parquet_files(data_dir: Path) -> List[Path]:
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")
    return files


def _to_numpy(col) -> np.ndarray:
    return col.to_numpy(zero_copy_only=False)


def _bj_epoch(local_time: str) -> int:
    """Parse Beijing-local timestamp and return UTC epoch seconds."""
    dt = datetime.strptime(local_time, "%Y-%m-%d %H:%M:%S")
    return int(dt.replace(tzinfo=timezone.utc).timestamp()) - 8 * 3600


def _fmt_utc(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_bj(ts: int) -> str:
    return datetime.fromtimestamp(
        int(ts) + 8 * 3600, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")


def _bj_day_hour_second(ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bj = ts.astype(np.int64) + 8 * 3600
    day = bj // 86400
    sec = bj % 86400
    hour = sec // 3600
    return day.astype(np.int64), hour.astype(np.int64), sec.astype(np.int64)


def _day_to_date(day: int) -> str:
    return datetime.fromtimestamp(
        int(day) * 86400, tz=timezone.utc
    ).strftime("%Y-%m-%d")


def _date_to_day(date_str: str) -> int:
    return int(
        datetime.strptime(date_str, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        // 86400
    )


def _add(stats: Dict[Tuple, Stat], key: Tuple, labels: np.ndarray) -> None:
    st = stats[key]
    st[0] += float(labels.size)
    st[1] += float(labels.sum())


def _stat_row(label: str, st: Stat, total_rows: float | None = None) -> Tuple[str, str, str, str]:
    rows, pos = st
    row_pct = rows / total_rows if total_rows else 0.0
    cvr = pos / rows if rows else 0.0
    return (f"{label:>20}", f"{int(rows):>9}", f"{100.0 * row_pct:7.3f}%", f"{cvr:9.6f}")


def _print_table(title: str, rows: List[Tuple[str, str, str, str]], limit: int = 200) -> None:
    print("")
    print(title)
    print("              bucket      rows row_pct       cvr")
    for row in rows[:limit]:
        print(" ".join(row))


def _rg_time_split(rows: List[Tuple[int, int, int]], valid_ratio: float) -> Tuple[int, int, int, int, int, int, int]:
    if not rows:
        raise RuntimeError("No row-group timestamp rows collected")
    ordered = sorted(rows, key=lambda x: x[1])  # max(timestamp) asc
    n_valid = max(1, int(len(ordered) * valid_ratio))
    train = ordered[:-n_valid]
    valid = ordered[-n_valid:]
    train_min = min(x[0] for x in train)
    train_max = max(x[1] for x in train)
    valid_min = min(x[0] for x in valid)
    valid_max = max(x[1] for x in valid)
    train_rows = sum(x[2] for x in train)
    valid_rows = sum(x[2] for x in valid)
    overlap = train_max - valid_min
    return train_min, train_max, valid_min, valid_max, train_rows, valid_rows, overlap


def main() -> None:
    ap = argparse.ArgumentParser(description="EDA-71 time-window backtest audit")
    ap.add_argument("--data_dir", default=os.environ.get("TRAIN_DATA_PATH", "data"))
    ap.add_argument("--max_row_groups", type=int, default=0, help="0 means full scan")
    ap.add_argument("--valid_ratio", type=float, default=0.05)
    ap.add_argument("--test_window_start_bj", default="2026-03-04 09:49:51")
    ap.add_argument("--test_window_end_bj", default="2026-03-04 14:49:07")
    ap.add_argument("--min_window_rows", type=int, default=10000)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = _find_parquet_files(data_dir)
    LOG.info("[plan] files=%d data_dir=%s max_row_groups=%d", len(files), data_dir, args.max_row_groups)

    test_start_ts = _bj_epoch(args.test_window_start_bj)
    test_end_ts = _bj_epoch(args.test_window_end_bj)
    test_start_bj_sec = (test_start_ts + 8 * 3600) % 86400
    test_end_bj_sec = (test_end_ts + 8 * 3600) % 86400
    target_day = _date_to_day(args.test_window_start_bj.split()[0])

    by_day: Dict[Tuple, Stat] = defaultdict(lambda: [0.0, 0.0])
    by_hour: Dict[Tuple, Stat] = defaultdict(lambda: [0.0, 0.0])
    by_test_clock_window_day: Dict[Tuple, Stat] = defaultdict(lambda: [0.0, 0.0])
    by_pre_window_day: Dict[Tuple, Stat] = defaultdict(lambda: [0.0, 0.0])
    by_daypart: Dict[Tuple, Stat] = defaultdict(lambda: [0.0, 0.0])
    rg_ranges: List[Tuple[int, int, int]] = []

    total_rows = 0
    total_pos = 0
    ts_min: int | None = None
    ts_max: int | None = None
    exact_public_train_rows = 0
    exact_public_train_pos = 0

    t0 = time.time()
    rg_seen = 0
    for fp_i, fp in enumerate(files, start=1):
        pf = pq.ParquetFile(str(fp))
        for rg_idx in range(pf.num_row_groups):
            if args.max_row_groups and rg_seen >= args.max_row_groups:
                break
            tbl = pf.read_row_group(rg_idx, columns=["timestamp", "label_type"])
            ts = _to_numpy(tbl.column("timestamp")).astype(np.int64)
            labels = (_to_numpy(tbl.column("label_type")).astype(np.int64) == 2).astype(np.float32)
            if ts.size == 0:
                rg_seen += 1
                continue

            mn = int(ts.min())
            mx = int(ts.max())
            ts_min = mn if ts_min is None else min(ts_min, mn)
            ts_max = mx if ts_max is None else max(ts_max, mx)
            rg_ranges.append((mn, mx, int(ts.size)))

            total_rows += int(ts.size)
            pos = int(labels.sum())
            total_pos += pos

            day, hour, sec = _bj_day_hour_second(ts)
            for d in np.unique(day):
                mask = day == d
                _add(by_day, (int(d),), labels[mask])
                window_mask = mask & (sec >= test_start_bj_sec) & (sec <= test_end_bj_sec)
                if window_mask.any():
                    _add(by_test_clock_window_day, (int(d),), labels[window_mask])
                pre_mask = mask & (sec < test_start_bj_sec)
                if pre_mask.any():
                    _add(by_pre_window_day, (int(d),), labels[pre_mask])

            for h in range(24):
                mask = hour == h
                if mask.any():
                    _add(by_hour, (h,), labels[mask])

            dayparts = {
                "00_05": sec < 6 * 3600,
                "06_08": (sec >= 6 * 3600) & (sec < 9 * 3600),
                "09_14": (sec >= 9 * 3600) & (sec <= 14 * 3600 + 3599),
                "15_20": (sec >= 15 * 3600) & (sec <= 20 * 3600 + 3599),
                "21_23": sec >= 21 * 3600,
            }
            for name, mask in dayparts.items():
                if mask.any():
                    _add(by_daypart, (name,), labels[mask])

            exact_mask = (ts >= test_start_ts) & (ts <= test_end_ts)
            if exact_mask.any():
                exact_public_train_rows += int(exact_mask.sum())
                exact_public_train_pos += int(labels[exact_mask].sum())

            rg_seen += 1
        if fp_i % 50 == 0 or fp_i == len(files) or (args.max_row_groups and rg_seen >= args.max_row_groups):
            LOG.info("[scan] files=%d/%d rg=%d rows=%d wall=%.1fs", fp_i, len(files), rg_seen, total_rows, time.time() - t0)
        if args.max_row_groups and rg_seen >= args.max_row_groups:
            break

    if ts_min is None or ts_max is None:
        raise RuntimeError("No rows scanned")

    train_min, train_max, valid_min, valid_max, train_rg_rows, valid_rg_rows, rg_overlap = _rg_time_split(
        rg_ranges, float(args.valid_ratio)
    )

    print("")
    print("========== EDA-71 · TIME-WINDOW BACKTEST AUDIT ==========")
    print(f"TOTAL_ROWS={total_rows:,} TOTAL_POS={total_pos:,} CVR={total_pos / max(total_rows, 1):.6f}")
    print(f"TRAIN_RANGE_UTC=[{ts_min},{ts_max}] { _fmt_utc(ts_min) } -> { _fmt_utc(ts_max) }")
    print(f"TRAIN_RANGE_BJ={_fmt_bj(ts_min)} -> {_fmt_bj(ts_max)}")
    print(f"PUBLIC_TEST_WINDOW_BJ={args.test_window_start_bj} -> {args.test_window_end_bj}")
    print(f"PUBLIC_TEST_WINDOW_UTC=[{test_start_ts},{test_end_ts}] {_fmt_utc(test_start_ts)} -> {_fmt_utc(test_end_ts)}")
    print(f"TRAIN_MAX_TO_PUBLIC_START_SEC={test_start_ts - ts_max}")
    print(f"EXACT_PUBLIC_WINDOW_ROWS_IN_TRAIN={exact_public_train_rows} POS={exact_public_train_pos}")

    print("")
    print("RG_TIME_SPLIT_DIAGNOSTIC")
    print(f"valid_ratio={args.valid_ratio:.4f} train_rows={train_rg_rows:,} valid_rows={valid_rg_rows:,}")
    print(f"rg_train_BJ={_fmt_bj(train_min)} -> {_fmt_bj(train_max)}")
    print(f"rg_valid_BJ={_fmt_bj(valid_min)} -> {_fmt_bj(valid_max)}")
    print(f"rg_overlap_sec=train_max-valid_min={rg_overlap}")

    day_rows = [
        _stat_row(_day_to_date(k[0]), st, float(total_rows))
        for k, st in sorted(by_day.items())
    ]
    _print_table("DATE_SUMMARY_BJ", day_rows)

    hour_rows = [
        _stat_row(f"hour_{k[0]:02d}", st, float(total_rows))
        for k, st in sorted(by_hour.items())
    ]
    _print_table("HOUR_SUMMARY_BJ", hour_rows)

    daypart_rows = [
        _stat_row(k[0], st, float(total_rows))
        for k, st in sorted(by_daypart.items())
    ]
    _print_table("DAYPART_SUMMARY_BJ", daypart_rows)

    print("")
    print("PUBLIC_CLOCK_WINDOW_BY_DATE")
    print("                date      rows row_pct       cvr  day_cvr  pre_window_cvr")
    viable_windows = 0
    for k, st in sorted(by_test_clock_window_day.items()):
        d = k[0]
        rows, pos = st
        if rows >= args.min_window_rows:
            viable_windows += 1
        day_st = by_day.get((d,), [0.0, 0.0])
        pre_st = by_pre_window_day.get((d,), [0.0, 0.0])
        cvr = pos / rows if rows else 0.0
        day_cvr = day_st[1] / day_st[0] if day_st[0] else 0.0
        pre_cvr = pre_st[1] / pre_st[0] if pre_st[0] else 0.0
        print(
            f"{_day_to_date(d):>20} {int(rows):>9} "
            f"{100.0 * rows / max(total_rows, 1):7.3f}% {cvr:9.6f} "
            f"{day_cvr:9.6f} {pre_cvr:14.6f}"
        )

    target_window_st = by_test_clock_window_day.get((target_day,), [0.0, 0.0])
    print("")
    print("FINAL_DECISION")
    print(f"target_day={_day_to_date(target_day)}")
    print(f"target_clock_window_train_rows={int(target_window_st[0])}")
    print(f"pseudo_public_windows_ge_{args.min_window_rows}_rows={viable_windows}")
    print(f"rg_time_split_overlap_sec={rg_overlap}")
    print(f"exact_public_train_rows={exact_public_train_rows}")
    print(f"recommend_clean_window_backtest={str(viable_windows >= 2).lower()}")
    print(f"recommend_more_absolute_time_dense={str(False).lower()}")
    print("<<<EDA_71_EOF>>>")


if __name__ == "__main__":
    main()
