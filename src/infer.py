"""Evaluation entry point for PCVRHyFormer.

This mirrors the training-side model and dataset construction from
``train.py`` and is required for checkpoints trained with synthetic dense
features, because platform default inference code may not know how to rebuild
those in-memory features.
"""

import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from dataset import PCVRParquetDataset, FeatureSchema, NUM_TIME_BUCKETS, parse_bucket_boundaries, parse_int_csv
from model import ModelInput, PCVRHyFormer


def _build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    specs: List[Tuple[int, int, int]] = []
    for _, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def _parse_seq_max_lens(raw: Any) -> Dict[str, int]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {str(k): int(v) for k, v in raw.items()}
    out: Dict[str, int] = {}
    for pair in str(raw).split(","):
        if not pair.strip():
            continue
        key, value = pair.split(":")
        out[key.strip()] = int(value.strip())
    return out


def _find_model_dir(model_output_path: str) -> str:
    root = Path(model_output_path)
    if (
        (root / "model.pt").exists()
        and (root / "train_config.json").exists()
        and (root / "schema.json").exists()
    ):
        return str(root)

    candidates = []
    for model_path in glob.glob(str(root / "**" / "model.pt"), recursive=True):
        model_dir = Path(model_path).parent
        if (model_dir / "train_config.json").exists() and (model_dir / "schema.json").exists():
            candidates.append(model_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No model.pt + train_config.json + schema.json found under {root}")

    candidates.sort(key=lambda p: (".best_model" not in p.name, str(p)))
    return str(candidates[0])


def _resolve_ns_groups_path(model_dir: str, cfg: Dict[str, Any]) -> str:
    raw = cfg.get("ns_groups_json") or ""
    if not raw:
        return ""
    candidates = []
    if os.path.isabs(raw):
        candidates.append(raw)
    candidates.append(os.path.join(model_dir, raw))
    candidates.append(os.path.join(model_dir, os.path.basename(raw)))
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return ""


def _resolve_history_cvr_path(model_dir: str, cfg: Dict[str, Any]) -> str:
    raw = cfg.get("history_cvr_cache_path") or ""
    if not raw:
        return ""
    candidates = []
    if os.path.isabs(raw):
        candidates.append(raw)
    candidates.append(os.path.join(model_dir, raw))
    candidates.append(os.path.join(model_dir, os.path.basename(raw)))
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return ""


def _parse_disable_seq_fids(raw: Any) -> Optional[List[int]]:
    """Parse ``disable_seq_fids`` from ``train_config.json``.

    Accepts a comma-separated string (what ``train.py`` writes) or a
    list (future-compat). Returns ``None`` when no fids are requested
    so ``PCVRParquetDataset`` takes its default "no-op" branch and we
    stay bit-identical to pre-T23 infer behaviour.
    """
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        parts = [s.strip() for s in raw.split(",") if s.strip()]
        if not parts:
            return None
        return [int(p) for p in parts]
    if isinstance(raw, (list, tuple)):
        out = [int(x) for x in raw]
        return out or None
    raise ValueError(
        f"Unexpected type for disable_seq_fids: {type(raw).__name__}")


def _resolve_bucket_boundaries(cfg: Dict[str, Any]):
    """T38 · Resolve time bucket boundaries from train_config.json.

    Priority:
    1. ``fitted_per_domain_bucket_boundaries`` (T38) · Dict[domain, List[int]]
       written by train.py when --enable_per_domain_buckets was set. Returned
       as-is to PCVRParquetDataset which will validate per-domain length.
    2. ``time_bucket_boundaries`` (T30) · csv string in train_config. Parsed
       to np.ndarray of length 63.
    3. None (default ladder) when neither is supplied.

    Back-compat: pre-T38 ckpts have neither key set (or
    ``fitted_per_domain_bucket_boundaries=None``). They keep using the T30
    csv path or the default ladder, bit-identical to before.
    """
    import numpy as np
    fitted = cfg.get("fitted_per_domain_bucket_boundaries", None)
    if fitted is not None and isinstance(fitted, dict) and len(fitted) > 0:
        return {
            d: np.asarray(arr, dtype=np.int64) for d, arr in fitted.items()
        }
    return parse_bucket_boundaries(cfg.get("time_bucket_boundaries", ""))


def _load_ns_groups(
    path: str,
    user_schema: FeatureSchema,
    item_schema: FeatureSchema,
) -> Tuple[List[List[int]], List[List[int]]]:
    if not path:
        return (
            [[i] for i in range(len(user_schema.entries))],
            [[i] for i in range(len(item_schema.entries))],
        )

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    user_fid_to_idx = {
        fid: i for i, (fid, _, _) in enumerate(user_schema.entries)
    }
    item_fid_to_idx = {
        fid: i for i, (fid, _, _) in enumerate(item_schema.entries)
    }
    user_groups = [
        [user_fid_to_idx[f] for f in fids]
        for fids in cfg["user_ns_groups"].values()
    ]
    item_groups = [
        [item_fid_to_idx[f] for f in fids]
        for fids in cfg["item_ns_groups"].values()
    ]
    return user_groups, item_groups


class _TestEdaState:
    """Main-process accumulator for test-set EDA (Option D · 2026-05-14).

    Replaces the dataset-internal _eda_state which required num_workers=0
    (worker process held dataset copies, main process saw empty state).
    By accumulating directly on collated batch tensors in the main loop,
    we keep num_workers=8 (full IO parallelism) AND collect EDA.

    Uses dataset's read-only attrs (schema entries, seq_domains, _seq_maxlen,
    _bucket_boundaries) which are identical across worker copies and main.

    Wall-time impact estimated +1~3% (counter+reservoir update on already-
    loaded numpy arrays).
    """

    def __init__(
        self,
        user_int_entries: List[Tuple[int, int, int]],
        item_int_entries: List[Tuple[int, int, int]],
        seq_domains: List[str],
        seq_maxlen: Dict[str, int],
        reservoir_size: int = 20000,
        per_fid_value_cap: int = 5000,
        id_counter_cap: int = 200000,
        seq_item_fid_plan: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        seq_item_id_top_k: int = 50000,
    ) -> None:
        # Plans: (fid, offset, length) for each user/item int feat
        self.user_int_plan = list(user_int_entries)
        self.item_int_plan = list(item_int_entries)
        self.seq_domains = list(seq_domains)
        self.seq_maxlen = dict(seq_maxlen)
        self.reservoir_size = reservoir_size
        self.per_fid_value_cap = per_fid_value_cap
        self.id_counter_cap = id_counter_cap

        # A2 (5/17 EXP-067) · Per (domain, fid) item-id frequency for the
        # high-vocab fids only. Used downstream (offline cross-reference with
        # train-side EDA) to compute test↔train seq item id overlap rate ·
        # which determines whether the seq encoder pathway has any signal in
        # the 100% user cold-start scenario. seq_item_fid_plan maps each
        # domain to a list of (slot, fid) tuples where slot is the index
        # along batch[domain] axis 1 (= sideinfo_fids order). Only set up
        # for fids with vocab > seq_item_id_min_vocab (1M).
        self.seq_item_fid_plan = dict(seq_item_fid_plan or {})
        self.seq_item_id_top_k = int(seq_item_id_top_k)

        # Accumulators (main-process-local, no worker copies)
        from collections import defaultdict
        self.total_samples = 0
        self.user_id_counter: Dict[str, int] = defaultdict(int)
        self.item_id_counter: Dict[str, int] = defaultdict(int)
        # per fid: [n_null, n_total]  (CVR slots stay 0 · no labels)
        self.user_int_missing: Dict[int, List[int]] = defaultdict(lambda: [0, 0, 0, 0])
        self.item_int_missing: Dict[int, List[int]] = defaultdict(lambda: [0, 0, 0, 0])
        # reservoirs: {'samples': list, 'seen': int}
        def _make_reservoir() -> Dict[str, Any]:
            return {'samples': [], 'seen': 0}
        self.seq_length_reservoir: Dict[str, Dict[str, Any]] = {
            d: _make_reservoir() for d in seq_domains
        }
        self.time_diff_reservoir: Dict[str, Dict[str, Any]] = {
            d: _make_reservoir() for d in seq_domains
        }
        self.abs_ts_reservoir: Dict[str, Any] = _make_reservoir()
        self.user_int_value_reservoir: Dict[int, Dict[str, Any]] = defaultdict(_make_reservoir)
        self.item_int_value_reservoir: Dict[int, Dict[str, Any]] = defaultdict(_make_reservoir)
        # A2 · seq item id counter · keyed by (domain, fid) → Counter
        # (item_id_value → freq). Capped at seq_item_id_top_k unique values
        # per (domain, fid) to bound memory.
        self.seq_item_id_counter: Dict[Tuple[str, int], Dict[int, int]] = (
            defaultdict(lambda: defaultdict(int)))

    @classmethod
    def from_dataset(
        cls,
        dataset: PCVRParquetDataset,
        seq_item_id_min_vocab: int = 1_000_000,
        seq_item_id_top_k: int = 50_000,
        id_counter_cap: int = 200_000,
    ) -> "_TestEdaState":
        """Build accumulator using dataset's read-only attrs (no shared state).

        A2 (5/17 EXP-067): Auto-discover high-vocab fids in each seq domain
        (vocab > seq_item_id_min_vocab · default 1M) which are most likely
        item_id-like fields. Used downstream to compute test↔train seq item
        id overlap rate (decides whether seq encoder pathway has signal in
        100% user cold-start scenario).
        """
        # Build seq_item_fid_plan: domain → [(slot_idx, fid), ...] for
        # high-vocab fids only. slot_idx is the index along batch[domain]
        # axis 1 (= sideinfo_fids order, see dataset.py line 952-955).
        seq_item_fid_plan: Dict[str, List[Tuple[int, int]]] = {}
        for domain in dataset.seq_domains:
            sideinfo = dataset.sideinfo_fids.get(domain, [])
            vocab_map = dataset.seq_vocab_sizes.get(domain, {})
            plan_list: List[Tuple[int, int]] = []
            for slot_idx, fid in enumerate(sideinfo):
                vs = int(vocab_map.get(fid, 0))
                if vs > seq_item_id_min_vocab:
                    plan_list.append((slot_idx, fid))
            if plan_list:
                seq_item_fid_plan[domain] = plan_list
        return cls(
            user_int_entries=dataset.user_int_schema.entries,
            item_int_entries=dataset.item_int_schema.entries,
            seq_domains=list(dataset.seq_domains),
            seq_maxlen={d: dataset._seq_maxlen.get(d, 0) for d in dataset.seq_domains},
            seq_item_fid_plan=seq_item_fid_plan,
            seq_item_id_top_k=seq_item_id_top_k,
            id_counter_cap=id_counter_cap,
        )

    def _reservoir_extend(self, entry: Dict[str, Any], new_values, max_size: int) -> None:
        """Streaming reservoir sampling (matches dataset.py logic)."""
        import numpy as np
        new_values = np.asarray(new_values).ravel()
        if new_values.size == 0:
            return
        seen_before = entry['seen']
        samples = entry['samples']
        for v in new_values:
            entry['seen'] += 1
            if len(samples) < max_size:
                samples.append(int(v))
            else:
                # reservoir replace
                idx = int(np.random.randint(0, entry['seen']))
                if idx < max_size:
                    samples[idx] = int(v)

    def update(self, batch: Dict[str, Any]) -> None:
        """Accumulate EDA stats from a collated batch dict.

        Expected batch keys (subset of dataset.collate output):
        - 'user_int_feats': (B, total_user_int_dim) torch tensor
        - 'item_int_feats': (B, total_item_int_dim) torch tensor
        - 'timestamp':       (B,) torch tensor (int64)
        - 'user_id':         list of str
        - For each domain in seq_domains:
            - batch[domain]:           (B, num_seq_fids, max_len) tensor
            - batch[f"{domain}_len"]:  (B,) tensor (true length before pad)
        """
        import numpy as np

        user_int = batch['user_int_feats'].cpu().numpy()  # (B, D_user)
        item_int = batch['item_int_feats'].cpu().numpy()  # (B, D_item)
        timestamps = batch['timestamp'].cpu().numpy().astype(np.int64)
        user_ids = batch['user_id']  # list of str
        B = len(user_ids)

        self.total_samples += B

        # Q1: user_id / item_id frequency (capped)
        if len(self.user_id_counter) < self.id_counter_cap:
            for uid in user_ids:
                if len(self.user_id_counter) >= self.id_counter_cap:
                    break
                self.user_id_counter[str(uid)] += 1

        # No item_id in batch dict (model uses item_int_feats); skip item_id_counter

        # Q2: per-fid null_rate for user_int / item_int
        for fid, offset, length in self.user_int_plan:
            fid_vals = user_int[:, offset:offset + length]
            is_null = (fid_vals == 0).all(axis=1)
            n_null = int(is_null.sum())
            acc = self.user_int_missing[fid]
            acc[0] += n_null
            acc[1] += B
        for fid, offset, length in self.item_int_plan:
            fid_vals = item_int[:, offset:offset + length]
            is_null = (fid_vals == 0).all(axis=1)
            n_null = int(is_null.sum())
            acc = self.item_int_missing[fid]
            acc[0] += n_null
            acc[1] += B

        # Q4: per-domain seq length reservoir (using batch[f"{d}_len"])
        for domain in self.seq_domains:
            len_key = f"{domain}_len"
            if len_key not in batch:
                continue
            lengths = batch[len_key].cpu().numpy().astype(np.int64)
            self._reservoir_extend(
                self.seq_length_reservoir[domain], lengths, self.reservoir_size,
            )

        # A2 (5/17 EXP-067): per-(domain, fid) item id frequency for
        # high-vocab fids only. batch[domain] is (B, n_feats, max_len)
        # where the n_feats axis follows sideinfo_fids order from
        # dataset.sideinfo_fids[domain]. Non-zero values are item ids
        # (zero = padding). Cap unique values per (domain, fid) at
        # seq_item_id_top_k via streaming top-K logic (here we just
        # accumulate; final top_k slice happens in finalize()).
        for domain, plan_list in self.seq_item_fid_plan.items():
            seq_tensor = batch.get(domain)
            if seq_tensor is None:
                continue
            seq_arr = seq_tensor.cpu().numpy()  # (B, n_feats, max_len)
            for slot_idx, fid in plan_list:
                if slot_idx >= seq_arr.shape[1]:
                    continue
                slot_vals = seq_arr[:, slot_idx, :].ravel()
                slot_vals = slot_vals[slot_vals > 0]  # drop padding
                if slot_vals.size == 0:
                    continue
                key = (domain, fid)
                counter = self.seq_item_id_counter[key]
                # Memory bound: stop accumulating new keys once we hit
                # 5x the top_k cap (top_k slice picks the highest-freq
                # subset · we need some headroom to track non-top items).
                hard_cap = self.seq_item_id_top_k * 5
                for v in slot_vals:
                    v_int = int(v)
                    if v_int in counter:
                        counter[v_int] += 1
                    elif len(counter) < hard_cap:
                        counter[v_int] = 1
                    # else: silently drop (the cap is much larger than
                    # top_k so the top-K result is unaffected by drops)

        # Q6: impression absolute timestamps → train↔test temporal gap
        ts_pos = timestamps[timestamps > 0]
        if ts_pos.size > 0:
            self._reservoir_extend(
                self.abs_ts_reservoir, ts_pos, self.reservoir_size,
            )

        # Q7: per-fid first non-zero value reservoir (covariate shift)
        for fid, offset, length in self.user_int_plan:
            fid_vals = user_int[:, offset:offset + length]
            # first nonzero per row
            nonzero_mask = fid_vals != 0
            row_has_any = nonzero_mask.any(axis=1)
            if not row_has_any.any():
                continue
            first_nz_idx = nonzero_mask.argmax(axis=1)
            row_first = fid_vals[np.arange(B), first_nz_idx][row_has_any]
            self._reservoir_extend(
                self.user_int_value_reservoir[fid],
                row_first,
                self.per_fid_value_cap,
            )
        for fid, offset, length in self.item_int_plan:
            fid_vals = item_int[:, offset:offset + length]
            nonzero_mask = fid_vals != 0
            row_has_any = nonzero_mask.any(axis=1)
            if not row_has_any.any():
                continue
            first_nz_idx = nonzero_mask.argmax(axis=1)
            row_first = fid_vals[np.arange(B), first_nz_idx][row_has_any]
            self._reservoir_extend(
                self.item_int_value_reservoir[fid],
                row_first,
                self.per_fid_value_cap,
            )

    def finalize(self) -> Dict[str, Any]:
        """Compute summary dict with quantiles + missing rates · for blob."""
        import numpy as np

        def _quantiles(samples: List[int]) -> Optional[Dict[str, float]]:
            if not samples:
                return None
            arr = np.asarray(samples)
            qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
            vals = np.quantile(arr, qs)
            return {
                'p1': float(vals[0]), 'p5': float(vals[1]), 'p25': float(vals[2]),
                'p50': float(vals[3]), 'p75': float(vals[4]),
                'p95': float(vals[5]), 'p99': float(vals[6]),
                'mean': float(arr.mean()), 'min': float(arr.min()),
                'max': float(arr.max()), 'n': int(len(samples)),
            }

        # User_int / item_int missing rates
        user_int_missing_summary = {
            f'user_int_feats_{fid}': {
                'n_null': v[0], 'n_total': v[1],
                'null_rate': (v[0] / v[1]) if v[1] else 0.0,
            }
            for fid, v in sorted(self.user_int_missing.items())
        }
        item_int_missing_summary = {
            f'item_int_feats_{fid}': {
                'n_null': v[0], 'n_total': v[1],
                'null_rate': (v[0] / v[1]) if v[1] else 0.0,
            }
            for fid, v in sorted(self.item_int_missing.items())
        }

        # Per-domain seq length quantiles
        seq_length_summary = {
            d: _quantiles(self.seq_length_reservoir[d]['samples'])
            for d in self.seq_domains
        }

        # Q6: absolute timestamp quantiles
        abs_ts_summary = _quantiles(self.abs_ts_reservoir['samples'])

        # Q7: per-fid value distribution (only top-N high-MI fids to keep blob small)
        # We dump quantiles for ALL fids; per-fid blob ~50 fids × 10 floats = 500 numbers.
        user_int_value_summary = {
            f'user_int_feats_{fid}': _quantiles(v['samples'])
            for fid, v in sorted(self.user_int_value_reservoir.items())
        }
        item_int_value_summary = {
            f'item_int_feats_{fid}': _quantiles(v['samples'])
            for fid, v in sorted(self.item_int_value_reservoir.items())
        }

        # User_id frequency histogram (top 100 buckets only · keep blob bounded)
        from collections import Counter
        if self.user_id_counter:
            freqs = Counter(self.user_id_counter.values())
            user_id_freq_hist = {
                str(k): int(v) for k, v in sorted(freqs.most_common(20))
            }
        else:
            user_id_freq_hist = {}

        # A2 (5/17 EXP-067) · seq item id top-K frequency per (domain, fid)
        # Only emit fids with non-empty counters. Each entry = top_k items
        # by frequency · format = list of [item_id, freq] for compactness.
        seq_item_id_summary: Dict[str, Any] = {}
        seq_item_id_distinct: Dict[str, int] = {}
        for (domain, fid), counter in self.seq_item_id_counter.items():
            if not counter:
                continue
            sorted_items = sorted(
                counter.items(), key=lambda kv: -kv[1])[:self.seq_item_id_top_k]
            key = f"{domain}_fid_{fid}"
            seq_item_id_summary[key] = [
                [int(item_id), int(freq)] for item_id, freq in sorted_items
            ]
            seq_item_id_distinct[key] = len(counter)

        return {
            'eda_format': 'test_eda_full_v1',
            'total_samples': self.total_samples,
            'user_id_distinct': len(self.user_id_counter),
            'user_id_counter_cap': self.id_counter_cap,
            'user_id_freq_hist_top20': user_id_freq_hist,
            'user_int_missing': user_int_missing_summary,
            'item_int_missing': item_int_missing_summary,
            'seq_length': seq_length_summary,
            'abs_timestamp': abs_ts_summary,
            'user_int_value': user_int_value_summary,
            'item_int_value': item_int_value_summary,
            # A2 cross-reference fields
            'seq_item_id_top_k': seq_item_id_summary,
            'seq_item_id_top_k_limit': self.seq_item_id_top_k,
            'seq_item_id_distinct_per_fid': seq_item_id_distinct,
            'seq_item_fid_plan': {
                d: [[int(s), int(f)] for s, f in plan]
                for d, plan in self.seq_item_fid_plan.items()
            },
        }


def _make_model_input(batch: Dict[str, Any], device: torch.device) -> ModelInput:
    seq_domains = batch["_seq_domains"]
    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    for domain in seq_domains:
        seq_data[domain] = batch[domain].to(device, non_blocking=True)
        seq_lens[domain] = batch[f"{domain}_len"].to(device, non_blocking=True)
        time_key = f"{domain}_time_bucket"
        if time_key in batch:
            seq_time_buckets[domain] = batch[time_key].to(device, non_blocking=True)
        else:
            bsz = batch[domain].shape[0]
            seq_len = batch[domain].shape[2]
            seq_time_buckets[domain] = torch.zeros(
                bsz, seq_len, dtype=torch.long, device=device)

    return ModelInput(
        user_int_feats=batch["user_int_feats"].to(device, non_blocking=True),
        item_int_feats=batch["item_int_feats"].to(device, non_blocking=True),
        user_dense_feats=batch["user_dense_feats"].to(device, non_blocking=True),
        item_dense_feats=batch["item_dense_feats"].to(device, non_blocking=True),
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
    )


def _load_state_dict(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        sd = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(path, map_location=device)
    # Strip "_orig_mod." prefix if present. Such prefix appears when a ckpt
    # was saved from a torch.compile()-wrapped model (ADR-008 T27 code had
    # an early bug where EarlyStopping received the compiled wrapper and
    # torch.save'd its state_dict directly). infer.py itself never compiles,
    # so its model.state_dict() keys are always un-prefixed; renaming at
    # load time keeps ancient ckpts evaluate-able without retraining.
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()
        }
    return sd


def _get_int_env(name: str, default: int, min_value: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logging.warning(
            f"[test-eda] invalid {name}={raw!r}; using default {default}")
        return default
    if value < min_value:
        logging.warning(
            f"[test-eda] {name}={value} < {min_value}; using default {default}")
        return default
    return value


def _get_eda_blob_mode() -> str:
    mode = os.environ.get("EDA_BLOB_MODE", "compact").strip().lower()
    if mode in {"compact", "small", "summary"}:
        return "compact"
    if mode in {"full", "raw"}:
        return "full"
    logging.warning(
        f"[test-eda] unknown EDA_BLOB_MODE={mode!r}; using compact")
    return "compact"


def _compact_test_eda_summary(
    summary: Dict[str, Any],
    seq_item_top_k_per_fid: int,
) -> Dict[str, Any]:
    """Drop fields that make platform Logs UI tail-truncate the EDA blob."""
    seq_item_top = summary.get("seq_item_id_top_k") or {}
    compact_seq_item_top = {
        key: values[:seq_item_top_k_per_fid]
        for key, values in seq_item_top.items()
    }
    return {
        "eda_format": "test_eda_compact_v1",
        "total_samples": summary.get("total_samples"),
        "user_id": {
            "distinct_capped": summary.get("user_id_distinct"),
            "counter_cap": summary.get("user_id_counter_cap"),
            "freq_hist_top20": summary.get("user_id_freq_hist_top20", {}),
            "note": "raw user_id values are not emitted by public EDA utilities",
        },
        "abs_timestamp": summary.get("abs_timestamp"),
        "seq_length": summary.get("seq_length", {}),
        "user_int_missing": summary.get("user_int_missing", {}),
        "item_int_missing": summary.get("item_int_missing", {}),
        "user_int_value": summary.get("user_int_value", {}),
        "item_int_value": summary.get("item_int_value", {}),
        "seq_item_fid_plan": summary.get("seq_item_fid_plan", {}),
        "seq_item_id_distinct_per_fid": summary.get(
            "seq_item_id_distinct_per_fid", {}),
        "seq_item_id_top_k_limit": min(
            int(summary.get("seq_item_id_top_k_limit", seq_item_top_k_per_fid)),
            seq_item_top_k_per_fid,
        ),
        "seq_item_id_top_k": compact_seq_item_top,
    }


def _fmt_ts_bj(ts_value: Any) -> str:
    if ts_value is None:
        return "NA"
    try:
        from datetime import datetime, timezone, timedelta
        tz_bj = timezone(timedelta(hours=8))
        return datetime.fromtimestamp(float(ts_value), tz=timezone.utc).astimezone(
            tz_bj).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:  # pylint: disable=broad-except
        return str(ts_value)


def _print_test_eda_compact_summary(summary: Dict[str, Any], mode: str) -> None:
    abs_ts = summary.get("abs_timestamp") or {}
    seq_length = summary.get("seq_length") or {}
    seq_item = summary.get("seq_item_id_top_k") or {}
    print()
    print("#" * 72)
    print("## TEST EDA COMPACT SUMMARY")
    print(f"# mode={mode} total_samples={summary.get('total_samples')}")
    print(
        "# user_id_distinct_capped="
        f"{summary.get('user_id', {}).get('distinct_capped', summary.get('user_id_distinct'))} "
        "cap="
        f"{summary.get('user_id', {}).get('counter_cap', summary.get('user_id_counter_cap'))}"
    )
    if abs_ts:
        print(
            "# abs_timestamp "
            f"min={abs_ts.get('min')} p50={abs_ts.get('p50')} max={abs_ts.get('max')} "
            f"bj_min={_fmt_ts_bj(abs_ts.get('min'))} "
            f"bj_max={_fmt_ts_bj(abs_ts.get('max'))}"
        )
    for domain, stats in seq_length.items():
        if not stats:
            continue
        print(
            f"# seq_length {domain}: "
            f"p50={stats.get('p50')} p95={stats.get('p95')} "
            f"max={stats.get('max')} n={stats.get('n')}"
        )
    print(
        "# seq_item_id_top_k "
        f"fids={len(seq_item)} limit={summary.get('seq_item_id_top_k_limit')}"
    )
    print("#" * 72)


def _emit_summary_blob(summary: Dict[str, Any], mode: str) -> None:
    import base64
    import gzip
    import sys

    try:
        payload = json.dumps(summary, sort_keys=True).encode("utf-8")
        gz = gzip.compress(payload, compresslevel=9)
        b64 = base64.b64encode(gz).decode("ascii")
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning(f"[test-eda] blob encode failed: {exc!r}")
        return

    chunk_size = 120
    chunks = [b64[i:i + chunk_size] for i in range(0, len(b64), chunk_size)]

    sys.stdout.flush()
    _print_test_eda_compact_summary(summary, mode)
    print()
    print("<<<EDA_BLOB_START>>>")
    for ch in chunks:
        print(ch)
    print("<<<EDA_BLOB_END>>>")
    sys.stdout.flush()
    logging.info(
        f"[test-eda] blob emitted: mode={mode}, {len(b64)} chars, "
        f"{len(chunks)} chunks, raw {len(payload)} bytes")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    model_output_path = os.environ["MODEL_OUTPUT_PATH"]
    eval_data_path = os.environ["EVAL_DATA_PATH"]
    eval_result_path = os.environ["EVAL_RESULT_PATH"]

    model_dir = _find_model_dir(model_output_path)
    cfg_path = os.path.join(model_dir, "train_config.json")
    schema_path = os.path.join(model_dir, "schema.json")
    model_path = os.path.join(model_dir, "model.pt")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    logging.info(f"Using model dir: {model_dir}")
    logging.info(f"Loaded train_config from {cfg_path}")
    logging.info(f"Using schema: {schema_path}")

    seq_max_lens = _parse_seq_max_lens(cfg.get("seq_max_lens"))
    batch_size = int(cfg.get("batch_size", 256))
    num_workers = int(os.environ.get("EVAL_NUM_WORKERS", cfg.get("num_workers", 8)))

    # ── Test-set EDA dump (default OFF · 2026-05-20 eval-speed hotfix) ─────
    # Refactored from dataset-internal _update_test_eda_stats to main-process
    # accumulator. Main process iterates batch dicts (already produced by
    # workers) and updates EDA state directly. No worker-state copy issue,
    # no num_workers=0 forcing. Still, score-max evaluations should not pay
    # the extra CPU/logging cost by default; set EDA_BLOB=1 only for
    # deliberate diagnostic evals.
    enable_test_eda = os.environ.get("EDA_BLOB", "0") != "0"
    eda_blob_mode = _get_eda_blob_mode()
    eda_seq_item_top_k = _get_int_env(
        "EDA_SEQ_ITEM_TOP_K",
        200 if eda_blob_mode == "compact" else 50_000,
        min_value=1,
    )
    eda_user_id_cap = _get_int_env("EDA_USER_ID_CAP", 200_000, min_value=1)
    eda_state: Optional[_TestEdaState] = None
    # eda_state is initialized AFTER dataset is built (needs dataset's
    # read-only schema attrs). See after PCVRParquetDataset(...) call below.

    dataset = PCVRParquetDataset(
        parquet_path=eval_data_path,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        clip_vocab=bool(cfg.get("clip_vocab", True)),
        is_training=False,
        enable_count_features=bool(cfg.get("enable_count_features", False)),
        enable_seq_stats_features=bool(cfg.get("enable_seq_stats_features", False)),
        enable_history_cvr_features=bool(cfg.get("enable_history_cvr_features", False)),
        history_cvr_cache_path=_resolve_history_cvr_path(model_dir, cfg),
        history_cvr_time_mode=str(cfg.get("history_cvr_time_mode", "timestamp_cutoff")),
        history_cvr_cutoff_sec=int(cfg.get("history_cvr_cutoff_sec", 86400)),
        history_cvr_available_lag_sec=int(cfg.get("history_cvr_available_lag_sec", 0)),
        history_cvr_prior_strength=float(cfg.get("history_cvr_prior_strength", 20.0)),
        enable_time_of_day_features=bool(cfg.get("enable_time_of_day_features", False)),
        enable_beijing_time_features=bool(cfg.get("enable_beijing_time_features", False)),
        enable_beijing_time_v2_features=bool(
            cfg.get("enable_beijing_time_v2_features", False)),
        enable_hour_only_features=bool(cfg.get("enable_hour_only_features", False)),
        disable_seq_fids=_parse_disable_seq_fids(cfg.get("disable_seq_fids", "")),
        bucket_boundaries=_resolve_bucket_boundaries(cfg),
        enable_is_missing=bool(cfg.get("enable_is_missing", False)),
        is_missing_user_int_fids=parse_int_csv(cfg.get("is_missing_user_int_fids", "")),
        is_missing_item_int_fids=parse_int_csv(cfg.get("is_missing_item_int_fids", "")),
        enable_eda_dump=False,  # Option D: EDA accumulated in main process via _TestEdaState
    )

    if enable_test_eda:
        eda_state = _TestEdaState.from_dataset(
            dataset,
            seq_item_id_top_k=eda_seq_item_top_k,
            id_counter_cap=eda_user_id_cap,
        )
        logging.info(
            "[test-eda] EDA_BLOB=1 · mode=%s · "
            "seq_item_id_top_k=%d · user_id_cap=%d · "
            "main-process accumulator · compatible with num_workers>0",
            eda_blob_mode, eda_seq_item_top_k, eda_user_id_cap,
        )

    loader_kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs,
    )

    user_ns_groups, item_ns_groups = _load_ns_groups(
        _resolve_ns_groups_path(model_dir, cfg),
        dataset.user_int_schema,
        dataset.item_int_schema,
    )

    # T34 DCN cross fids · resolve raw schema fid → list index in
    # feature_specs (matches train.py:794-821 logic). 兼容两种 train_config
    # storage 格式:
    #   - raw csv str (e.g. '1,49')  ← train.py vars(args) 写入的格式 (常见)
    #   - list of int                 ← 早期 model_args 直接写入 (rare)
    #   - already-mapped indices      ← 极少 · 当 raw fids 与 list indices 巧合相等
    # 通过 user_int_schema.entries 重建 raw_fid → list_index 映射 · 严格
    # 区分 "raw fid (在 schema 中)" vs "list index (在 feature_specs 中)"。
    def _resolve_dcn_fids(
        raw_value: Any,
        schema_entries: List[Tuple[int, int, int]],
        side: str,
    ) -> Optional[List[int]]:
        if raw_value is None:
            return None
        if isinstance(raw_value, str):
            try:
                raw_fids = [int(x.strip()) for x in raw_value.split(',') if x.strip()]
            except ValueError as e:
                raise ValueError(
                    f"dcn_cross_{side}_fids '{raw_value}' parse error: {e}"
                ) from e
            fid_to_idx = {entry[0]: i for i, entry in enumerate(schema_entries)}
            unknown = [f for f in raw_fids if f not in fid_to_idx]
            if unknown:
                raise ValueError(
                    f"dcn_cross_{side}_fids contains unknown fids "
                    f"{unknown} (available: {sorted(fid_to_idx.keys())})"
                )
            return [fid_to_idx[f] for f in raw_fids]
        elif isinstance(raw_value, list):
            values = [int(x) for x in raw_value]
        else:
            raise ValueError(
                f"dcn_cross_{side}_fids type {type(raw_value).__name__} "
                f"unexpected · expected str or list"
            )
        # Newer train.py writes list indices, while older configs may contain
        # raw fid strings. When a JSON list is present, prefer index semantics
        # because it mirrors train.py's train_config override and avoids
        # re-mapping indices like [0, 5] as raw fid numbers during evaluation.
        if all(0 <= v < len(schema_entries) for v in values):
            return values
        fid_to_idx = {entry[0]: i for i, entry in enumerate(schema_entries)}
        if all(v in fid_to_idx for v in values):
            return [fid_to_idx[v] for v in values]
        raise ValueError(
            f"dcn_cross_{side}_fids contains values {values} that are "
            f"neither valid list indices [0,{len(schema_entries) - 1}] nor "
            f"known raw fids (available: {sorted(fid_to_idx.keys())})"
        )

    enable_dcn_cross = bool(cfg.get("enable_dcn_cross", False))
    if enable_dcn_cross:
        dcn_user_fid_indices = _resolve_dcn_fids(
            cfg.get("dcn_cross_user_fids"),
            dataset.user_int_schema.entries,
            "user",
        )
        dcn_item_fid_indices = _resolve_dcn_fids(
            cfg.get("dcn_cross_item_fids"),
            dataset.item_int_schema.entries,
            "item",
        )
    else:
        dcn_user_fid_indices = None
        dcn_item_fid_indices = None

    model_args = {
        "user_int_feature_specs": _build_feature_specs(
            dataset.user_int_schema, dataset.user_int_vocab_sizes),
        "item_int_feature_specs": _build_feature_specs(
            dataset.item_int_schema, dataset.item_int_vocab_sizes),
        "user_dense_dim": dataset.user_dense_schema.total_dim,
        "item_dense_dim": dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": int(cfg.get("d_model", 64)),
        "emb_dim": int(cfg.get("emb_dim", 64)),
        "num_queries": int(cfg.get("num_queries", 1)),
        "num_hyformer_blocks": int(cfg.get("num_hyformer_blocks", 2)),
        "num_heads": int(cfg.get("num_heads", 4)),
        "seq_encoder_type": cfg.get("seq_encoder_type", "transformer"),
        "hidden_mult": int(cfg.get("hidden_mult", 4)),
        "dropout_rate": float(cfg.get("dropout_rate", 0.01)),
        "seq_top_k": int(cfg.get("seq_top_k", 50)),
        "seq_causal": bool(cfg.get("seq_causal", False)),
        "action_num": int(cfg.get("action_num", 1)),
        "num_time_buckets": NUM_TIME_BUCKETS if cfg.get("use_time_buckets", True) else 0,
        "rank_mixer_mode": cfg.get("rank_mixer_mode", "full"),
        "rank_mixer_ffn_mode": cfg.get("rank_mixer_ffn_mode", "shared"),
        "use_rope": bool(cfg.get("use_rope", False)),
        "rope_base": float(cfg.get("rope_base", 10000.0)),
        "emb_skip_threshold": int(cfg.get("emb_skip_threshold", 0)),
        "seq_id_threshold": int(cfg.get("seq_id_threshold", 10000)),
        "ns_tokenizer_type": cfg.get("ns_tokenizer_type", "rankmixer"),
        "user_ns_tokens": int(cfg.get("user_ns_tokens", 0)),
        "item_ns_tokens": int(cfg.get("item_ns_tokens", 0)),
        "use_onetrans": bool(cfg.get("use_onetrans", False)),
        "onetrans_top_k": int(cfg.get("onetrans_top_k", 50)),
        "seq_hash_vocab": int(cfg.get("seq_hash_vocab", 0)),
        "item_conditioned_query": bool(cfg.get("item_conditioned_query", False)),
        # 5/17 fix: train_config.json 实际 key 是 `enable_q_init_item`
        # (来自 train.py vars(args)) · 老旧别名 `q_init_item` fallback 兼容
        # (T32 commit 6979fd7 设计期望 key=q_init_item · 但 vars(args) 实际
        # 写入 enable_q_init_item)。bug 实证: M68 evaluation state_dict
        # mismatch (q_init_item_proj keys not loaded · model 不创建模块)。
        "q_init_item": bool(
            cfg.get("enable_q_init_item",
                    cfg.get("q_init_item", False))),
        "multi_emb_k": int(cfg.get("multi_emb_k", 1)),
        "target_item_seq_injection": str(cfg.get("target_item_seq_injection", "off")),
        "target_inject_alpha_init": float(cfg.get("target_inject_alpha_init", 0.0)),
        "enable_dense_bypass": bool(cfg.get("enable_dense_bypass", False)),
        "enable_din_interest": bool(cfg.get("enable_din_interest", False)),
        "din_interest_source": str(cfg.get("din_interest_source", "raw")),
        "din_interest_merge": str(cfg.get("din_interest_merge", "compact")),
        "enable_tin_interest": bool(cfg.get("enable_tin_interest", False)),
        "tin_time_alpha_init": float(cfg.get("tin_time_alpha_init", 1.0)),
        "enable_din_integrated": bool(cfg.get("enable_din_integrated", False)),
        "din_integrated_alpha_init": float(cfg.get("din_integrated_alpha_init", 0.1)),
        "enable_nlir_gating": bool(cfg.get("enable_nlir_gating", False)),
        "enable_fafe": bool(cfg.get("enable_fafe", False)),
        "enable_dcn_cross": enable_dcn_cross,
        # T34 · DCN cross list indices (resolved from raw fids by
        # _resolve_dcn_fids above · no longer trust cfg raw value).
        "dcn_cross_user_fids": dcn_user_fid_indices,
        "dcn_cross_item_fids": dcn_item_fid_indices,
        "dcn_cross_layers": int(cfg.get("dcn_cross_layers", 2)),
        "enable_global_token": bool(cfg.get("enable_global_token", False)),
        "enable_ue_split": bool(cfg.get("enable_ue_split", False)),
        "ue_offset": int(cfg.get("ue_offset", 0)),
        "ue_dim": int(cfg.get("ue_dim", 256)),
        "ue_slices": cfg.get("ue_slices", None),
        "ue_split_separate_tokens": bool(
            cfg.get("ue_split_separate_tokens", False)),
        "enable_ue_int_bilinear": bool(
            cfg.get("enable_ue_int_bilinear", False)),
        "ue_int_bilinear_alpha_init": float(
            cfg.get("ue_int_bilinear_alpha_init", 0.5)),
        "enable_ue_item_interaction": bool(
            cfg.get("enable_ue_item_interaction", False)),
        "ue_item_interaction_alpha_init": float(
            cfg.get("ue_item_interaction_alpha_init", 1.0)),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Building PCVRHyFormer with cfg: {model_args}")
    model = PCVRHyFormer(**model_args).to(device)
    model.load_state_dict(_load_state_dict(model_path, device))
    model.eval()

    predictions: Dict[str, float] = {}
    seen_rows = 0
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            model_input = _make_model_input(batch, device)
            logits, _ = model.predict(model_input)
            if logits.ndim == 2 and logits.shape[1] > 1:
                logits = logits[:, -1]
            else:
                logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
            probs_list = probs.detach().cpu().tolist()
            for user_id, prob in zip(batch["user_id"], probs_list):
                predictions[str(user_id)] = float(prob)
            seen_rows += len(probs_list)
            # Option D: accumulate EDA in main process (no worker copy issue).
            # Wraps in try/except so EDA failure NEVER fails inference.
            if eda_state is not None:
                try:
                    eda_state.update(batch)
                except Exception as exc:  # pylint: disable=broad-except
                    logging.warning(f"[test-eda] update failed at step {step}: {exc!r}")
                    eda_state = None  # disable for remaining steps
            if step % 100 == 0:
                logging.info(
                    f"Inference progress: step={step}, rows={seen_rows}, "
                    f"unique_user_ids={len(predictions)}")

    os.makedirs(eval_result_path, exist_ok=True)
    out_path = os.path.join(eval_result_path, "predictions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"predictions": predictions}, f)
    logging.info(
        f"Saved {len(predictions)} predictions from {seen_rows} rows to {out_path}")

    if eda_state is not None:
        _emit_test_eda_blob_from_state(eda_state, mode=eda_blob_mode)


def _emit_test_eda_blob_from_state(
    state: "_TestEdaState",
    mode: str = "compact",
) -> None:
    """Emit base64+gzip JSON blob of test-set EDA stats to stdout.

    Placed at the very END of stdout (after predictions are saved and after
    any other logging) so the platform Logs UI tail-1000-lines window
    contains the blob even if log volume is high. Sentinels make extraction
    by ``tools/decode_eda_blob.py`` reliable.

    Wraps in try/except so any EDA failure NEVER fails the overall
    evaluation (predictions.json is the contract; EDA is bonus).
    """
    try:
        summary = state.finalize()
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning(f"[test-eda] finalize failed: {exc!r}")
        return
    if summary is None or not summary.get('total_samples', 0):
        logging.warning("[test-eda] empty EDA state, skipping blob")
        return
    if mode == "compact":
        summary = _compact_test_eda_summary(
            summary,
            seq_item_top_k_per_fid=_get_int_env(
                "EDA_COMPACT_SEQ_ITEM_TOP_K", 200, min_value=1),
        )
    _emit_summary_blob(summary, mode)


def _emit_test_eda_blob(dataset: PCVRParquetDataset) -> None:
    """Legacy emit (kept for compat with non-refactored call sites)."""
    try:
        summary = dataset.finalize_eda()
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning(f"[test-eda] finalize_eda failed: {exc!r}")
        return
    if summary is None:
        logging.warning("[test-eda] EDA_BLOB=1 but no EDA state collected")
        return
    mode = _get_eda_blob_mode()
    if mode == "compact":
        summary = _compact_test_eda_summary(
            summary,
            seq_item_top_k_per_fid=_get_int_env(
                "EDA_COMPACT_SEQ_ITEM_TOP_K", 200, min_value=1),
        )
    _emit_summary_blob(summary, mode)


if __name__ == "__main__":
    main()
