"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python train.py [--num_epochs 10] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS, parse_bucket_boundaries, parse_int_csv
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_aux_valid_windows(raw: str) -> List[Tuple[str, int, int]]:
    """Parse ``name:start:end`` auxiliary validation windows from CLI."""
    if not raw or not raw.strip():
        return []
    windows: List[Tuple[str, int, int]] = []
    seen: set[str] = set()
    for idx, token in enumerate(raw.split(','), start=1):
        part = token.strip()
        if not part:
            continue
        fields = [x.strip() for x in part.split(':')]
        if len(fields) == 2:
            name = f"aux{idx}"
            start_raw, end_raw = fields
        elif len(fields) == 3:
            name, start_raw, end_raw = fields
        else:
            raise ValueError(
                "--aux_valid_windows entries must be 'name:start:end' or "
                f"'start:end'; got {part!r}")
        name = ''.join(
            ch if ch.isalnum() or ch in ('_', '-') else '_'
            for ch in name.strip()
        )
        if not name:
            name = f"aux{idx}"
        if name in seen:
            raise ValueError(
                f"duplicate auxiliary validation window name: {name}")
        seen.add(name)
        try:
            start_ts = int(start_raw)
            end_ts = int(end_raw)
        except ValueError as e:
            raise ValueError(
                f"aux validation window timestamps must be integers: "
                f"{part!r}") from e
        if end_ts <= start_ts:
            raise ValueError(
                f"aux validation window {name!r} has end <= start: "
                f"{start_ts}..{end_ts}")
        windows.append((name, start_ts, end_ts))
    return windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--early_stop_metric', type=str, default='auc',
                        choices=['auc', 'logloss'],
                        help="Metric used for early stopping: 'auc' (maximize, default) "
                             "or 'logloss' (minimize). Using logloss may preserve "
                             "better-calibrated checkpoints that AUC would skip.")
    parser.add_argument('--logloss_rebound_patience', type=int, default=1,
                        help="Co-monitor (LogLoss rebound) patience. When >0, training "
                             "stops once val_logloss exceeds the best-seen logloss for "
                             "this many consecutive validations, even if AUC is still "
                             "climbing. best_model remains AUC-selected. Default 1 "
                             "(stop on first rebound); set 0 to disable. Rationale: "
                             "2026-05-04 t13tf-loss-full LB incident.")
    parser.add_argument('--logloss_rebound_delta', type=float, default=0.0,
                        help="Absolute worsening of val_logloss beyond the best-seen "
                             "that counts as a rebound. 0.0 = any uptick trips the guard.")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--limit_train_rgs', type=int, default=0,
                        help='Fast-proxy switch: cap train Row Groups after the '
                             'normal split, keeping the tail closest to the '
                             'validation cut (0 = disabled)')
    parser.add_argument('--limit_valid_rgs', type=int, default=0,
                        help='Fast-proxy switch: cap validation Row Groups after '
                             'the normal split, keeping the first N validation '
                             'Row Groups (0 = disabled)')
    parser.add_argument('--valid_split_strategy', type=str, default='filename',
                        choices=['filename', 'time'],
                        help='How to order Row Groups before taking the tail '
                             'as validation: '
                             "'filename' (default, behavior-preserving) keeps "
                             'the sorted(glob)+RG-index order assumed by the '
                             'baseline; '
                             "'time' sorts RGs by max(timestamp) ASC so the "
                             'validation tail is strictly later than train, '
                             'matching the TAAC 2026 test-set time gap.')
    parser.add_argument('--holdout_ratio', type=float, default=0.0,
                        help='ADR-005 dual-valid switch: when > 0, carve the '
                             'latest holdout_ratio fraction of Row Groups as '
                             'a third disjoint split used purely as a '
                             'generalization probe (holdout AUC/LogLoss is '
                             'logged every epoch but never drives weight '
                             'updates or checkpoint selection). Requires '
                             "valid_split_strategy='time'. 0.0 (default) = "
                             'disabled, behavior-preserving. Typical: 0.05.')
    parser.add_argument('--valid_gap_ratio', type=float, default=0.0,
                        help='Funnel Redesign P3 (ADR-005 alt C): carve a '
                             'time-axis gap of this fraction of Row Groups '
                             'between the train tail and the dev head. Those '
                             'gap Row Groups are dropped entirely (never '
                             'loaded). Simulates the "remote-future" test-set '
                             'shift so dev AUC becomes a closer proxy for LB '
                             "AUC. Requires valid_split_strategy='time'. "
                             '0.0 (default) = disabled, behavior-preserving. '
                             'Typical: 0.10~0.15.')
    parser.add_argument('--id_mask_prob', type=float, default=0.0,
                        help='Item-ID Random Mask (community 0.85+ hint, '
                             '2026-05-04): per-token probability of dropping '
                             'a sequence token to 0 (padding) on the training '
                             'path only. Counteracts "dev up / LB down" by '
                             'forcing the model not to overfit specific IDs '
                             'that may be OOV in the future-dated test set. '
                             '0.0 (default) = disabled. Typical: 0.10~0.20.')
    parser.add_argument('--id_mask_seq_domains', type=str, default='',
                        help='Comma-separated sequence-domain letters to '
                             'apply --id_mask_prob to (e.g. "c" to only mask '
                             'domain_c_seq, the longest-tail vocab=278M '
                             'domain). Empty (default) = mask all domains.')
    parser.add_argument('--disable_seq_fids', type=str, default='',
                        help='T23 structural disable: comma-separated '
                             'sequence-side-info fids that should be zeroed '
                             'out on every row (train / valid / inference). '
                             'Motivated by community reports '
                             'reporting `seq_c feat47 == top-level item_id` '
                             'is effectively noise, and our P1 random mask '
                             'ablation results '
                             'confirming token-level mask is not enough. '
                             'Typical: "47" to drop only the suspected '
                             'item_id fid in seq_c. Requires uploading a '
                             'custom infer.py that reads this flag from '
                             'train_config.json.')
    parser.add_argument('--enable_eda_dump', action='store_true', default=False,
                        help='EDA · collect Q1 (user/item_id frequency), '
                             'Q2 (per-fid null rate + null-vs-notnull CVR '
                             'gap), Q4 (seq true length), Q5 (time_diff '
                             'quantiles) statistics during training. '
                             'Written to training_summary.json["eda"] at '
                             'end of run. Only active on training path '
                             '(is_training=True); zero effect on inference. '
                             'See EXP-042 for rationale. Overhead: ~2-5% '
                             'per batch due to reservoir sampling.')
    parser.add_argument('--eda_reservoir_size', type=int, default=20000,
                        help='Reservoir sample size per seq domain for '
                             'time_diff / seq_length quantile estimation '
                             '(20k gives p1~p99.9 accuracy ~0.1%).')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--max_train_steps', type=int, default=0,
                        help='Fast-proxy switch: stop after N optimizer steps, '
                             'run a final validation, and save best checkpoint '
                             'if improved (0 = disabled/full training)')
    parser.add_argument('--enable_row_time_cutoff', action='store_true', default=False,
                        help='T8.5 switch: after the Row Group split, filter rows '
                             'by the validation minimum timestamp so train keeps '
                             'timestamp < cutoff and valid keeps timestamp >= cutoff '
                             '(default off)')
    parser.add_argument('--user_train_timestamp_min', type=int, default=0,
                        help='M45 / EXP-056 · weekend-CVR-low filter (5/16 EDA): '
                             'when > 0, drop train rows with timestamp < this value. '
                             'Reference timestamps (verified · UTC midnight): '
                             '1772236800=02-28 / 1772323200=03-01 / 1772409600=03-02 / '
                             '1772496000=03-03 / 1772582400=03-04. '
                             'M45 weekend filter: 1772409600 (drop 02-28/03-01 · keep '
                             '92.8% rows). M48 daypart-only: 1772582400 (drop 4 days · '
                             'keep only 03-04 · 24.6% rows). Per EXP-056 EDA Q6: '
                             '02-28/03-01 weekend CVR 0.024-0.034 · 5x lower than '
                             'weekday CVR 0.10-0.12 · noise removal target. '
                             'Valid set unaffected. Default 0 = no filter.')
    parser.add_argument('--enable_row_valid_gap', action='store_true', default=False,
                        help='C6 / T22-fix switch: row-level double cutoff so '
                             'train / gap (dropped) / dev are separated by '
                             'exact row-wise timestamp quantiles. Fixes the '
                             'RG-level gap failure documented in EXP-026 '
                             '(where each RG spans ~4 days and RG-ordering '
                             'cannot isolate a clean time window). Requires '
                             '--valid_split_strategy time + --valid_gap_ratio '
                             '> 0. Mutually exclusive with '
                             '--enable_row_time_cutoff and --holdout_ratio.')
    parser.add_argument('--fixed_valid_timestamp_min', type=int, default=0,
                        help='EDA-71 clean-window backtest: fixed inclusive '
                             'validation timestamp lower bound. Requires '
                             '--valid_split_strategy time and '
                             '--fixed_valid_timestamp_max. Default 0 = disabled.')
    parser.add_argument('--fixed_valid_timestamp_max', type=int, default=0,
                        help='EDA-71 clean-window backtest: fixed exclusive '
                             'validation timestamp upper bound. Default 0 = disabled.')
    parser.add_argument('--fixed_train_timestamp_max', type=int, default=0,
                        help='EDA-71 clean-window backtest: fixed exclusive '
                             'training timestamp upper bound. Defaults to '
                             '--fixed_valid_timestamp_min when omitted.')
    parser.add_argument('--aux_valid_windows', type=str, default='',
                        help="Optional comma-separated auxiliary validation "
                             "windows, format 'name:start:end' with inclusive "
                             "start and exclusive end timestamps, e.g. "
                             "'m91:1772502591:1772520547'. These windows are "
                             "logged every epoch but never drive best_model, "
                             "top-k, early stopping, or training updates. "
                             "Requires --valid_split_strategy time.")
    parser.add_argument('--enable_aux_ckpt_select', action='store_true',
                        default=False,
                        help='Opt-in checkpoint side selection from an aux '
                             'validation window. This does NOT replace the '
                             'primary-valid best_model/top-k/early-stopping; '
                             'it only saves extra aux/blend best checkpoint '
                             'directories for manual public eval selection.')
    parser.add_argument('--aux_ckpt_name', type=str, default='m91',
                        help='Aux validation window name used by '
                             '--enable_aux_ckpt_select. Must match one entry '
                             'from --aux_valid_windows. Default: m91.')
    parser.add_argument('--aux_ckpt_primary_tolerance', type=float,
                        default=0.0008,
                        help='Primary valid AUC guard for aux-selected '
                             'checkpoints: save aux/blend ckpts only when '
                             'current primary AUC is at least '
                             'best_primary_auc - tolerance. Default 0.0008.')
    parser.add_argument('--aux_ckpt_blend_weight', type=float, default=0.5,
                        help='Aux weight for blend-selected checkpoint score: '
                             '(1-w)*primary_auc + w*aux_auc. Default 0.5.')
    parser.add_argument('--enable_count_features', action='store_true', default=False,
                        help='P0-A switch: append user/item non-zero feature counts '
                             'to user dense features (default off)')
    parser.add_argument('--enable_seq_stats_features', action='store_true', default=False,
                        help='P0-B switch: append per-domain sequence length and '
                             'timestamp-range features to user dense features '
                             '(default off)')
    parser.add_argument('--enable_history_cvr_features', action='store_true', default=False,
                        help='P0-C switch: append leak-safe historical item CVR '
                             'features from a training-only sidecar (default off)')
    parser.add_argument('--history_cvr_cache_path', type=str, default=None,
                        help='Path for the generated P0-C history CVR sidecar '
                             '(defaults to <log_dir>/history_cvr_stats.npz)')
    parser.add_argument('--history_cvr_item_fids', type=str, default='scalar',
                        help="Item int fids for P0-C history CVR features. "
                             "'scalar' means all scalar item_int fids; or pass "
                             "a comma-separated list such as '5,7,12,16'.")
    parser.add_argument('--history_cvr_bin_sec', type=int, default=3600,
                        help='Time-bin width for cumulative P0-C history stats')
    parser.add_argument('--history_cvr_cutoff_sec', type=int, default=86400,
                        help='Only use history rows at least this many seconds '
                             'older than the current row')
    parser.add_argument('--history_cvr_time_mode', type=str,
                        default='timestamp_cutoff',
                        choices=['timestamp_cutoff', 'available'],
                        help='History-CVR time semantics: timestamp_cutoff bins '
                             'rows by event timestamp; available bins positives '
                             'by label_time and negatives by matured availability')
    parser.add_argument('--history_cvr_available_lag_sec', type=int, default=0,
                        help='Additional lookup lag in seconds when '
                             '--history_cvr_time_mode=available')
    parser.add_argument('--history_cvr_prior_strength', type=float, default=20.0,
                        help='Smoothing strength for historical CVR rates')
    parser.add_argument('--enable_mature_negative_weighting',
                        action='store_true', default=False,
                        help='Delayed-feedback switch: downweight negatives '
                             'whose conversion window has not fully elapsed '
                             'by the train observation end timestamp')
    parser.add_argument('--negative_maturity_sec', type=int, default=86400,
                        help='Conversion observation window used by available '
                             'history-CVR and mature-negative weighting')
    parser.add_argument('--immature_negative_weight', type=float, default=0.0,
                        help='Sample weight for immature negatives; 0 drops '
                             'them from the conversion loss')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--time_bucket_boundaries', type=str, default='',
                        help='T30 · per-run override of the 63 time-delta '
                             'bucket edges used for sequence recency encoding. '
                             'Comma-separated positive integers in seconds, '
                             'strictly increasing, length must be exactly 63 '
                             '(= the fixed embedding slot count). Empty '
                             'string (default) keeps the hand-designed ladder '
                             'from 5s to 1yr (see dataset.DEFAULT_BUCKET_'
                             'BOUNDARIES). DECEM Trick 1 · recommended to set '
                             'values at true p1/p5/.../p99 of the observed '
                             'time_diff distribution after EDA dump; demo-'
                             'smoke showed p1≈7.3h so many default buckets '
                             'under 1h are starved in the real data.')
    parser.add_argument('--enable_per_domain_buckets', action='store_true',
                        default=False,
                        help='T38 · per-domain time bucket boundaries. When '
                             'set, fit log-scale 63-edge ladders separately '
                             'for each seq domain (a/b/c/d) by scanning a '
                             'sample of train RGs at startup. EXP-058 §Error '
                             'C reasoning: 4 domains have time_diff scales '
                             'spanning 11x (seq_d p99=56d vs seq_c p99=629d) '
                             '· global ladder either too coarse for short '
                             'domains or too sparse for long ones. NUM_TIME_'
                             'BUCKETS stays 64 so embedding table shape and '
                             'ckpt compat are bit-identical. Mutually '
                             'exclusive with --time_bucket_boundaries (when '
                             'both supplied, --time_bucket_boundaries takes '
                             'priority and per-domain fit is skipped with a '
                             'warning).')
    parser.add_argument('--per_domain_bucket_sample_rgs', type=int, default=8,
                        help='T38 · Number of train Row Groups to scan when '
                             'fitting per-domain bucket boundaries. Default '
                             '8 (~6.4M time_diff samples per domain at '
                             'typical 800k-row RGs). Higher = more robust '
                             'p99 estimate but slower startup. Ignored when '
                             '--enable_per_domain_buckets is off.')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixer mode: full = token mixing + FFN, '
                             'ffn_only = FFN only (no token mixing), '
                             'none = identity passthrough')
    parser.add_argument('--rank_mixer_ffn_mode', type=str, default='shared',
                        choices=['shared', 'per_token'],
                        help='T19 / ADR-007: RankMixer FFN parameterization. '
                             'shared = one Linear pair shared across T tokens '
                             '(legacy, default, back-compat). '
                             'per_token = independent FFN weights per token '
                             '(RankMixer paper original: avoid dominant '
                             'feature flooding long-tail tokens). '
                             'Parameters grow ~T× per block (T=15 @ d=64, '
                             'hidden_mult=4 ≈ +495k dense params per block).')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal', 'auc'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss, '
                             'auc = CombinedAUCLoss = (1-w)*BCE + w*PairwiseBPR '
                             '(T18 / ADR-006, directly optimizes AUC)')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--auc_loss_weight', type=float, default=0.5,
                        help='CombinedAUCLoss mixing weight w in [0, 1]: '
                             'total = (1-w)*BCE + w*PairwiseBPR. '
                             '0.0 = pure BCE, 1.0 = pure BPR. '
                             'Effective only when --loss_type=auc')
    parser.add_argument('--auc_margin', type=float, default=0.0,
                        help='BPR margin subtracted from (s_pos - s_neg) '
                             'before sigmoid. Larger margin pushes positives '
                             'harder above negatives. UniRec default 0.5. '
                             'Effective only when --loss_type=auc')
    parser.add_argument('--auc_max_pairs', type=int, default=256,
                        help='Cap on positives / negatives sampled per step '
                             'for BPR; memory is O(auc_max_pairs^2). '
                             'Effective only when --loss_type=auc')
    parser.add_argument('--multi_task_loss', action='store_true', default=False,
                        help='P1 switch: train action_num=2 heads as '
                             'click(label_type>=1) + conversion(label_type==2); '
                             'inference/eval use the conversion head')
    parser.add_argument('--click_loss_weight', type=float, default=1.0,
                        help='Loss weight for the click head when '
                             '--multi_task_loss is enabled')
    parser.add_argument('--conversion_loss_weight', type=float, default=2.0,
                        help='Loss weight for the conversion head when '
                             '--multi_task_loss is enabled')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--dense_log1p_fids', type=str, default='62,63,64,65,66',
                        help='Comma-separated list of user_dense fids whose raw values '
                             'are large-magnitude counts/frequencies and should be '
                             'compressed with log1p before the dense projection layer. '
                             'EDA confirmed fids 62-66 store raw counts (10^3-10^5) '
                             'while other dense columns (fid 61/87) are pre-normalised '
                             'embeddings. Pass "" to disable all scaling.')
    parser.add_argument('--enable_time_of_day_features', action='store_true', default=False,
                        help='Append 5 synthetic dense features derived from the impression '
                             'timestamp: hour_sin, hour_cos (hour-of-day circular encoding), '
                             'dow_sin, dow_cos (day-of-week circular encoding), is_weekend. '
                             'Captures diurnal and weekly purchase-intent periodicity absent '
                             'from existing sequence features. (T12)')
    parser.add_argument('--enable_beijing_time_features', action='store_true', default=False,
                        help='Append 10 synthetic dense features derived from Beijing local '
                             'time (UTC+8): first/second harmonic hour, day-of-week, '
                             'weekend/workday, test-window 09-14, and early-morning 06-08. '
                             'Can replace or append --enable_time_of_day_features.')
    parser.add_argument('--enable_beijing_time_v2_features', action='store_true', default=False,
                        help='Append 14 Beijing-local distribution features: distance to '
                             'the observed test date, festival/post-festival flags, distance '
                             'to the 09-14 test window, coarse daypart one-hot features, and '
                             'a target-day test-window indicator. Feature-only; no sample '
                             'reweighting.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    parser.add_argument('--use_onetrans', action='store_true', default=False,
                        help='Replace HyFormer query-decoding blocks with OneTrans '
                             'Mixed-Parameterization unified attention. When enabled, '
                             'seq_encoder_type is used for the per-domain lightweight '
                             'SwiGLU pre-encoder; the num_queries and rank_mixer_mode '
                             'flags are ignored.')
    parser.add_argument('--onetrans_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept per domain before '
                             'the unified OneTrans attention (default 50)')
    parser.add_argument('--seq_hash_vocab', type=int, default=0,
                        help='T13: when > 0, sequence features with vocab > '
                             'emb_skip_threshold use a hash embedding of this size '
                             'instead of a zero vector. Values are mapped via '
                             'value %% seq_hash_vocab before embedding lookup. '
                             '0 = disabled (default).')
    parser.add_argument('--item_conditioned_query', action='store_true', default=False,
                        help='Enable item-conditioned query generation (DIN-style). '
                             'Each sequence domain learns a gate that blends the standard '
                             'global-info query with a target-item-aware query branch, '
                             'letting the model focus user history on the target ad.')

    parser.add_argument('--enable_q_init_item', action='store_true', default=False,
                        help='T32 / DECEM Trick 6 · Q_init=MLP(item_emb)+NS_residual. '
                             'Replaces query base path from global_info FFN to '
                             'item_repr FFN with NS context residual projection. '
                             'Compatible with --item_conditioned_query (ICQ mix gate '
                             'stays in outer layer). First launch recommended single-variable. '
                             'Predicted +0.001~+0.003 LB (community-intel.md DECEM 2026-05-08).')

    parser.add_argument('--multi_emb_k', type=int, default=1, choices=[1, 2],
                        help='T34 / RankUp Multi-Embedding K · ADR-012. '
                             'When K=1 (default), bit-identical to baseline single '
                             'embedding table per fid. When K=2, each non-skipped fid '
                             'has 2 independent embedding tables; outputs concatenated '
                             'and projected via shared Linear(2*emb_dim, emb_dim). '
                             'Adds ~+300M sparse params (~+1.2GB). May need '
                             '--batch_size 96 for VRAM. RankUp ablation +0.13~0.21% '
                             'AUC online (Tencent self · arXiv:2604.17878).')

    parser.add_argument('--target_item_seq_injection', type=str, default='off',
                        choices=['off', 'additive'],
                        help='ADR-004: inject target-item representation into every '
                             'sequence token BEFORE the seq encoder, making the encoder '
                             'target-aware (orthogonal to item_conditioned_query, which '
                             'is query-side). "additive" adds alpha_i * Linear_i(item_repr) '
                             'to each token of sequence i. With alpha_init=0 (default), '
                             'training starts equivalent to baseline.')
    parser.add_argument('--target_inject_alpha_init', type=float, default=0.0,
                        help='Initial value of the per-sequence scalar alpha_i used to '
                             'gate target-item injection strength. 0.0 means the injector '
                             'starts as an identity; optimizer decides how much each '
                             'domain attends to target over training.')

    parser.add_argument('--enable_dense_bypass', action='store_true', default=False,
                        help='Dense bypass: project dense features directly to the '
                             'classifier input instead of (or in addition to) the NS '
                             'token path. The bypass vector is concatenated to the '
                             'output_proj output, making the classifier input 2*d_model. '
                             'This avoids dense signal dilution inside HyFormer blocks.')
    parser.add_argument('--enable_ue_split', action='store_true', default=False,
                        help='T37 / Path B Step 1 (ADR-014 修订版 5/16): UE 分离 = '
                             '把 user_dense_feats_61 (256-dim 预归一化 production user '
                             'embedding) 单独成 NS token (复用 user_dense slot). 其他 '
                             'dense feats (count fid_62~66 + fid_87/89/90/91 共 732-dim) '
                             '走 dense_bypass 路径 (T25 已 work · 自动启用). 不与 '
                             '--enable_global_token 同启 (raise ValueError). 与 '
                             '--enable_dense_bypass 自动 stack。基于社区讨论 '
                             '第 (1) 条情报 + EXP-011 实证 fid_61 是预归一化 UE. '
                             'ue_offset/ue_dim 由 dataset 动态计算 (从 user_dense_schema '
                             '取 fid_61 在 user_dense_feats tensor 中的 (offset, length)).')
    parser.add_argument('--ue_split_fid', type=int, default=61,
                        help='T37: raw schema fid for UE (default 61 = 预归一化 user '
                             'embedding · 256-dim). Resolved to (ue_offset, ue_dim) via '
                             'pcvr_dataset.user_dense_schema.entries.')
    parser.add_argument('--ue_split_fids', type=str, default='',
                        help='T40: comma-separated user_dense raw fids to concatenate '
                             'into the UE token. Empty keeps legacy --ue_split_fid. '
                             'External intel 5/20 suggests "61,87".')
    parser.add_argument('--ue_split_separate_tokens', action='store_true',
                        default=False,
                        help='T42: with --enable_ue_split and multiple '
                             '--ue_split_fids, project each selected dense '
                             'fid into its own NS token instead of '
                             'concatenating them into one UE token. Keep the '
                             'RankMixer token budget valid manually, e.g. '
                             '--user_ns_tokens 4 with fids 61,87 keeps '
                             'T=16 at d_model=64.')
    parser.add_argument('--enable_ue_int_bilinear', action='store_true',
                        default=False,
                        help='T39 / M55 · UE × user_int Bilinear gating. Implements '
                             'Seed 主楼 5/5 第 (1) 条后半句 "其他跟 int pair 加权": '
                             'gate the "other" dense feats (everything except UE '
                             'fid_61 slice when --enable_ue_split) by a Linear '
                             'projection of the pooled user_int representation '
                             'BEFORE the dense_bypass projection. Mechanism '
                             'differs from DCN-V2 (M23 LB −0.011 dead): single-'
                             'layer multiplicative gating (not recursive cross), '
                             'OOV-friendly because it reweights existing dense '
                             'features by user_int signal rather than learning '
                             'train-internal user × item co-occurrence. Requires '
                             '--enable_dense_bypass=True or --enable_ue_split=True. '
                             'Init weights to zero so initial gating ≈ 1 (bit-'
                             'identical to non-bilinear path at step 0).')
    parser.add_argument('--ue_int_bilinear_alpha_init', type=float, default=0.5,
                        help='T39 / M55 · Initial value for the learnable scalar '
                             'alpha controlling gating range: gating ∈ [1-alpha, '
                             '1+alpha] (via tanh). Default 0.5 = mild reweighting '
                             '[0.5, 1.5]. Larger = more aggressive gating, more '
                             'risk of training instability.')
    parser.add_argument('--enable_ue_item_interaction', action='store_true',
                        default=False,
                        help='T40 · Explicit UE x target-item interaction bypass. '
                             'Requires --enable_ue_split. Projects '
                             '[UE, item_repr, UE*item_repr] to d_model and '
                             'concatenates it to the classifier input.')
    parser.add_argument('--ue_item_interaction_alpha_init', type=float, default=1.0,
                        help='T40 · Initial learnable scalar for the UE x item '
                             'interaction bypass. Default 1.0.')
    parser.add_argument('--enable_din_interest', action='store_true', default=False,
                        help='T25 / G1: DIN-style target-aware interest pooling. '
                             'Per-domain cross-attention uses the mean of item_ns '
                             'tokens as query against each seq, producing S interest '
                             'vectors which are merged into a single (B, D) composite '
                             'interest. This composite is concatenated to the backbone '
                             'output before the classifier (bypass mode, identical to '
                             '--enable_dense_bypass integration pattern). Orthogonal to '
                             '--target_item_seq_injection (T16 additive, which modifies '
                             'seq encoder inputs). Default off (back-compat).')
    parser.add_argument('--din_interest_source', type=str, default='raw',
                        choices=['raw', 'encoded'],
                        help='DIN 2.0: sequence representation used by '
                             '--enable_din_interest. raw preserves the original '
                             'T25 bypass path. encoded runs classifier-time DIN '
                             'over the final HyFormer-encoded sequence states, '
                             'keeping the signal late/bypass-only while giving '
                             'DIN a richer sequence representation.')
    parser.add_argument('--din_interest_merge', type=str, default='compact',
                        choices=['compact', 'per_domain'],
                        help='DIN 2.0: how per-domain DIN interests enter the '
                             'classifier. compact preserves T25 by merging all '
                             'domains to one d_model vector. per_domain keeps '
                             'the S domain interest vectors concatenated '
                             '(S*d_model) before the classifier.')
    parser.add_argument('--enable_tin_interest', action='store_true',
                        default=False,
                        help='T41 · TIN-lite target-aware temporal interest. '
                             'Requires --enable_din_interest and raw DIN '
                             'source. Replaces the DIN classifier-time '
                             'extractor with a shape-compatible variant that '
                             'modulates sequence tokens by target-item and '
                             'time-bucket embeddings before attention.')
    parser.add_argument('--tin_time_alpha_init', type=float, default=1.0,
                        help='T41 · Initial scalar for the TIN-lite temporal '
                             'modulation. The final temporal MLP layer is '
                             'zero-initialized, so the extractor starts close '
                             'to ordinary DIN regardless of this value.')
    parser.add_argument('--enable_din_integrated', action='store_true', default=False,
                        help='T28: DIN target-aware interest injected INSIDE each '
                             'HyFormer block (not as a classifier-time bypass like '
                             '--enable_din_interest). Each block computes its own '
                             'composite interest vector from its just-encoded '
                             'sequences and adds it (per-block learnable scalar '
                             'alpha, init 0.1) to every decoded_q_i before token '
                             'fusion into the RankMixer. Orthogonal to --enable_din_interest '
                             '(can be combined, but first ablation should keep T28 '
                             'on and T25 bypass off). Adds ~num_blocks * 104k dense '
                             'params (~208k at num_blocks=2 / d_model=64). '
                             'Zero effect on OneTrans path.')
    parser.add_argument('--din_integrated_alpha_init', type=float, default=0.1,
                        help='T28: initial value of the per-block learnable scalar '
                             'alpha that scales the DIN interest injection. Default '
                             '0.1 (safe partial injection). Set to 0.0 for identity '
                             'init (requires training to pick up signal).')
    parser.add_argument('--enable_nlir_gating', action='store_true', default=False,
                        help='T33 / ADR-011: enable TokenFormer-style NLIR gating '
                             'on cross-attention output inside every HyFormer '
                             'block. The pure cross-attention delta A is gated '
                             'by sigmoid(X @ W_g) where X is the block-input '
                             'query, producing: decoded = X + sigmoid(X @ W_g) ⊙ A '
                             '(TokenFormer §4.4 Eq. 16-18). Adds ~D² params '
                             'per block (~4k at d_model=64, ~8k total for 2 '
                             'blocks). Shared across all S sequence domains '
                             'within each block (not per-seq). Orthogonal to '
                             '--enable_din_interest (T25), --multi_emb_k (T34), '
                             'and dataset-side T30/T31. Default off (backward '
                             'compatible, bit-identical when disabled).')
    parser.add_argument('--enable_fafe', action='store_true', default=False,
                        help='T25 / H2: Feature-aware Feature Embedding. When set, '
                             'each fid in the RankMixer NS tokenizer passes through '
                             'a per-fid LayerNorm+Linear(emb_dim, emb_dim) transform '
                             'before the shared chunked projection. Mirrors InterFormer '
                             '(CIKM 2025) + Seed 主楼方案 2. Adds ~130k dense params '
                             'for 46 user fids @ emb_dim=64. Zero effect on group '
                             'tokenizer path (ns_tokenizer_type=group).')

    parser.add_argument('--enable_dcn_cross', action='store_true', default=False,
                        help='T34 / EXP-049: enable DCN-V2 cross-feature bypass. '
                             'Builds an explicit outer-product cross signal from '
                             'selected user_int x item_int scalar fids (top '
                             'correlations from EDA-correlation 2026-05-13) and '
                             'concatenates a (B, D) vector to the classifier input '
                             '(same bypass pattern as T25 dense / DIN). Orthogonal '
                             'to all attention paths. Adds ~42k dense params at '
                             'default config. Embeddings shared with NS tokenizer.')
    parser.add_argument('--dcn_cross_user_fids', type=str,
                        default='1,49',
                        help='T34: comma-separated raw schema fid numbers '
                             '(the X in user_int_feats_X) for the user side of '
                             'the DCN cross input. Default "1,49" follows '
                             'EDA-correlation top user x item pairs '
                             '(user_1 / user_49). Each fid MUST be scalar '
                             '(dim=1) — multi-hot fids raise ValueError to '
                             'avoid silent mis-pooling.')
    parser.add_argument('--dcn_cross_item_fids', type=str,
                        default='5,6,7,10,12',
                        help='T34: comma-separated raw schema fid numbers '
                             '(the X in item_int_feats_X) for the item side of '
                             'the DCN cross input. Default "5,6,7,10,12" '
                             'follows EDA-correlation top fid->label MI ranking '
                             '(item_5 / item_10 / item_6 / item_7 / item_12). '
                             'Each fid MUST be scalar (dim=1).')
    parser.add_argument('--dcn_cross_layers', type=int, default=2,
                        help='T34: number of stacked DCN-V2 cross layers. '
                             'Default 2 (≈ 4-th order interactions). Each '
                             'layer adds Linear(d_model, d_model) — at '
                             'd_model=64 that is ~4k dense params per layer. '
                             'Range [1, 4] is sensible; larger may overfit '
                             'on this dataset size (1.9M rows).')

    parser.add_argument('--enable_global_token', action='store_true', default=False,
                        help='T36 / ADR-013 (Plan B): replace the user_dense NS '
                             'token slot with an aggregated Global Token '
                             '(LN(Linear(concat[user_dense, mean(user_ns), '
                             'mean(item_ns), mean(seq_*)]))). Preserves '
                             'T-divisibility (num_ns unchanged) per RankUp §3.4 '
                             'aggregation interpretation. Requires user_dense_dim>0. '
                             'Adds ~25k dense params at d_model=64. Default off = '
                             'bit-identical baseline.')

    parser.add_argument('--enable_torch_compile', action='store_true', default=False,
                        help='T27: wrap the model with torch.compile on the train '
                             'path only (evaluate/predict use the un-compiled raw '
                             'model, mirroring DECEM Rank-1 post 2026-05-08). '
                             'Uses mode="default" and suppresses dynamo errors to '
                             'tolerate recompiles triggered by sparse embeddings / '
                             'variable batch shape. No-op on CPU device. The saved '
                             'model.pt contains raw-model state_dict, so '
                             'checkpoints remain compatible with platform default '
                             'infer.py.')
    parser.add_argument('--enable_amp', action='store_true', default=False,
                        help='T27: enable torch.autocast for the forward+loss of '
                             'the training step. bfloat16 (default) avoids '
                             'loss-scaling overhead and is numerically stable for '
                             'BCE/Focal losses; fp16 goes through GradScaler. '
                             'Reduces memory traffic ~40% on modern GPUs. '
                             'Inference path is never autocast (stays fp32).')
    parser.add_argument('--amp_dtype', type=str, default='bfloat16',
                        choices=['bfloat16', 'float16'],
                        help='T27: AMP precision. bfloat16 recommended for '
                             'Ampere+ GPUs (A100/V100 partial support); float16 '
                             'requires GradScaler but has wider hardware support. '
                             'Only consulted when --enable_amp is set.')
    parser.add_argument('--keep_topk_ckpt', type=int, default=1,
                        help='Keep top-K checkpoints (by val AUC DESC) in '
                             'addition to the global-best *.best_model/ dir. '
                             'Default 1 = current behavior (only best_model). '
                             'K>1 persists runners-up in *.topk_{rank}/ dirs '
                             'that are renamed on every validation to mirror '
                             'the live ranking. Motivation (2026-05-09, 冲 0.85 '
                             '战略): icq-full E6 LB < icq-half E4 LB showed '
                             'that "best AUC" != "best LB". Keeping top-K '
                             'lets evaluation quota test multiple ckpts from '
                             'a single training run (4x effective evaluation '
                             'utilization when K=4).')
    parser.add_argument('--enable_weight_soup', action='store_true',
                        default=False,
                        help='Post-train side checkpoint: average eligible '
                             'same-run best/top-k model weights into one '
                             '*.weight_soup checkpoint. This does not affect '
                             'training, best_model, top-k, early stopping, or '
                             'aux checkpoint selection. Requires multiple '
                             'saved checkpoints, typically --keep_topk_ckpt 4.')
    parser.add_argument('--weight_soup_topk', type=int, default=4,
                        help='Maximum number of eligible best/top-k '
                             'checkpoints to average when '
                             '--enable_weight_soup is set. Default 4.')
    parser.add_argument('--weight_soup_primary_tolerance', type=float,
                        default=0.0010,
                        help='Only average checkpoints with primary-valid '
                             'AUC at least best_auc - tolerance. Default '
                             '0.0010, intended to include late near-best '
                             'epochs while excluding underfit early models.')
    parser.add_argument('--weight_soup_min_members', type=int, default=2,
                        help='Minimum number of eligible checkpoints required '
                             'to materialize a weight-soup checkpoint. '
                             'Default 2; otherwise the run logs a skip.')
    parser.add_argument('--weight_soup_include_sparse', action='store_true',
                        default=False,
                        help='Also average sparse embedding tensors in the '
                             'weight soup. Default off because this project '
                             'cold-restarts high-cardinality embeddings after '
                             'epoch validation; dense-only soup preserves '
                             'embedding weights from the best checkpoint.')


    parser.add_argument('--temporal_weight_alpha', type=float, default=0.0,
                        help='Recency reweighting: give samples closer to the test period '
                             'higher training weight via weight=exp(alpha*norm_ts). '
                             'norm_ts is linearly scaled to [0,1] over the training split. '
                             '0 = disabled (default); 2~4 = moderate upweighting of '
                             'recent samples; useful for reducing train/test temporal gap.')
    parser.add_argument('--hour_weight_min', type=int, default=-1,
                        help='5/17 EXP-069 · Hour-aware sample reweight (Beijing time = '
                             'UTC + 8h). When >= 0, train rows with beijing_hour ∈ '
                             '[hour_weight_min, hour_weight_max] get sample_weight × '
                             '--hour_weight_multiplier. Test daypart = 北京 09~14 → set '
                             '--hour_weight_min 9 --hour_weight_max 14 --hour_weight_multiplier 5.0. '
                             'Composes multiplicatively with --temporal_weight_alpha (M41) '
                             'so combined "alpha=2.5 + hour_weight×5" upweights 03-04 ' 
                             '09~14 daypart vs the rest of training data. Default -1 = disabled.')
    parser.add_argument('--hour_weight_max', type=int, default=-1,
                        help='5/17 EXP-069 · Upper bound (inclusive) for --hour_weight_min. '
                             'Default -1 = disabled.')
    parser.add_argument('--hour_weight_multiplier', type=float, default=-1.0,
                        help='5/17 EXP-069 · Multiplier (e.g. 5.0) applied to in-window '
                             'samples. Default -1.0 = disabled.')
    parser.add_argument('--target_day_hour_weight_date', type=str, default='',
                        help='5/18 M83 · Exact Beijing date sample reweight '
                             '(YYYY-MM-DD). When set with hour min/max and multiplier, '
                             'only rows on this Beijing date and hour window are '
                             'upweighted, using a global train-set normalizer.')
    parser.add_argument('--target_day_hour_weight_min', type=int, default=-1,
                        help='5/18 M83 · Inclusive Beijing hour lower bound for '
                             '--target_day_hour_weight_date. Default -1 = disabled.')
    parser.add_argument('--target_day_hour_weight_max', type=int, default=-1,
                        help='5/18 M83 · Inclusive Beijing hour upper bound for '
                             '--target_day_hour_weight_date. Default -1 = disabled.')
    parser.add_argument('--target_day_hour_weight_multiplier', type=float, default=-1.0,
                        help='5/18 M83 · Multiplier for exact date+hour rows, '
                             'normalized by global expected mean weight.')
    parser.add_argument('--enable_hour_only_features', action='store_true', default=False,
                        help='Append hour-of-day sin/cos (2 dims) to user_dense. '
                             'Captures diurnal periodicity WITHOUT day-of-week, which '
                             'avoids overfitting to the weekday distribution of training '
                             'data (train is dominated by a single weekday ~63%%). '
                             'Use this instead of --enable_time_of_day_features when '
                             'DOW features hurt leaderboard generalization.')
    parser.add_argument('--enable_is_missing', action='store_true', default=False,
                        help='T31 · DECEM Trick 3 · surface missing values as '
                             'an independent signal rather than collapsing '
                             'null / <=0 into the embedding padding slot (0). '
                             'Two redundant paths: (1) each selected fid gets '
                             'one extra vocab slot (vs -> vs+1) where missing '
                             'rows are routed; (2) a per-sample 0/1 bitmap is '
                             'appended to user_dense. Zero effect when off. '
                             'Backward-compat: schema + model picks up the '
                             'new dims automatically from the dataset.')
    parser.add_argument('--is_missing_user_int_fids', type=str, default='',
                        help='T31 · comma-separated user_int fids to treat as '
                             '"missing-aware". Empty (default) selects ALL '
                             'scalar user_int fids. DECEM-listed high-null '
                             'fids: 86,94,99,100,101,102,103,104,108,109.')
    parser.add_argument('--is_missing_item_int_fids', type=str, default='',
                        help='T31 · comma-separated item_int fids to treat as '
                             '"missing-aware". Empty (default) means NONE; '
                             'item_int is opt-in because DECEM only listed '
                             'user_int fids as high-null.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')

    args = parser.parse_args()

    for name in (
        'limit_train_rgs',
        'limit_valid_rgs',
        'max_train_steps',
        'history_cvr_cutoff_sec',
        'history_cvr_available_lag_sec',
        'negative_maturity_sec',
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be >= 0")
    if args.history_cvr_bin_sec <= 0:
        parser.error("--history_cvr_bin_sec must be > 0")
    if args.history_cvr_prior_strength < 0:
        parser.error("--history_cvr_prior_strength must be >= 0")
    if args.immature_negative_weight < 0:
        parser.error("--immature_negative_weight must be >= 0")
    if args.weight_soup_topk <= 0:
        parser.error("--weight_soup_topk must be > 0")
    if args.weight_soup_primary_tolerance < 0:
        parser.error("--weight_soup_primary_tolerance must be >= 0")
    if args.weight_soup_min_members < 2:
        parser.error("--weight_soup_min_members must be >= 2")
    if args.enable_tin_interest and not args.enable_din_interest:
        parser.error("--enable_tin_interest requires --enable_din_interest")
    if args.enable_tin_interest and args.din_interest_source != 'raw':
        parser.error("--enable_tin_interest requires --din_interest_source raw")
    if args.enable_tin_interest and not args.use_time_buckets:
        parser.error("--enable_tin_interest requires --use_time_buckets")

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    # TRAIN_LOG_PATH is project-internal (not part of the platform's
    # documented env var set); fall back to a writable temp dir so the
    # script does not crash when neither env nor --log_dir is provided.
    args.log_dir = (os.environ.get('TRAIN_LOG_PATH', args.log_dir)
                    or '/tmp/taac_train_log')
    args.tf_events_dir = (
        os.environ.get('TRAIN_TF_EVENTS_PATH')
        or os.path.join(args.log_dir, 'tb')
    )
    if args.enable_history_cvr_features and not args.history_cvr_cache_path:
        args.history_cvr_cache_path = os.path.join(
            args.log_dir, 'history_cvr_stats.npz')
    if args.multi_task_loss:
        args.action_num = 2

    return args


def main() -> None:
    args = parse_args()

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")
    logging.info(f"Resolved schema_path: {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    aux_valid_windows = parse_aux_valid_windows(args.aux_valid_windows)
    if aux_valid_windows:
        logging.info(f"Aux validation windows: {aux_valid_windows}")

    data_result = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
        valid_split_strategy=args.valid_split_strategy,
        limit_train_rgs=args.limit_train_rgs,
        limit_valid_rgs=args.limit_valid_rgs,
        enable_row_time_cutoff=args.enable_row_time_cutoff,
        enable_row_valid_gap=args.enable_row_valid_gap,
        fixed_valid_timestamp_min=(
            args.fixed_valid_timestamp_min
            if args.fixed_valid_timestamp_min > 0 else None
        ),
        fixed_valid_timestamp_max=(
            args.fixed_valid_timestamp_max
            if args.fixed_valid_timestamp_max > 0 else None
        ),
        fixed_train_timestamp_max=(
            args.fixed_train_timestamp_max
            if args.fixed_train_timestamp_max > 0 else None
        ),
        aux_valid_windows=aux_valid_windows,
        enable_count_features=args.enable_count_features,
        enable_seq_stats_features=args.enable_seq_stats_features,
        enable_history_cvr_features=args.enable_history_cvr_features,
        history_cvr_cache_path=args.history_cvr_cache_path,
        history_cvr_item_fids=args.history_cvr_item_fids,
        history_cvr_bin_sec=args.history_cvr_bin_sec,
        history_cvr_cutoff_sec=args.history_cvr_cutoff_sec,
        history_cvr_time_mode=args.history_cvr_time_mode,
        history_cvr_available_lag_sec=args.history_cvr_available_lag_sec,
        history_cvr_prior_strength=args.history_cvr_prior_strength,
        enable_mature_negative_weighting=args.enable_mature_negative_weighting,
        negative_maturity_sec=args.negative_maturity_sec,
        immature_negative_weight=args.immature_negative_weight,
        dense_log1p_fids=frozenset(
            int(x) for x in args.dense_log1p_fids.split(',') if x.strip()
        ),
        enable_time_of_day_features=args.enable_time_of_day_features,
        enable_beijing_time_features=args.enable_beijing_time_features,
        enable_beijing_time_v2_features=args.enable_beijing_time_v2_features,
        temporal_weight_alpha=args.temporal_weight_alpha,
        # 5/17 EXP-069 · hour-aware reweight (None disables · only train)
        hour_weight_min=(args.hour_weight_min
                         if args.hour_weight_min >= 0 else None),
        hour_weight_max=(args.hour_weight_max
                         if args.hour_weight_max >= 0 else None),
        hour_weight_multiplier=(args.hour_weight_multiplier
                                if args.hour_weight_multiplier > 0 else None),
        target_day_hour_weight_date=args.target_day_hour_weight_date,
        target_day_hour_weight_min=(args.target_day_hour_weight_min
                                    if args.target_day_hour_weight_min >= 0 else None),
        target_day_hour_weight_max=(args.target_day_hour_weight_max
                                    if args.target_day_hour_weight_max >= 0 else None),
        target_day_hour_weight_multiplier=(
            args.target_day_hour_weight_multiplier
            if args.target_day_hour_weight_multiplier > 0 else None
        ),
        enable_hour_only_features=args.enable_hour_only_features,
        # M45 / EXP-056 · weekend-CVR-low filter (0 = disabled · backward compat)
        user_train_timestamp_min=(args.user_train_timestamp_min
                                  if args.user_train_timestamp_min > 0 else None),
        holdout_ratio=args.holdout_ratio,
        valid_gap_ratio=args.valid_gap_ratio,
        id_mask_prob=args.id_mask_prob,
        id_mask_seq_domains=(
            [s.strip() for s in args.id_mask_seq_domains.split(',') if s.strip()]
            or None
        ),
        disable_seq_fids=(
            [int(s.strip()) for s in args.disable_seq_fids.split(',') if s.strip()]
            or None
        ),
        enable_eda_dump=args.enable_eda_dump,
        eda_reservoir_size=args.eda_reservoir_size,
        # T38 silent skip bug fix (5/16 EXP-061 M52/M54 事故 · 见 EXP-064)
        # 当 --enable_per_domain_buckets=True 时,必须传 None 让 dataset.py
        # line 3430 的 fit 分支触发 (检测条件 = `bucket_boundaries is None`)。
        # 否则 parse_bucket_boundaries('') 返回 default ladder copy (非 None),
        # 进入 line 3445 elif 触发 silent skip warning · per-domain fit 被
        # 跳过 · 实际跑全局 default ladder · M52/M54 因此实际 = M49/M22 rerun。
        # 历史损失: 10 GPU-hour + per-domain bucket 路径 0 数据点。
        # 互斥语义: 当用户**同时**显式给 --time_bucket_boundaries 非空字符串
        # + --enable_per_domain_buckets=True 时,supplied 优先 (现有 elif 警告
        # 路径) · 这是 train.py CLI help line 318-321 文档化的语义。
        bucket_boundaries=(
            None
            if args.enable_per_domain_buckets
            and not str(args.time_bucket_boundaries).strip()
            else parse_bucket_boundaries(args.time_bucket_boundaries)
        ),
        enable_per_domain_buckets=args.enable_per_domain_buckets,
        per_domain_bucket_sample_rgs=args.per_domain_bucket_sample_rgs,
        enable_is_missing=args.enable_is_missing,
        is_missing_user_int_fids=parse_int_csv(args.is_missing_user_int_fids),
        is_missing_item_int_fids=parse_int_csv(args.is_missing_item_int_fids),
    )
    if len(data_result) == 5:
        (
            train_loader,
            valid_loader,
            holdout_loader,
            aux_valid_loaders,
            pcvr_dataset,
        ) = data_result
    elif len(data_result) == 4:
        train_loader, valid_loader, holdout_loader, pcvr_dataset = data_result
        aux_valid_loaders = {}
        if aux_valid_windows:
            logging.warning(
                "get_pcvr_data returned 4 values; aux_valid_windows will not "
                "be logged. Upload the matching dataset.py with aux validation "
                "support for diagnostic metrics.")
    else:
        raise ValueError(
            f"get_pcvr_data returned {len(data_result)} values; expected 4 or 5")

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    # ---- T34: Map raw schema fid numbers (the X in user_int_feats_X) to
    # list indices into user_int_feature_specs / item_int_feature_specs.
    # This decoupling lets the model API stay typed (indices) while
    # users specify the more intuitive raw fid numbers on CLI / run.sh.
    # ----
    if args.enable_dcn_cross:
        user_fid_to_idx = {
            fid: i for i, (fid, _, _)
            in enumerate(pcvr_dataset.user_int_schema.entries)
        }
        item_fid_to_idx = {
            fid: i for i, (fid, _, _)
            in enumerate(pcvr_dataset.item_int_schema.entries)
        }
        try:
            user_dcn_raw = parse_int_csv(args.dcn_cross_user_fids)
            item_dcn_raw = parse_int_csv(args.dcn_cross_item_fids)
        except ValueError as e:
            raise ValueError(
                f"--dcn_cross_user_fids / --dcn_cross_item_fids parse "
                f"error: {e}") from e

        unknown_u = [f for f in user_dcn_raw if f not in user_fid_to_idx]
        unknown_i = [f for f in item_dcn_raw if f not in item_fid_to_idx]
        if unknown_u:
            raise ValueError(
                f"--dcn_cross_user_fids contains unknown fids "
                f"{unknown_u} (available: "
                f"{sorted(user_fid_to_idx.keys())})")
        if unknown_i:
            raise ValueError(
                f"--dcn_cross_item_fids contains unknown fids "
                f"{unknown_i} (available: "
                f"{sorted(item_fid_to_idx.keys())})")

        dcn_user_fid_indices = [user_fid_to_idx[f] for f in user_dcn_raw]
        dcn_item_fid_indices = [item_fid_to_idx[f] for f in item_dcn_raw]
        logging.info(
            f"T34 DCN cross enabled: user fids={user_dcn_raw} "
            f"(indices {dcn_user_fid_indices}), item fids="
            f"{item_dcn_raw} (indices {dcn_item_fid_indices}), "
            f"layers={args.dcn_cross_layers}")
    else:
        dcn_user_fid_indices = None
        dcn_item_fid_indices = None

    # T37 / ADR-014 · UE split fid resolution. Legacy path resolves one
    # fid from --ue_split_fid. T40 extends this to --ue_split_fids (e.g.
    # "61,87") and concatenates all selected dense slices into one UE token.
    ue_offset_resolved = 0
    ue_dim_resolved = 256  # default; updated by schema resolve below
    ue_slices_resolved = None
    ue_split_fids_resolved = None
    if args.enable_ue_split:
        target_fids = parse_int_csv(args.ue_split_fids) or [int(args.ue_split_fid)]
        entry_by_fid = {
            entry[0]: entry for entry in pcvr_dataset.user_dense_schema.entries
        }
        unknown_ue_fids = [fid for fid in target_fids if fid not in entry_by_fid]
        if unknown_ue_fids:
            raise ValueError(
                f"--enable_ue_split=True but UE fid(s) {unknown_ue_fids} "
                f"not found in user_dense_schema (available fids: "
                f"{[e[0] for e in pcvr_dataset.user_dense_schema.entries]})")
        ue_split_fids_resolved = list(target_fids)
        ue_slices_resolved = [
            (int(entry_by_fid[fid][1]), int(entry_by_fid[fid][2]))
            for fid in target_fids
        ]
        ue_offset_resolved = int(ue_slices_resolved[0][0])
        ue_dim_resolved = int(sum(dim for _, dim in ue_slices_resolved))
        logging.info(
            f"T37/T40 UE split: fids={target_fids} resolved to slices="
            f"{ue_slices_resolved}, total_dim={ue_dim_resolved}")

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "rank_mixer_ffn_mode": args.rank_mixer_ffn_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "use_onetrans": args.use_onetrans,
        "onetrans_top_k": args.onetrans_top_k,
        "seq_hash_vocab": args.seq_hash_vocab,
        "item_conditioned_query": args.item_conditioned_query,
        "q_init_item": args.enable_q_init_item,
        "multi_emb_k": args.multi_emb_k,
        "target_item_seq_injection": args.target_item_seq_injection,
        "target_inject_alpha_init": args.target_inject_alpha_init,
        "enable_dense_bypass": args.enable_dense_bypass,
        "enable_din_interest": args.enable_din_interest,
        "din_interest_source": args.din_interest_source,
        "din_interest_merge": args.din_interest_merge,
        "enable_tin_interest": args.enable_tin_interest,
        "tin_time_alpha_init": args.tin_time_alpha_init,
        "enable_din_integrated": args.enable_din_integrated,
        "din_integrated_alpha_init": args.din_integrated_alpha_init,
        "enable_nlir_gating": args.enable_nlir_gating,
        "enable_fafe": args.enable_fafe,
        "enable_dcn_cross": args.enable_dcn_cross,
        "dcn_cross_user_fids": dcn_user_fid_indices,
        "dcn_cross_item_fids": dcn_item_fid_indices,
        "dcn_cross_layers": args.dcn_cross_layers,
        "enable_global_token": args.enable_global_token,
        "enable_ue_split": args.enable_ue_split,
        "ue_offset": ue_offset_resolved,
        "ue_dim": ue_dim_resolved,
        "ue_slices": ue_slices_resolved,
        "ue_split_separate_tokens": args.ue_split_separate_tokens,
        "enable_ue_int_bilinear": args.enable_ue_int_bilinear,
        "ue_int_bilinear_alpha_init": args.ue_int_bilinear_alpha_init,
        "enable_ue_item_interaction": args.enable_ue_item_interaction,
        "ue_item_interaction_alpha_init": args.ue_item_interaction_alpha_init,
    }

    model = PCVRHyFormer(**model_args).to(args.device)

    # Log model sizing info.
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}, rank_mixer_ffn_mode={args.rank_mixer_ffn_mode}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    es_mode = 'min' if args.early_stop_metric == 'logloss' else 'max'
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
        mode=es_mode,
    )
    logging.info(f"EarlyStopping metric={args.early_stop_metric}, mode={es_mode}")

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        auc_loss_weight=args.auc_loss_weight,
        auc_margin=args.auc_margin,
        auc_max_pairs=args.auc_max_pairs,
        multi_task_loss=args.multi_task_loss,
        click_loss_weight=args.click_loss_weight,
        conversion_loss_weight=args.conversion_loss_weight,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        max_train_steps=args.max_train_steps,
        # 5/16 fix: train_config 存 raw csv str 会让 evaluation 时 infer.py
        # 处理失败 (eval_log.txt M23 evaluation '<=' int vs str error)。
        # infer.py 端已修复解析 raw str (resolve_dcn_fids · 兼容已有 ckpt)。
        # 同时这里 override · 让新 ckpt 的 train_config.json 直接存 list
        # of int · 与 model_args 保持一致 · 避免 raw str 再传给下游。
        # T38 (5/16): 把 fitted per-domain bucket boundaries 也存进 config
        # 给 infer.py 复现 (否则 infer.py fit 出来的 boundaries 与 train 不一致 ·
        # silent bug)。Dict[domain, List[int]] · None 时用 default ladder。
        train_config={
            **vars(args),
            "dcn_cross_user_fids": dcn_user_fid_indices,
            "dcn_cross_item_fids": dcn_item_fid_indices,
            "ue_split_fids_resolved": ue_split_fids_resolved,
            "ue_slices": ue_slices_resolved,
            "ue_offset": ue_offset_resolved,
            "ue_dim": ue_dim_resolved,
            "fitted_per_domain_bucket_boundaries": (
                {
                    d: arr.tolist()
                    for d, arr in pcvr_dataset._bucket_boundaries.items()
                }
                if isinstance(pcvr_dataset._bucket_boundaries, dict)
                else None
            ),
        },
        logloss_rebound_patience=args.logloss_rebound_patience,
        logloss_rebound_delta=args.logloss_rebound_delta,
        holdout_loader=holdout_loader,
        aux_valid_loaders=aux_valid_loaders,
        enable_aux_ckpt_select=args.enable_aux_ckpt_select,
        aux_ckpt_name=args.aux_ckpt_name,
        aux_ckpt_primary_tolerance=args.aux_ckpt_primary_tolerance,
        aux_ckpt_blend_weight=args.aux_ckpt_blend_weight,
        enable_torch_compile=args.enable_torch_compile,
        enable_amp=args.enable_amp,
        amp_dtype=args.amp_dtype,
        keep_topk_ckpt=args.keep_topk_ckpt,
        enable_weight_soup=args.enable_weight_soup,
        weight_soup_topk=args.weight_soup_topk,
        weight_soup_primary_tolerance=args.weight_soup_primary_tolerance,
        weight_soup_min_members=args.weight_soup_min_members,
        weight_soup_include_sparse=args.weight_soup_include_sparse,
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
