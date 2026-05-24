"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import os
import glob
import shutil
import logging
import json
import gc
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, combined_auc_loss, EarlyStopping
from model import ModelInput


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification.

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Metrics: BinaryAUROC + binary logloss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        auc_loss_weight: float = 0.5,
        auc_margin: float = 0.0,
        auc_max_pairs: int = 256,
        multi_task_loss: bool = False,
        click_loss_weight: float = 1.0,
        conversion_loss_weight: float = 2.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        max_train_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        logloss_rebound_patience: int = 0,
        logloss_rebound_delta: float = 0.0,
        holdout_loader: Optional[DataLoader] = None,
        aux_valid_loaders: Optional[Dict[str, DataLoader]] = None,
        enable_aux_ckpt_select: bool = False,
        aux_ckpt_name: str = 'm91',
        aux_ckpt_primary_tolerance: float = 0.0008,
        aux_ckpt_blend_weight: float = 0.5,
        enable_torch_compile: bool = False,
        enable_amp: bool = False,
        amp_dtype: str = 'bfloat16',
        keep_topk_ckpt: int = 1,
        enable_weight_soup: bool = False,
        weight_soup_topk: int = 4,
        weight_soup_primary_tolerance: float = 0.0010,
        weight_soup_min_members: int = 2,
        weight_soup_include_sparse: bool = False,
    ) -> None:
        self.model: nn.Module = model
        # ``_raw_model`` always points at the un-compiled instance. Used
        # for evaluation / predict paths (DECEM 2026-05-08 explicitly
        # recommends NOT compiling inference), for calls to
        # ``reinit_high_cardinality_params`` / ``get_sparse_params``
        # which rely on the original Python attributes, and for
        # re-wrapping compile after sparse reinit.
        self._raw_model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        # ADR-005 holdout loader. Evaluated end-of-epoch for a covariate-shift
        # aware generalization probe. Never drives best_model selection or
        # EarlyStopping counters. `None` = disabled (back-compat).
        self.holdout_loader: Optional[DataLoader] = holdout_loader
        self.last_val_holdout_auc: Optional[float] = None
        self.last_val_holdout_logloss: Optional[float] = None
        # Auxiliary validation loaders are read-only diagnostics evaluated
        # after the primary valid split. They never drive best_model
        # selection, top-k retention, early stopping, or logloss rebound.
        self.aux_valid_loaders: Dict[str, DataLoader] = aux_valid_loaders or {}
        self.last_aux_valid_metrics: Dict[str, Dict[str, float]] = {}
        self.enable_aux_ckpt_select: bool = bool(enable_aux_ckpt_select)
        self.aux_ckpt_name: str = str(aux_ckpt_name or 'm91')
        self.aux_ckpt_primary_tolerance: float = float(
            aux_ckpt_primary_tolerance)
        self.aux_ckpt_blend_weight: float = float(aux_ckpt_blend_weight)
        if not (0.0 <= self.aux_ckpt_blend_weight <= 1.0):
            raise ValueError(
                "aux_ckpt_blend_weight must be in [0, 1], "
                f"got {self.aux_ckpt_blend_weight}")
        if self.aux_ckpt_primary_tolerance < 0:
            raise ValueError(
                "aux_ckpt_primary_tolerance must be >= 0, "
                f"got {self.aux_ckpt_primary_tolerance}")
        self.aux_best_checkpoint_dir: Optional[str] = None
        self.aux_best_score: Optional[float] = None
        self.aux_best_primary_auc: Optional[float] = None
        self.aux_best_aux_auc: Optional[float] = None
        self.aux_best_global_step: Optional[int] = None
        self.blend_best_checkpoint_dir: Optional[str] = None
        self.blend_best_score: Optional[float] = None
        self.blend_best_primary_auc: Optional[float] = None
        self.blend_best_aux_auc: Optional[float] = None
        self.blend_best_global_step: Optional[int] = None
        self._aux_ckpt_missing_warned: bool = False
        # Optional post-hoc checkpoint weight soup. This is a side artifact:
        # it averages already-saved same-run top-k checkpoints after training
        # finishes, and never affects optimization, early stopping, best_model,
        # or top-k retention.
        self.enable_weight_soup: bool = bool(enable_weight_soup)
        self.weight_soup_topk: int = max(1, int(weight_soup_topk))
        self.weight_soup_primary_tolerance: float = float(
            weight_soup_primary_tolerance)
        self.weight_soup_min_members: int = max(2, int(weight_soup_min_members))
        self.weight_soup_include_sparse: bool = bool(
            weight_soup_include_sparse)
        if self.weight_soup_primary_tolerance < 0:
            raise ValueError(
                "weight_soup_primary_tolerance must be >= 0, "
                f"got {self.weight_soup_primary_tolerance}")
        self.weight_soup_checkpoint_dir: Optional[str] = None
        self.weight_soup_sources: list[Dict[str, Any]] = []
        self.weight_soup_error: Optional[str] = None
        self._weight_soup_attempted: bool = False
        self.writer = writer
        # schema_path is copied alongside every checkpoint so that infer.py can
        # rebuild the exact same feature schema the model was trained with.
        self.schema_path: Optional[str] = schema_path
        # ns_groups_path is optional; copied next to schema.json when provided
        # and points at an existing file. Keeping the JSON inside the ckpt dir
        # makes the checkpoint self-contained for evaluation environments that
        # do not ship ns_groups.json separately.
        self.ns_groups_path: Optional[str] = ns_groups_path

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.98)
            )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.98)
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir

        # T27 · torch.compile + AMP (training-only).
        # Community notes suggested applying torch.compile during training
        # but not during inference. AMP cuts memory
        # traffic by ~40% on GPU; bfloat16 (default) avoids loss-scaling
        # and is numerically stable for sparse-embedding lookups.
        self.enable_torch_compile: bool = bool(enable_torch_compile)
        self.enable_amp: bool = bool(enable_amp)
        if amp_dtype not in ('bfloat16', 'float16'):
            raise ValueError(
                f"amp_dtype must be 'bfloat16' or 'float16', "
                f"got {amp_dtype!r}")
        self.amp_dtype: torch.dtype = (
            torch.bfloat16 if amp_dtype == 'bfloat16' else torch.float16
        )
        # fp16 needs a loss scaler; bf16 does not.
        self._amp_needs_scaler: bool = (self.amp_dtype == torch.float16)
        self.amp_scaler: Optional["torch.amp.GradScaler"] = None
        if self.enable_amp and self._amp_needs_scaler:
            self.amp_scaler = torch.amp.GradScaler('cuda')
        # Wrap model with torch.compile if requested. Sparse Embeddings
        # + dynamic batch shape can trigger recompilations; keep the
        # default 'default' mode (not 'reduce-overhead') so dynamo falls
        # back gracefully. Suppress errors to prevent dynamo bugs from
        # killing a run.
        if self.enable_torch_compile:
            if device.startswith('cuda'):
                dynamo_cfg = getattr(
                    getattr(torch, '_dynamo', None), 'config', None)
                if dynamo_cfg is not None:
                    dynamo_cfg.suppress_errors = True
                self.model = torch.compile(self._raw_model, mode='default')
                logging.info(
                    "[T27] torch.compile ENABLED (train-path only, "
                    "mode='default', dynamo errors suppressed)")
            else:
                logging.warning(
                    "[T27] torch.compile requested but device=%s is not "
                    "cuda; leaving model uncompiled.", device)
                self.enable_torch_compile = False
        if self.enable_amp:
            logging.info(
                f"[T27] AMP ENABLED (dtype={amp_dtype}, "
                f"scaler={'on' if self._amp_needs_scaler else 'off (bf16)'})"
            )

        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        # CombinedAUCLoss parameters (T18 / ADR-006). Only consulted when
        # loss_type == 'auc'; silently ignored otherwise.
        self.auc_loss_weight: float = float(auc_loss_weight)
        self.auc_margin: float = float(auc_margin)
        self.auc_max_pairs: int = int(auc_max_pairs)
        if loss_type == 'auc' and not (0.0 <= self.auc_loss_weight <= 1.0):
            raise ValueError(
                f"--auc_loss_weight must be in [0, 1], got {self.auc_loss_weight}"
            )
        if loss_type == 'auc' and self.auc_max_pairs <= 0:
            raise ValueError(
                f"--auc_max_pairs must be > 0, got {self.auc_max_pairs}"
            )
        self.multi_task_loss: bool = multi_task_loss
        self.click_loss_weight: float = click_loss_weight
        self.conversion_loss_weight: float = conversion_loss_weight
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.max_train_steps: int = max_train_steps
        self.train_config: Optional[Dict[str, Any]] = train_config
        self.validation_history = []
        self.best_epoch: Optional[int] = None
        self.best_global_step: Optional[int] = None
        self.best_val_auc: Optional[float] = None
        self.best_val_logloss: Optional[float] = None
        self.best_checkpoint_dir: Optional[str] = None
        self.last_epoch: Optional[int] = None
        self.last_global_step: int = 0
        self.last_val_auc: Optional[float] = None
        self.last_val_logloss: Optional[float] = None

        # ── keep_topk_ckpt (2026-05-09 · EXP-024 Method P2 + 冲 0.85 战略) ──
        # Retain up to K checkpoints ranked by val_auc (descending). The
        # global best is still tracked via ``best_checkpoint_dir`` and
        # lives in a ``*.best_model`` directory (platform Publish
        # requirement). Runners-up live in ``*.topk_{rank}`` dirs and are
        # renamed on every update so the filesystem mirrors the current
        # ranking exactly. Rationale: prior public-eval checks showed that
        # "best validation AUC" is not always "best leaderboard checkpoint";
        # keeping top-K lets us send every plausible ckpt to evaluation
        # rather than committing to the AUC argmax. Default 1 = current
        # behavior, fully backward-compat.
        self.keep_topk_ckpt: int = max(1, int(keep_topk_ckpt))
        # List of (val_auc, val_logloss, global_step, ckpt_dir). Kept
        # sorted by val_auc DESCENDING. Length <= keep_topk_ckpt.
        self._topk_ckpts: list[tuple[float, float, int, str]] = []

        # LogLoss rebound co-monitor (complements AUC-based EarlyStopping).
        # Motivation (2026-05-04 t13tf-loss-full LB incident): AUC-only ES
        # keeps training past the point where LogLoss starts rebounding,
        # which empirically correlates with LB degradation (valid AUC +0.0005
        # vs LB AUC -0.006 between E4 and E9). The guard tracks the best
        # val_logloss observed so far; once val_logloss exceeds that best by
        # more than `logloss_rebound_delta` for `logloss_rebound_patience`
        # consecutive validations, we flip `early_stopping.early_stop=True`.
        # Disabled when `logloss_rebound_patience <= 0` (back-compat).
        self.logloss_rebound_patience: int = int(logloss_rebound_patience)
        self.logloss_rebound_delta: float = float(logloss_rebound_delta)
        self._best_logloss_for_guard: Optional[float] = None
        self._logloss_rebound_counter: int = 0

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"auc_loss_weight={auc_loss_weight}, "
                     f"auc_margin={auc_margin}, auc_max_pairs={auc_max_pairs}, "
                     f"multi_task_loss={multi_task_loss}, "
                     f"click_loss_weight={click_loss_weight}, "
                     f"conversion_loss_weight={conversion_loss_weight}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}")
        if self.max_train_steps < 0:
            raise ValueError("max_train_steps must be >= 0")
        if self.max_train_steps > 0:
            logging.info(
                f"Fast proxy enabled: max_train_steps={self.max_train_steps}")
        if self.logloss_rebound_patience > 0:
            logging.info(
                f"LogLoss rebound guard ENABLED: patience="
                f"{self.logloss_rebound_patience}, delta="
                f"{self.logloss_rebound_delta:.6f}. Training will stop once "
                f"val_logloss worsens relative to the best observed for "
                f"{self.logloss_rebound_patience} consecutive validation(s), "
                f"regardless of AUC trend. best_model is still AUC-selected.")
        else:
            logging.info("LogLoss rebound guard DISABLED (patience<=0)")
        if self.holdout_loader is not None:
            logging.info(
                "[ADR-005] holdout_loader attached: holdout AUC/LogLoss "
                "will be logged every epoch. holdout never drives best_model "
                "selection, EarlyStopping, or LogLoss rebound guard.")
        if self.aux_valid_loaders:
            logging.info(
                "[aux_valid] attached auxiliary validation windows: "
                f"{list(self.aux_valid_loaders)}. These metrics are logged "
                "only and never drive checkpoint selection.")
        if self.enable_aux_ckpt_select:
            logging.info(
                "[aux_ckpt_select] ENABLED: name=%s, primary_tolerance=%.6f, "
                "blend_weight=%.3f. Primary best_model/top-k/early-stopping "
                "remain unchanged; aux/blend checkpoints are extra side "
                "candidates.",
                self.aux_ckpt_name,
                self.aux_ckpt_primary_tolerance,
                self.aux_ckpt_blend_weight,
            )
        if self.enable_weight_soup:
            logging.info(
                "[weight_soup] ENABLED: topk=%d, primary_tolerance=%.6f, "
                "min_members=%d, include_sparse=%s. The soup checkpoint is "
                "created after training from saved best/top-k checkpoints "
                "only.",
                self.weight_soup_topk,
                self.weight_soup_primary_tolerance,
                self.weight_soup_min_members,
                self.weight_soup_include_sparse,
            )

    def _record_validation_result(
        self,
        epoch: int,
        total_step: int,
        val_auc: float,
        val_logloss: float,
        previous_best: Optional[float],
    ) -> None:
        """Track compact validation history for end-of-run summaries."""
        # ADR-005: evaluate holdout immediately after the primary valid so the
        # two metrics share the same training state. holdout is purely a
        # read-only probe: it does NOT influence best_model selection,
        # EarlyStopping, or the LogLoss rebound guard.
        holdout_auc: Optional[float] = None
        holdout_logloss: Optional[float] = None
        if self.holdout_loader is not None:
            try:
                result = self.evaluate_holdout()
                if result is not None:
                    holdout_auc, holdout_logloss = result
                    self.last_val_holdout_auc = float(holdout_auc)
                    self.last_val_holdout_logloss = float(holdout_logloss)
                    logging.info(
                        f"[ADR-005] Epoch {epoch} step {total_step} "
                        f"holdout AUC: {holdout_auc:.6f}, "
                        f"LogLoss: {holdout_logloss:.6f} "
                        f"(valid AUC: {val_auc:.6f}, "
                        f"LogLoss: {val_logloss:.6f}; "
                        f"gap_auc=valid-holdout="
                        f"{val_auc - holdout_auc:+.6f})")
            except Exception as e:
                # Never let holdout eval crash training.
                logging.warning(f"[ADR-005] holdout evaluation failed: {e}")

        aux_metrics: Dict[str, Dict[str, float]] = {}
        for name, loader in self.aux_valid_loaders.items():
            try:
                aux_auc, aux_logloss = self.evaluate_aux_valid(name, loader)
                aux_metrics[name] = {
                    "auc": float(aux_auc),
                    "logloss": float(aux_logloss),
                }
                logging.info(
                    f"[aux_valid/{name}] Epoch {epoch} step {total_step} "
                    f"AUC: {aux_auc:.6f}, LogLoss: {aux_logloss:.6f} "
                    f"(valid AUC: {val_auc:.6f}, LogLoss: {val_logloss:.6f}; "
                    f"gap_auc=valid-aux={val_auc - aux_auc:+.6f})")
            except Exception as e:
                logging.warning(f"[aux_valid/{name}] evaluation failed: {e}")
        if aux_metrics:
            self.last_aux_valid_metrics = aux_metrics

        is_new_best = (
            self.early_stopping.best_score is not None
            and (previous_best is None
                 or self.early_stopping.best_score != previous_best)
        )
        row = {
            "epoch": int(epoch),
            "global_step": int(total_step),
            "val_auc": float(val_auc),
            "val_logloss": float(val_logloss),
            "is_best": bool(is_new_best),
        }
        if holdout_auc is not None and holdout_logloss is not None:
            row["holdout_auc"] = float(holdout_auc)
            row["holdout_logloss"] = float(holdout_logloss)
        if aux_metrics:
            row["aux_valid"] = aux_metrics
        self.validation_history.append(row)
        self.last_epoch = int(epoch)
        self.last_global_step = int(total_step)
        self.last_val_auc = float(val_auc)
        self.last_val_logloss = float(val_logloss)
        if is_new_best:
            self.best_epoch = int(epoch)
            self.best_global_step = int(total_step)
            self.best_val_auc = float(val_auc)
            self.best_val_logloss = float(val_logloss)
            self.best_checkpoint_dir = os.path.dirname(
                self.early_stopping.checkpoint_path)

        self._maybe_update_aux_selected_ckpts(
            total_step=total_step,
            val_auc=val_auc,
            val_logloss=val_logloss,
            aux_metrics=aux_metrics,
        )

        self._check_logloss_rebound_guard(
            epoch=epoch, total_step=total_step, val_logloss=val_logloss)

        # Top-K ring maintenance (2026-05-09). No-op when keep_topk_ckpt==1.
        # Runs on every validation regardless of whether this is a new best,
        # so that e.g. E3 best + E4 drop both get a chance to enter the
        # ring when K>=2 (covers the icq-full E6 vs E4 LB-inversion case).
        self._maintain_topk_ckpts(total_step, val_auc, val_logloss)

    def _check_logloss_rebound_guard(
        self,
        epoch: int,
        total_step: int,
        val_logloss: float,
    ) -> None:
        """Co-monitor that trips EarlyStopping when val_logloss rebounds.

        Complements the primary AUC-based ``EarlyStopping`` (which selects the
        global best_model by AUC) by catching the overfitting regime where
        AUC still climbs but LogLoss has already turned upward. See
        ``__init__`` docstring for motivation (2026-05-04 t13tf-loss-full LB
        incident: +0.0005 valid AUC between E4→E9 translated into −0.006 LB
        AUC because the model kept sharpening toward the valid tail's local
        time patterns, which do not transfer to the test window).

        Semantics:
        - Track ``_best_logloss_for_guard`` (lowest val_logloss observed).
        - A validation is "rebound" iff
          ``val_logloss > best_logloss + logloss_rebound_delta``.
        - Rebound increments ``_logloss_rebound_counter``; a non-rebound
          validation resets the counter to 0 (consecutive-only trip).
        - When counter reaches ``logloss_rebound_patience``, we flip
          ``early_stopping.early_stop=True`` so the main loop terminates
          after finishing the current epoch-end handler.

        No-op when ``logloss_rebound_patience <= 0`` (back-compat with
        existing configs and jobs that predate this guard).
        """
        if self.logloss_rebound_patience <= 0:
            return

        if self._best_logloss_for_guard is None:
            self._best_logloss_for_guard = float(val_logloss)
            return

        # Strictly-worse-than-best is a rebound; equality is tolerated to
        # avoid tripping on FP noise in plateau regions.
        is_rebound = (
            float(val_logloss)
            > self._best_logloss_for_guard + self.logloss_rebound_delta
        )

        if is_rebound:
            self._logloss_rebound_counter += 1
            logging.info(
                f"[logloss-guard] epoch={epoch} step={total_step} "
                f"val_logloss={val_logloss:.6f} > best="
                f"{self._best_logloss_for_guard:.6f} + delta="
                f"{self.logloss_rebound_delta:.6f}; "
                f"rebound_counter={self._logloss_rebound_counter}/"
                f"{self.logloss_rebound_patience}")
            if self._logloss_rebound_counter >= self.logloss_rebound_patience:
                logging.info(
                    f"[logloss-guard] Triggering EarlyStopping: "
                    f"val_logloss rebound sustained for "
                    f"{self._logloss_rebound_counter} validation(s). "
                    f"best_model (AUC-selected) remains the final ckpt.")
                self.early_stopping.early_stop = True
        else:
            if self._logloss_rebound_counter > 0:
                logging.info(
                    f"[logloss-guard] epoch={epoch} step={total_step} "
                    f"val_logloss recovered to {val_logloss:.6f}; "
                    f"counter reset from {self._logloss_rebound_counter}")
                self._logloss_rebound_counter = 0
            if float(val_logloss) < self._best_logloss_for_guard:
                self._best_logloss_for_guard = float(val_logloss)

    @staticmethod
    def _load_checkpoint_state_dict(model_path: str) -> Dict[str, torch.Tensor]:
        """Load a checkpoint state_dict on CPU across torch versions."""
        try:
            state = torch.load(
                model_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError(
                f"Expected state_dict dict from {model_path}, "
                f"got {type(state).__name__}")
        return state

    def _collect_weight_soup_sources(self) -> list[Dict[str, Any]]:
        """Return deduplicated, score-filtered checkpoint sources for soup."""
        if self.best_val_auc is None:
            return []

        candidates: list[Dict[str, Any]] = []

        def add_candidate(
            *,
            kind: str,
            rank: Optional[int],
            val_auc: Optional[float],
            val_logloss: Optional[float],
            global_step: Optional[int],
            ckpt_dir: Optional[str],
        ) -> None:
            if not ckpt_dir:
                return
            model_path = os.path.join(ckpt_dir, "model.pt")
            if not os.path.exists(model_path):
                return
            candidates.append({
                "kind": kind,
                "rank": rank,
                "val_auc": (
                    float(val_auc) if val_auc is not None else None
                ),
                "val_logloss": (
                    float(val_logloss) if val_logloss is not None else None
                ),
                "global_step": (
                    int(global_step) if global_step is not None else None
                ),
                "ckpt_dir": ckpt_dir,
                "model_path": model_path,
            })

        add_candidate(
            kind="best_model",
            rank=None,
            val_auc=self.best_val_auc,
            val_logloss=self.best_val_logloss,
            global_step=self.best_global_step,
            ckpt_dir=self.best_checkpoint_dir,
        )
        for rank, (auc, ll, step, ckpt_dir) in enumerate(
            self._topk_ckpts, start=1
        ):
            add_candidate(
                kind="topk",
                rank=rank,
                val_auc=auc,
                val_logloss=ll,
                global_step=step,
                ckpt_dir=ckpt_dir,
            )

        seen: set[Any] = set()
        deduped: list[Dict[str, Any]] = []
        for item in candidates:
            key = item["global_step"]
            if key is None:
                key = os.path.realpath(item["model_path"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        if not deduped:
            return []

        scored_aucs = [
            float(item["val_auc"])
            for item in deduped
            if item["val_auc"] is not None
        ]
        if not scored_aucs:
            return []
        best_auc = max(scored_aucs)
        auc_floor = best_auc - self.weight_soup_primary_tolerance
        eligible = [
            item for item in deduped
            if item["val_auc"] is not None and item["val_auc"] >= auc_floor
        ]
        eligible.sort(key=lambda item: (
            -float(item["val_auc"]),
            int(item["global_step"] or 0),
        ))
        return eligible[:self.weight_soup_topk]

    def _build_weight_soup_dir_name(
        self,
        member_count: int,
        best_auc: float,
    ) -> str:
        parts = [f"global_step{self.last_global_step or 0}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        parts.extend([
            f"auc={best_auc:.6f}",
            f"members={member_count}",
            "weight_soup",
        ])
        return ".".join(parts)

    def _remove_old_weight_soup_dirs(self) -> None:
        pattern = os.path.join(self.save_dir, "global_step*.weight_soup")
        for old_dir in glob.glob(pattern):
            if os.path.isdir(old_dir):
                shutil.rmtree(old_dir)
                logging.info("[weight_soup] removed old dir: %s", old_dir)

    def _get_sparse_state_keys(self) -> set[str]:
        """Return state_dict keys that belong to nn.Embedding weights."""
        sparse_keys: set[str] = set()
        for module_name, module in self._raw_model.named_modules():
            if isinstance(module, nn.Embedding):
                key = f"{module_name}.weight" if module_name else "weight"
                sparse_keys.add(key)
        return sparse_keys

    def _average_checkpoint_state_dicts(
        self,
        sources: list[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Average floating tensors from same-architecture checkpoints.

        Non-floating tensors are copied from the first checkpoint after shape
        and dtype compatibility checks. Accumulation is CPU fp32 for numerical
        stability, then cast back to each tensor's original dtype before save.
        By default, sparse embedding tensors are also copied from the best
        source instead of averaged because this project cold-restarts
        high-cardinality embeddings at epoch boundaries.
        """
        if not sources:
            raise ValueError("weight soup requires at least one source")

        acc: Optional[Dict[str, torch.Tensor]] = None
        expected_keys: Optional[set[str]] = None
        floating_keys: set[str] = set()
        original_dtypes: Dict[str, torch.dtype] = {}
        sparse_keys = (
            set()
            if self.weight_soup_include_sparse
            else self._get_sparse_state_keys()
        )
        if sparse_keys:
            logging.info(
                "[weight_soup] dense-only averaging: preserving %d sparse "
                "embedding tensor(s) from the best source.",
                len(sparse_keys),
            )

        for index, source in enumerate(sources):
            model_path = source["model_path"]
            logging.info(
                "[weight_soup] loading source %d/%d: %s",
                index + 1,
                len(sources),
                model_path,
            )
            state = self._load_checkpoint_state_dict(model_path)
            state_keys = set(state.keys())
            if index == 0:
                expected_keys = state_keys
                acc = {}
                for key, tensor in state.items():
                    if not isinstance(tensor, torch.Tensor):
                        raise ValueError(
                            f"Unexpected non-tensor value for key {key!r} "
                            f"in {model_path}: {type(tensor).__name__}")
                    cpu_tensor = tensor.detach().cpu()
                    original_dtypes[key] = cpu_tensor.dtype
                    if (
                        torch.is_floating_point(cpu_tensor)
                        and key not in sparse_keys
                    ):
                        floating_keys.add(key)
                        acc[key] = cpu_tensor.float()
                    else:
                        acc[key] = cpu_tensor
            else:
                if expected_keys is None or acc is None:
                    raise RuntimeError("weight soup accumulator not initialized")
                missing = expected_keys - state_keys
                extra = state_keys - expected_keys
                if missing or extra:
                    raise ValueError(
                        "Checkpoint key mismatch while building weight soup: "
                        f"missing={sorted(missing)[:5]}, "
                        f"extra={sorted(extra)[:5]}")
                for key in expected_keys:
                    other = state[key]
                    if not isinstance(other, torch.Tensor):
                        raise ValueError(
                            f"Unexpected non-tensor value for key {key!r} "
                            f"in {model_path}: {type(other).__name__}")
                    base = acc[key]
                    other_cpu = other.detach().cpu()
                    if base.shape != other_cpu.shape:
                        raise ValueError(
                            f"Shape mismatch for key {key!r}: "
                            f"{tuple(base.shape)} vs {tuple(other_cpu.shape)}")
                    if key in floating_keys:
                        acc[key].add_(other_cpu.float())
                    elif base.dtype != other_cpu.dtype:
                        raise ValueError(
                            f"Dtype mismatch for key {key!r}: "
                            f"{base.dtype} vs {other_cpu.dtype}")
            del state
            gc.collect()

        if acc is None:
            raise RuntimeError("weight soup accumulator is empty")
        member_count = len(sources)
        for key in floating_keys:
            acc[key].div_(member_count)
            target_dtype = original_dtypes[key]
            if acc[key].dtype != target_dtype:
                acc[key] = acc[key].to(dtype=target_dtype)
        return acc

    def _maybe_create_weight_soup_checkpoint(self) -> None:
        """Create one post-hoc weight-soup side checkpoint if enabled."""
        if not self.enable_weight_soup or self._weight_soup_attempted:
            return
        self._weight_soup_attempted = True
        try:
            sources = self._collect_weight_soup_sources()
            self.weight_soup_sources = sources
            if len(sources) < self.weight_soup_min_members:
                self.weight_soup_error = (
                    f"not enough eligible checkpoints: {len(sources)} "
                    f"< {self.weight_soup_min_members}; "
                    "set --keep_topk_ckpt > 1 and/or relax "
                    "--weight_soup_primary_tolerance"
                )
                logging.info("[weight_soup] skipped: %s", self.weight_soup_error)
                return

            best_auc = max(float(src["val_auc"]) for src in sources)
            logging.info(
                "[weight_soup] building soup from %d source(s): %s",
                len(sources),
                ", ".join(
                    f"{src['kind']}@step{src['global_step']}:"
                    f"auc={src['val_auc']:.6f}"
                    for src in sources
                ),
            )
            soup_state = self._average_checkpoint_state_dicts(sources)
            self._remove_old_weight_soup_dirs()
            dir_name = self._build_weight_soup_dir_name(
                member_count=len(sources),
                best_auc=best_auc,
            )
            ckpt_dir = os.path.join(self.save_dir, dir_name)
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(soup_state, os.path.join(ckpt_dir, "model.pt"))
            del soup_state
            gc.collect()
            self._write_sidecar_files(ckpt_dir)
            metadata = {
                "member_count": len(sources),
                "best_source_auc": best_auc,
                "primary_tolerance": self.weight_soup_primary_tolerance,
                "include_sparse": self.weight_soup_include_sparse,
                "sources": sources,
            }
            with open(
                os.path.join(ckpt_dir, "soup_metadata.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
            self.weight_soup_checkpoint_dir = ckpt_dir
            logging.info(
                "[weight_soup] saved checkpoint to %s/model.pt", ckpt_dir)
        except Exception as exc:
            self.weight_soup_error = repr(exc)
            logging.warning("[weight_soup] failed: %s", exc)

    def _write_training_summary(self, stop_reason: str) -> None:
        """Emit a grep-friendly summary and persist it next to logs/checkpoints."""
        self._maybe_create_weight_soup_checkpoint()
        summary = {
            "stop_reason": stop_reason,
            "best_val_auc": self.best_val_auc,
            "best_val_logloss": self.best_val_logloss,
            "best_epoch": self.best_epoch,
            "best_global_step": self.best_global_step,
            "best_checkpoint_dir": self.best_checkpoint_dir,
            "last_val_auc": self.last_val_auc,
            "last_val_logloss": self.last_val_logloss,
            "last_epoch": self.last_epoch,
            "last_global_step": self.last_global_step,
            "early_stopping_counter": self.early_stopping.counter,
            "early_stopping_patience": self.early_stopping.patience,
            "logloss_rebound_patience": self.logloss_rebound_patience,
            "logloss_rebound_delta": self.logloss_rebound_delta,
            "logloss_rebound_counter": self._logloss_rebound_counter,
            "best_logloss_for_guard": self._best_logloss_for_guard,
            "validation_history": self.validation_history,
            "last_aux_valid_metrics": self.last_aux_valid_metrics,
            "aux_ckpt_select": {
                "enabled": self.enable_aux_ckpt_select,
                "name": self.aux_ckpt_name,
                "primary_tolerance": self.aux_ckpt_primary_tolerance,
                "blend_weight": self.aux_ckpt_blend_weight,
                "aux_best_checkpoint_dir": self.aux_best_checkpoint_dir,
                "aux_best_score": self.aux_best_score,
                "aux_best_primary_auc": self.aux_best_primary_auc,
                "aux_best_aux_auc": self.aux_best_aux_auc,
                "aux_best_global_step": self.aux_best_global_step,
                "blend_best_checkpoint_dir": self.blend_best_checkpoint_dir,
                "blend_best_score": self.blend_best_score,
                "blend_best_primary_auc": self.blend_best_primary_auc,
                "blend_best_aux_auc": self.blend_best_aux_auc,
                "blend_best_global_step": self.blend_best_global_step,
            },
            "weight_soup": {
                "enabled": self.enable_weight_soup,
                "topk": self.weight_soup_topk,
                "primary_tolerance": self.weight_soup_primary_tolerance,
                "min_members": self.weight_soup_min_members,
                "include_sparse": self.weight_soup_include_sparse,
                "checkpoint_dir": self.weight_soup_checkpoint_dir,
                "sources": self.weight_soup_sources,
                "error": self.weight_soup_error,
            },
            "keep_topk_ckpt": self.keep_topk_ckpt,
            "topk_checkpoints": [
                {
                    "rank": rank,
                    "val_auc": auc,
                    "val_logloss": ll,
                    "global_step": step,
                    "ckpt_dir": ckpt_dir,
                }
                for rank, (auc, ll, step, ckpt_dir) in enumerate(
                    self._topk_ckpts, start=1)
            ],
            "key_train_config": {},
        }
        if self.train_config:
            key_names = (
                "seq_encoder_type", "valid_split_strategy", "valid_ratio",
                "train_ratio", "limit_train_rgs", "limit_valid_rgs",
                "max_train_steps", "enable_row_time_cutoff",
                "aux_valid_windows",
                "enable_aux_ckpt_select", "aux_ckpt_name",
                "aux_ckpt_primary_tolerance", "aux_ckpt_blend_weight",
                "enable_weight_soup", "weight_soup_topk",
                "weight_soup_primary_tolerance",
                "weight_soup_min_members", "weight_soup_include_sparse",
                "enable_count_features", "enable_seq_stats_features",
                "enable_history_cvr_features", "history_cvr_item_fids",
                "history_cvr_cache_path",
                "history_cvr_bin_sec", "history_cvr_cutoff_sec",
                "history_cvr_time_mode", "history_cvr_available_lag_sec",
                "history_cvr_prior_strength",
                "enable_mature_negative_weighting",
                "negative_maturity_sec", "immature_negative_weight",
                "d_model", "num_queries",
                "enable_din_interest", "din_interest_source",
                "din_interest_merge", "enable_tin_interest",
                "tin_time_alpha_init",
                "dropout_rate", "loss_type", "multi_task_loss",
                "click_loss_weight", "conversion_loss_weight",
                "auc_loss_weight", "auc_margin", "auc_max_pairs",
                "rank_mixer_mode", "rank_mixer_ffn_mode",
                "action_num", "batch_size", "num_workers", "seed",
            )
            summary["key_train_config"] = {
                k: self.train_config.get(k) for k in key_names
                if k in self.train_config
            }

        # EDA data (if dataset collected it during training)
        try:
            train_ds = getattr(self.train_loader, 'dataset', None)
            if train_ds is not None and hasattr(train_ds, 'finalize_eda'):
                eda_data = train_ds.finalize_eda()
                if eda_data is not None:
                    summary['eda'] = eda_data
                    logging.info(
                        f"[summary] EDA collected: "
                        f"{eda_data['global']['total_samples_observed']} "
                        f"samples observed, "
                        f"user_id_distinct={eda_data['q1_user_id'].get('distinct', 0)}, "
                        f"item_id_distinct={eda_data['q1_item_id'].get('distinct', 0)}"
                    )
        except Exception as e:
            logging.warning(f"[summary] EDA finalize failed: {e}")

        logging.info("[summary] ===== Training Summary =====")
        logging.info(f"[summary] stop_reason={stop_reason}")
        logging.info(
            f"[summary] best_val_AUC={self.best_val_auc} "
            f"best_val_logloss={self.best_val_logloss} "
            f"best_epoch={self.best_epoch} "
            f"best_global_step={self.best_global_step}")
        logging.info(
            f"[summary] last_val_AUC={self.last_val_auc} "
            f"last_val_logloss={self.last_val_logloss} "
            f"last_epoch={self.last_epoch} "
            f"last_global_step={self.last_global_step}")
        logging.info(f"[summary] best_checkpoint_dir={self.best_checkpoint_dir}")
        if self.validation_history:
            compact = ", ".join(
                f"E{r['epoch']}@{r['global_step']}:"
                f"{r['val_auc']:.6f}"
                f"{'*' if r['is_best'] else ''}"
                for r in self.validation_history
            )
            logging.info(f"[summary] val_auc_history={compact}")
        if self.keep_topk_ckpt > 1 and self._topk_ckpts:
            topk_compact = ", ".join(
                f"rank{rank}@step{step}:auc={auc:.6f}"
                for rank, (auc, _, step, _) in enumerate(
                    self._topk_ckpts, start=1)
            )
            logging.info(
                f"[summary] topk_ckpts (K={self.keep_topk_ckpt}): "
                f"{topk_compact}")
        if self.enable_aux_ckpt_select:
            logging.info(
                "[summary] aux_ckpt_select aux_best=%s blend_best=%s",
                self.aux_best_checkpoint_dir,
                self.blend_best_checkpoint_dir,
            )
        if self.enable_weight_soup:
            logging.info(
                "[summary] weight_soup checkpoint=%s error=%s",
                self.weight_soup_checkpoint_dir,
                self.weight_soup_error,
            )
        logging.info("[summary] ===== End Training Summary =====")

        out_paths = [os.path.join(self.save_dir, "training_summary.json")]
        if self.train_config and self.train_config.get("log_dir"):
            out_paths.append(os.path.join(
                self.train_config["log_dir"], "training_summary.json"))
        for path in out_paths:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, sort_keys=True)
                logging.info(f"[summary] wrote {path}")
            except Exception as e:
                logging.warning(f"[summary] failed to write {path}: {e}")

    def _build_step_dir_name(
        self,
        global_step: int,
        is_best: bool = False,
        val_auc: Optional[float] = None,
    ) -> str:
        """Build a checkpoint sub-directory name such as
        ``global_step2500.layer=2.head=4.hidden=64[.auc=0.865000][.best_model]``.
        """
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        if is_best and val_auc is not None:
            # Platform checkpoint names must be <=300 chars and only use
            # alnum, "_", "-", "=", ".". Fixed precision is readable and
            # stable enough for selecting the best local-validation ckpt.
            parts.append(f"auc={val_auc:.6f}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``.

        Currently persists up to three files, all overwritten on every call:

        - ``schema.json`` (copied from ``self.schema_path``): feature layout
          metadata needed to rebuild the Parquet dataset.
        - ``ns_groups.json`` (copied from ``self.ns_groups_path`` when set
          and the file exists): NS-token grouping used to construct the
          tokenizer. Making a per-ckpt copy lets evaluation environments
          consume the checkpoint without having to ship the original
          project-level ``ns_groups.json``.
        - ``train_config.json`` (serialized from ``self.train_config``):
          full set of training-time hyperparameters. When ``ns_groups.json``
          is copied into ``ckpt_dir``, the ``ns_groups_json`` field is
          rewritten to the bare filename so that ``infer.py`` resolves it
          against ``ckpt_dir`` rather than the original absolute path on
          the training machine.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config:
            import json
            cfg_to_dump = dict(self.train_config)
            if ns_groups_copied:
                # Override the stored path to a filename relative to ckpt_dir;
                # infer.py already falls back to `<ckpt_dir>/<basename>` when
                # the recorded path is not absolute, which keeps the ckpt
                # portable across hosts.
                cfg_to_dump['ns_groups_json'] = os.path.basename(
                    self.ns_groups_path)
            history_path = cfg_to_dump.get('history_cvr_cache_path')
            if (
                cfg_to_dump.get('enable_history_cvr_features')
                and history_path
                and os.path.exists(history_path)
            ):
                shutil.copy2(history_path, ckpt_dir)
                cfg_to_dump['history_cvr_cache_path'] = os.path.basename(
                    history_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        val_auc: Optional[float] = None,
        skip_model_file: bool = False,
    ) -> str:
        """Save ``model.pt`` plus sidecar files under a ``global_step`` sub-dir.

        Args:
            global_step: current global step used to name the directory.
            is_best: whether this is a new-best checkpoint.
            val_auc: optional validation AUC included in best checkpoint names.
            skip_model_file: if True, skip writing ``model.pt`` (because the
                caller, e.g. EarlyStopping, has already persisted it to the
                same path). Sidecar files are still (re)written.

        Returns:
            The absolute path of the checkpoint directory.
        """
        dir_name = self._build_step_dir_name(
            global_step, is_best=is_best, val_auc=val_auc)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self._raw_model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    @staticmethod
    def _safe_ckpt_token(raw: str) -> str:
        """Sanitize a token for platform checkpoint directory names."""
        safe_chars = []
        for ch in str(raw):
            if ch.isalnum() or ch in ("_", "-", "=", "."):
                safe_chars.append(ch)
            else:
                safe_chars.append("_")
        return "".join(safe_chars).strip("._") or "aux"

    def _build_aux_select_dir_name(
        self,
        global_step: int,
        kind: str,
        score: float,
        val_auc: float,
        aux_auc: float,
    ) -> str:
        aux_name = self._safe_ckpt_token(self.aux_ckpt_name)
        kind_token = self._safe_ckpt_token(kind)
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        parts.extend([
            f"score={score:.6f}",
            f"auc={val_auc:.6f}",
            f"aux_{aux_name}={aux_auc:.6f}",
            f"{kind_token}_{aux_name}_best",
        ])
        return ".".join(parts)

    def _remove_old_aux_select_dirs(self, kind: str) -> None:
        kind_token = self._safe_ckpt_token(kind)
        aux_name = self._safe_ckpt_token(self.aux_ckpt_name)
        pattern = os.path.join(
            self.save_dir, f"global_step*.{kind_token}_{aux_name}_best")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(
                f"[aux_ckpt_select] removed old {kind_token} dir: {old_dir}")

    def _save_aux_select_checkpoint(
        self,
        global_step: int,
        kind: str,
        score: float,
        val_auc: float,
        aux_auc: float,
    ) -> str:
        kind_token = self._safe_ckpt_token(kind)
        self._remove_old_aux_select_dirs(kind_token)
        dir_name = self._build_aux_select_dir_name(
            global_step=global_step,
            kind=kind_token,
            score=score,
            val_auc=val_auc,
            aux_auc=aux_auc,
        )
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(
            self._raw_model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(
            f"[aux_ckpt_select] saved {kind_token} checkpoint to "
            f"{ckpt_dir}/model.pt")
        return ckpt_dir

    def _maybe_update_aux_selected_ckpts(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
        aux_metrics: Dict[str, Dict[str, float]],
    ) -> None:
        """Save extra aux/blend-best checkpoints without touching primary ES.

        This is intentionally a side selector. The primary AUC best_model and
        top-k ring remain the canonical platform-safe checkpoints. Aux/blend
        candidates are only materialized when the current primary AUC is close
        enough to the current best primary AUC, preventing a public-like aux
        spike from selecting a badly underfit primary model.
        """
        del val_logloss  # reserved for future calibrated blend variants
        if not self.enable_aux_ckpt_select:
            return
        if not aux_metrics:
            return
        if self.aux_ckpt_name not in aux_metrics:
            if not self._aux_ckpt_missing_warned:
                logging.warning(
                    "[aux_ckpt_select] aux window %r not found in metrics %s; "
                    "side checkpoint selection disabled for now.",
                    self.aux_ckpt_name,
                    sorted(aux_metrics),
                )
                self._aux_ckpt_missing_warned = True
            return

        primary_ref = (
            float(self.best_val_auc)
            if self.best_val_auc is not None
            else float(val_auc)
        )
        primary_floor = primary_ref - self.aux_ckpt_primary_tolerance
        if float(val_auc) < primary_floor:
            logging.info(
                "[aux_ckpt_select] skip step=%s: primary_auc=%.6f below "
                "floor %.6f (best_primary=%.6f, tolerance=%.6f)",
                total_step,
                val_auc,
                primary_floor,
                primary_ref,
                self.aux_ckpt_primary_tolerance,
            )
            return

        aux_auc = float(aux_metrics[self.aux_ckpt_name]["auc"])
        aux_score = aux_auc
        blend_score = (
            (1.0 - self.aux_ckpt_blend_weight) * float(val_auc)
            + self.aux_ckpt_blend_weight * aux_auc
        )

        if self.aux_best_score is None or aux_score > self.aux_best_score:
            ckpt_dir = self._save_aux_select_checkpoint(
                global_step=total_step,
                kind="aux",
                score=aux_score,
                val_auc=float(val_auc),
                aux_auc=aux_auc,
            )
            self.aux_best_checkpoint_dir = ckpt_dir
            self.aux_best_score = aux_score
            self.aux_best_primary_auc = float(val_auc)
            self.aux_best_aux_auc = aux_auc
            self.aux_best_global_step = int(total_step)

        if (
            self.blend_best_score is None
            or blend_score > self.blend_best_score
        ):
            ckpt_dir = self._save_aux_select_checkpoint(
                global_step=total_step,
                kind="blend",
                score=blend_score,
                val_auc=float(val_auc),
                aux_auc=aux_auc,
            )
            self.blend_best_checkpoint_dir = ckpt_dir
            self.blend_best_score = blend_score
            self.blend_best_primary_auc = float(val_auc)
            self.blend_best_aux_auc = aux_auc
            self.blend_best_global_step = int(total_step)

    def _remove_old_best_dirs(self) -> None:
        """Delete stale ``*.best_model`` directories so that only the latest
        best checkpoint is kept on disk.
        """
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _build_topk_dir_name(
        self,
        global_step: int,
        rank: int,
        val_auc: float,
    ) -> str:
        """Build a ``*.topk_{rank}`` checkpoint sub-directory name.

        Format mirrors ``_build_step_dir_name`` but terminates with
        ``.topk_{rank}`` instead of ``.best_model``. The ``.auc=``
        fragment is always included so platform inspection / offline
        selection can read the validation score straight from the name.
        """
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        parts.append(f"auc={val_auc:.6f}")
        return ".".join(parts) + f".topk_{rank}"

    def _maintain_topk_ckpts(
        self,
        global_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        """Maintain the Top-K (by val_auc DESC) checkpoint ring.

        Called once per validation event. No-op when ``keep_topk_ckpt
        == 1`` (the default, backward-compat path: global best is
        already handled by ``_handle_validation_result``).

        Behavior for K > 1:
          1. Decide whether this ckpt qualifies for the current Top-K.
             It does if either the ring is not yet full (<K entries) or
             this ``val_auc`` > current worst retained score.
          2. If qualifies:
             a. Materialize a snapshot of the current model into a
                temporary ``*.topk_pending`` directory (we don't yet
                know its final rank).
             b. Insert (val_auc, logloss, step, temp_dir) into
                ``self._topk_ckpts``, re-sort by val_auc DESC, then
                truncate to K. If the incoming entry is evicted (can
                happen if tie on AUC pushes it out), its temp dir is
                deleted.
             c. Rename all surviving entries to ``*.topk_{rank}``
                according to their new rank.
             d. Delete any stale ``*.topk_*`` directories on disk that
                don't correspond to a current top-K entry (cleanup).
          3. If not qualifies: no-op.

        This is SEPARATE from the ``*.best_model`` path maintained by
        ``_handle_validation_result``. The global-best ckpt has its own
        directory and is not counted among the top-K siblings — so if
        K=3, disk will carry at most 1 ``best_model`` + 3 ``topk_N``
        dirs = 4 checkpoint directories (but global-best often matches
        top-1 so the distinct model files typically total 3 for K=3).
        """
        if self.keep_topk_ckpt <= 1:
            return

        # Decide qualification using val_auc.
        ring_full = len(self._topk_ckpts) >= self.keep_topk_ckpt
        if ring_full:
            current_worst = min(self._topk_ckpts, key=lambda e: e[0])[0]
            if val_auc <= current_worst:
                return

        # Step a: write the snapshot. Use a unique temp name keyed by step
        # to avoid collisions if two validations trigger back-to-back.
        tmp_name = f"global_step{global_step}.topk_pending"
        tmp_dir = os.path.join(self.save_dir, tmp_name)
        os.makedirs(tmp_dir, exist_ok=True)
        torch.save(
            self._raw_model.state_dict(), os.path.join(tmp_dir, "model.pt"))
        self._write_sidecar_files(tmp_dir)

        # Step b: insert + re-sort + truncate + evict.
        self._topk_ckpts.append((val_auc, val_logloss, global_step, tmp_dir))
        self._topk_ckpts.sort(key=lambda e: (-e[0], e[2]))  # AUC DESC, step ASC tie-break
        if len(self._topk_ckpts) > self.keep_topk_ckpt:
            evicted = self._topk_ckpts[self.keep_topk_ckpt:]
            self._topk_ckpts = self._topk_ckpts[:self.keep_topk_ckpt]
            for _, _, _, evicted_dir in evicted:
                if os.path.isdir(evicted_dir):
                    shutil.rmtree(evicted_dir)
                    logging.info(
                        f"[topk] evicted lower-ranked ckpt: {evicted_dir}")

        # Step c: rename survivors to final *.topk_{rank} names. Two-phase
        # rename via intermediate .topk_renaming suffix to avoid collisions
        # when a new ckpt inherits an existing rank's directory name.
        staging: list[tuple[int, float, float, int, str]] = []  # (rank, auc, ll, step, staged_dir)
        for rank, (auc, ll, step, cur_dir) in enumerate(self._topk_ckpts, start=1):
            final_name = self._build_topk_dir_name(step, rank, auc)
            final_dir = os.path.join(self.save_dir, final_name)
            if cur_dir == final_dir:
                staging.append((rank, auc, ll, step, cur_dir))
                continue
            staged_name = f"{final_name}.renaming"
            staged_dir = os.path.join(self.save_dir, staged_name)
            if os.path.isdir(staged_dir):
                shutil.rmtree(staged_dir)
            os.rename(cur_dir, staged_dir)
            staging.append((rank, auc, ll, step, staged_dir))

        new_topk: list[tuple[float, float, int, str]] = []
        for rank, auc, ll, step, staged_dir in staging:
            final_name = self._build_topk_dir_name(step, rank, auc)
            final_dir = os.path.join(self.save_dir, final_name)
            if staged_dir != final_dir:
                if os.path.isdir(final_dir):
                    # Paranoid: a leftover from an interrupted rename.
                    shutil.rmtree(final_dir)
                os.rename(staged_dir, final_dir)
            new_topk.append((auc, ll, step, final_dir))
        self._topk_ckpts = new_topk

        # Step d: cleanup any stray *.topk_{rank} or *.topk_pending dirs
        # that aren't in our current ring.
        kept_dirs = {d for _, _, _, d in self._topk_ckpts}
        stray_pattern = os.path.join(
            self.save_dir, "global_step*.topk_*")
        for stray in glob.glob(stray_pattern):
            if stray not in kept_dirs and not stray.endswith('.renaming'):
                if os.path.isdir(stray):
                    shutil.rmtree(stray)
                    logging.info(f"[topk] removed stray dir: {stray}")

        logging.info(
            f"[topk] updated ring (K={self.keep_topk_ckpt}): "
            + ", ".join(
                f"rank{r}@step{s}:auc={a:.6f}"
                for r, (a, _, s, _) in enumerate(self._topk_ckpts, start=1)
            )
        )

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in ``batch`` to ``self.device`` (``non_blocking=True``,
        to cooperate with ``pin_memory``). Non-tensor values pass through.
        """
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        """Persist a new-best checkpoint atomically.

        Flow (ordered to avoid leaving empty sidecar-only directories on disk):

        1. Decide whether ``val_auc`` is *likely* to beat the current best
           using the same threshold as ``EarlyStopping._is_not_improved``,
           so our pre-cleanup and EarlyStopping's internal save decision
           stay in sync.
        2. If unlikely, short-circuit: do nothing on disk. We must NOT
           touch ``self.early_stopping.checkpoint_path`` or call
           ``_write_sidecar_files`` because the target directory may not
           exist yet (sidecar-only dirs would otherwise be created here,
           producing checkpoints with missing ``model.pt``).
        3. If likely, point ``EarlyStopping`` at the canonical
           ``global_stepN.best_model/model.pt`` path, remove any stale
           ``*.best_model`` dirs, then run ``EarlyStopping`` (which writes
           ``model.pt`` when it actually confirms a new best).
        4. Only after ``EarlyStopping`` has confirmed a new best
           (``best_score != old_best``) do we write the sidecar files into
           the freshly-created directory; this is guarded so that a
           razor-close score that tripped ``is_likely_new_best`` but not
           ``EarlyStopping``'s own gate does not create a stray dir.
        """
        old_best = self.early_stopping.best_score
        # Choose the metric that EarlyStopping optimizes (AUC or LogLoss).
        es_score = val_logloss if getattr(self.early_stopping, 'mode', 'max') == 'min' else val_auc
        if self.early_stopping.mode == 'max':
            is_likely_new_best = (
                old_best is None
                or es_score > old_best + self.early_stopping.delta
            )
        else:  # 'min'
            is_likely_new_best = (
                old_best is None
                or es_score < old_best - self.early_stopping.delta
            )
        if not is_likely_new_best:
            self.early_stopping(es_score, self._raw_model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            })
            return

        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(
                total_step, is_best=True, val_auc=val_auc),
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")

        self._remove_old_best_dirs()

        self.early_stopping(es_score, self._raw_model, {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
        })

        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self._save_step_checkpoint(
                total_step, is_best=True, val_auc=val_auc,
                skip_model_file=True)

    def train(self) -> None:
        """Main training loop: iterates over epochs, performs step-level and
        epoch-level validation, triggers EarlyStopping and the periodic sparse
        re-initialization strategy.
        """
        print("Start training (PCVRHyFormer)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                              dynamic_ncols=True)
            loss_sum = 0.0
            epoch_steps = 0

            for step, batch in train_pbar:
                loss = self._train_step(batch)
                total_step += 1
                epoch_steps += 1
                loss_sum += loss
                validated_this_step = False

                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, total_step)

                train_pbar.set_postfix({"loss": f"{loss:.4f}"})

                # Step-level validation (only when eval_every_n_steps > 0).
                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    logging.info(f"Evaluating at step {total_step}")
                    val_auc, val_logloss = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()

                    logging.info(f"Step {total_step} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, total_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                    previous_best = self.early_stopping.best_score
                    self._handle_validation_result(total_step, val_auc, val_logloss)
                    self._record_validation_result(
                        epoch, total_step, val_auc, val_logloss, previous_best)
                    validated_this_step = True

                    if self.early_stopping.early_stop:
                        reason_tag = (
                            "logloss_rebound" if self._logloss_rebound_counter
                            >= self.logloss_rebound_patience
                            and self.logloss_rebound_patience > 0
                            else "early_stop"
                        )
                        logging.info(
                            f"Early stopping at step {total_step} ({reason_tag})")
                        self._write_training_summary(
                            f"{reason_tag}_step_{total_step}")
                        return

                if self.max_train_steps > 0 and total_step >= self.max_train_steps:
                    avg_loss = loss_sum / max(1, epoch_steps)
                    logging.info(
                        f"Reached max_train_steps={self.max_train_steps} at "
                        f"epoch {epoch}, step {step + 1}; "
                        f"Average Loss: {avg_loss}")

                    if not validated_this_step:
                        val_auc, val_logloss = self.evaluate(epoch=epoch)
                        self.model.train()
                        torch.cuda.empty_cache()

                        logging.info(
                            f"Max-step Validation | AUC: {val_auc}, "
                            f"LogLoss: {val_logloss}")

                        if self.writer:
                            self.writer.add_scalar('AUC/valid', val_auc, total_step)
                            self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                        previous_best = self.early_stopping.best_score
                        self._handle_validation_result(
                            total_step, val_auc, val_logloss)
                        self._record_validation_result(
                            epoch, total_step, val_auc, val_logloss,
                            previous_best)

                    # Trigger sparse reinit once at proxy end so that proxy
                    # embedding dynamics match epoch-boundary full training.
                    # Without this, proxy runs never experience the cold-restart
                    # that fires at every epoch end in full training, making
                    # proxy→full signal extrapolation unreliable.
                    if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                        old_state: Dict[int, Any] = {}
                        for group in self.sparse_optimizer.param_groups:
                            for p in group['params']:
                                if p.data_ptr() in self.sparse_optimizer.state:
                                    old_state[p.data_ptr()] = self.sparse_optimizer.state[p]
                        reinit_ptrs = self._raw_model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                        sparse_params = self._raw_model.get_sparse_params()
                        self.sparse_optimizer = torch.optim.Adagrad(
                            sparse_params, lr=self.sparse_lr,
                            weight_decay=self.sparse_weight_decay)
                        restored = 0
                        for p in sparse_params:
                            if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                                self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                                restored += 1
                        # T27 · re-compile after sparse params change.
                        if self.enable_torch_compile:
                            self.model = torch.compile(self._raw_model, mode='default')
                        logging.info(
                            f"[proxy-reinit] Triggered sparse reinit at "
                            f"max_train_steps={self.max_train_steps} "
                            f"(mirrors full-train epoch-end reinit); "
                            f"restored {restored} low-cardinality optimizer states")

                    logging.info(
                        f"Stopping training after max_train_steps="
                        f"{self.max_train_steps}")
                    self._write_training_summary(
                        f"max_train_steps_{self.max_train_steps}")
                    return

            logging.info(
                f"Epoch {epoch}, Average Loss: "
                f"{loss_sum / max(1, epoch_steps)}")

            val_auc, val_logloss = self.evaluate(epoch=epoch)
            self.model.train()
            torch.cuda.empty_cache()

            logging.info(f"Epoch {epoch} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

            previous_best = self.early_stopping.best_score
            self._handle_validation_result(total_step, val_auc, val_logloss)
            self._record_validation_result(
                epoch, total_step, val_auc, val_logloss, previous_best)

            if self.early_stopping.early_stop:
                reason_tag = (
                    "logloss_rebound" if self._logloss_rebound_counter
                    >= self.logloss_rebound_patience
                    and self.logloss_rebound_patience > 0
                    else "early_stop"
                )
                logging.info(f"Early stopping at epoch {epoch} ({reason_tag})")
                self._write_training_summary(f"{reason_tag}_epoch_{epoch}")
                break

            # After the configured epoch, reinitialize high-cardinality sparse
            # params (Embeddings) as a form of cold restart to reduce overfit.
            # Reference: KuaiShou Tech., "MultiEpoch: Reusing Training Data
            # for Click-Through Rate Prediction",
            # https://arxiv.org/pdf/2305.19531
            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                # Snapshot Adagrad state per parameter via data_ptr, so state
                # of low-cardinality embeddings can be preserved across rebuild.
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self._raw_model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                sparse_params = self._raw_model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                # Restore optimizer state for low-cardinality embeddings only.
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                # T27 · re-compile after sparse params change.
                if self.enable_torch_compile:
                    self.model = torch.compile(self._raw_model, mode='default')
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")
        else:
            self._write_training_summary("max_epochs_completed")

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ``ModelInput`` NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
        )

    def _binary_loss(
        self,
        logits: torch.Tensor,
        label: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.loss_type == 'focal':
            loss_vec = sigmoid_focal_loss(
                logits, label, alpha=self.focal_alpha,
                gamma=self.focal_gamma, reduction='none')
        elif self.loss_type == 'auc':
            # CombinedAUCLoss returns a scalar directly (BCE + PairwiseBPR).
            # BPR is inherently pairwise and doesn't accept per-sample weights;
            # sample_weight is applied to the BCE component only.
            return combined_auc_loss(
                logits, label,
                auc_weight=self.auc_loss_weight,
                margin=self.auc_margin,
                max_pairs=self.auc_max_pairs,
                sample_weight=sample_weight,
            )
        else:
            loss_vec = F.binary_cross_entropy_with_logits(
                logits, label, reduction='none')
        if sample_weight is None:
            return loss_vec.mean()
        sample_weight = sample_weight.to(loss_vec.dtype)
        return (loss_vec * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)

    def _train_step(self, batch: Dict[str, Any]) -> float:
        """Run a single training step and return the scalar loss value."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()
        sample_weight = device_batch.get('sample_weight')
        if sample_weight is not None:
            sample_weight = sample_weight.float()

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)

        # T27 · optional AMP autocast for forward + loss. bfloat16 is
        # numerically stable enough for CVR BCE loss and doesn't need a
        # loss scaler. fp16 goes through GradScaler.
        if self.enable_amp:
            with torch.autocast(
                'cuda' if self.device.startswith('cuda') else 'cpu',
                dtype=self.amp_dtype,
            ):
                logits = self.model(model_input)

                if self.multi_task_loss:
                    if logits.ndim != 2 or logits.shape[1] < 2:
                        raise ValueError(
                            "multi_task_loss requires model action_num >= 2")
                    click_label = device_batch.get('label_click', label).float()
                    click_logits = logits[:, 0]
                    conversion_logits = logits[:, -1]
                    click_loss = F.binary_cross_entropy_with_logits(
                        click_logits, click_label)
                    conversion_loss = self._binary_loss(
                        conversion_logits, label, sample_weight)
                    loss = (
                        self.click_loss_weight * click_loss
                        + self.conversion_loss_weight * conversion_loss
                    )
                else:
                    if logits.ndim == 2 and logits.shape[1] > 1:
                        logits = logits[:, -1]
                    else:
                        logits = logits.squeeze(-1)
                    loss = self._binary_loss(logits, label, sample_weight)
        else:
            logits = self.model(model_input)  # (B, action_num)

            if self.multi_task_loss:
                if logits.ndim != 2 or logits.shape[1] < 2:
                    raise ValueError(
                        "multi_task_loss requires model action_num >= 2")
                click_label = device_batch.get('label_click', label).float()
                click_logits = logits[:, 0]
                conversion_logits = logits[:, -1]
                click_loss = F.binary_cross_entropy_with_logits(
                    click_logits, click_label)
                conversion_loss = self._binary_loss(
                    conversion_logits, label, sample_weight)
                loss = (
                    self.click_loss_weight * click_loss
                    + self.conversion_loss_weight * conversion_loss
                )
            else:
                if logits.ndim == 2 and logits.shape[1] > 1:
                    logits = logits[:, -1]
                else:
                    logits = logits.squeeze(-1)
                loss = self._binary_loss(logits, label, sample_weight)

        # Backward + optimizer step. Loss-scaling only when fp16.
        if self.amp_scaler is not None:
            self.amp_scaler.scale(loss).backward()
            self.amp_scaler.unscale_(self.dense_optimizer)
            torch.nn.utils.clip_grad_norm_(
                self._raw_model.parameters(), max_norm=1.0, foreach=False)
            self.amp_scaler.step(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                # Sparse optimizer doesn't need scaler (no fp16 gradients
                # on sparse embeddings), but we must step it only after
                # scaler updates.
                self.sparse_optimizer.step()
            self.amp_scaler.update()
        else:
            loss.backward()
            # foreach=False: avoids a PyTorch _foreach_norm CUDA kernel bug
            # observed with certain tensor shapes in this project.
            torch.nn.utils.clip_grad_norm_(
                self._raw_model.parameters(), max_norm=1.0, foreach=False)
            self.dense_optimizer.step()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.step()

        return loss.item()

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """Run validation over ``self.valid_loader`` and return ``(AUC, logloss)``.

        NaN predictions (which can arise from exploding gradients) are filtered
        out before computing both metrics.
        """
        return self._evaluate_on_loader(
            loader=self.valid_loader,
            split_tag='validation',
            run_slice_diag=True,
        )

    def evaluate_holdout(self) -> Optional[Tuple[float, float]]:
        """ADR-005: evaluate the detached holdout split if attached.

        Runs the same forward/NaN-filtering pipeline as ``evaluate`` but on
        ``self.holdout_loader``. Returns ``None`` when holdout is not
        configured, so callers can unconditionally call this at epoch end.
        Slice diagnostics are skipped to keep logs focused on the primary
        dev signal.
        """
        if self.holdout_loader is None:
            return None
        return self._evaluate_on_loader(
            loader=self.holdout_loader,
            split_tag='holdout',
            run_slice_diag=False,
        )

    def evaluate_aux_valid(
        self,
        name: str,
        loader: DataLoader,
    ) -> Tuple[float, float]:
        """Evaluate one read-only auxiliary validation loader."""
        return self._evaluate_on_loader(
            loader=loader,
            split_tag=f'aux_valid/{name}',
            run_slice_diag=False,
        )

    def _evaluate_on_loader(
        self,
        loader: DataLoader,
        split_tag: str,
        run_slice_diag: bool,
    ) -> Tuple[float, float]:
        print(f"Start Evaluation (PCVRHyFormer) - {split_tag}")
        # T27 · evaluation uses the un-compiled raw model. DECEM's
        # post explicitly recommends NOT applying torch.compile to
        # inference; evaluation is on the inference code path.
        self._raw_model.eval()

        pbar = tqdm(enumerate(loader), total=len(loader))

        all_logits_list = []
        all_labels_list = []
        # Per-row diagnostic side-channel (only populated if dataset emits the
        # _diag_* fields; safe to be empty otherwise).
        diag_keys = ('_diag_user_int_nz', '_diag_item_int_nz',
                     '_diag_seq_total_len', '_diag_label_type_raw')
        all_diag: Dict[str, list] = {k: [] for k in diag_keys}

        with torch.no_grad():
            for step, batch in pbar:
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())
                for k in diag_keys:
                    if k in batch:
                        all_diag[k].append(batch[k].numpy())

        if not all_logits_list:
            logging.warning(
                f"[Evaluate/{split_tag}] loader produced no batches; "
                "returning AUC=0.0 and LogLoss=inf")
            return 0.0, float('inf')

        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long()

        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(
                f"[Evaluate/{split_tag}] {n_nan}/{len(probs)} predictions "
                f"are NaN, filtering them out")
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(
                valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        if run_slice_diag and any(len(v) > 0 for v in all_diag.values()):
            try:
                diag_concat = {
                    k: np.concatenate(v)
                    for k, v in all_diag.items() if len(v) > 0
                }
                if nan_mask.any():
                    keep = ~nan_mask
                    diag_concat = {
                        k: arr[keep] for k, arr in diag_concat.items()}
                self._slice_diagnose(probs, labels_np, diag_concat)
            except Exception as e:
                logging.warning(f"[diag] slice diagnose failed: {e}")

        return auc, logloss

    def _slice_diagnose(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        diag: Dict[str, np.ndarray],
    ) -> None:
        """Log per-slice (n / pos_rate / AUC / pred_mean) tables for valid set.

        Slices:
        - ``user_int_nonzero`` (4-quantile): how many user_int features are
          non-padding. Detects how the model handles cold/sparse users.
        - ``item_int_nonzero`` (4-quantile): same on the item side.
        - ``seq_total_len`` (4-quantile): sum of seq lengths across all 4
          domains. Detects how the model handles low-activity users.
        - ``label_type`` (categorical {0,1,2}): per-class AUC and base rate.
          Sanity-checks that label_type==2 is the only positive class.

        AUC bucket-vs-overall delta > 0.02 typically indicates a slice the
        model is materially worse on, i.e. a feature-engineering target.
        """
        n_total = len(probs)
        if n_total == 0:
            return

        header = (f"  {'bucket':<28s} {'n':>10s} {'pos_rate':>10s} "
                  f"{'AUC':>8s} {'pred_mean':>10s}")

        def _row(name: str, mask: np.ndarray) -> str:
            n = int(mask.sum())
            if n == 0:
                return f"  {name:<28s} {n:>10d}"
            p, s = labels[mask], probs[mask]
            pos_rate = float(p.mean())
            pred_mean = float(s.mean())
            if len(np.unique(p)) < 2:
                auc_str = '     nan'
            else:
                auc_str = f"{roc_auc_score(p, s):>8.4f}"
            return (f"  {name:<28s} {n:>10d} {pos_rate:>10.4f} "
                    f"{auc_str} {pred_mean:>10.4f}")

        def _print_quantile(name: str, arr: np.ndarray, n_buckets: int = 4) -> None:
            qs = np.unique(np.quantile(arr, np.linspace(0, 1, n_buckets + 1)))
            if len(qs) < 2:
                logging.info(f"[diag] {name}: degenerate (single value {qs[0]}), skipped")
                return
            # searchsorted on the interior cut points; side='right' so that
            # the maximum value lands in the last bucket.
            bucket = np.searchsorted(qs[1:-1], arr, side='right')
            lines = [f"[diag] slice by {name} ({len(qs) - 1}-quantile):", header]
            for b in range(len(qs) - 1):
                mask = bucket == b
                label = f"[{int(qs[b])},{int(qs[b + 1])}]"
                lines.append(_row(label, mask))
            logging.info('\n'.join(lines))

        if '_diag_user_int_nz' in diag:
            _print_quantile('user_int_nonzero', diag['_diag_user_int_nz'])
        if '_diag_item_int_nz' in diag:
            _print_quantile('item_int_nonzero', diag['_diag_item_int_nz'])
        if '_diag_seq_total_len' in diag:
            _print_quantile('seq_total_len', diag['_diag_seq_total_len'])
        if '_diag_label_type_raw' in diag:
            arr = diag['_diag_label_type_raw']
            lines = ["[diag] slice by label_type (raw 0/1/2):", header]
            for lt in (0, 1, 2):
                lines.append(_row(f"label_type={lt}", arr == lt))
            logging.info('\n'.join(lines))

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single validation step and return ``(logits, labels)``."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        model_input = self._make_model_input(device_batch)
        # T27 · use raw (un-compiled) model for evaluation.
        logits, _ = self._raw_model.predict(model_input)  # (B, 1), (B, D)
        if logits.ndim == 2 and logits.shape[1] > 1:
            logits = logits[:, -1]
        else:
            logits = logits.squeeze(-1)  # (B,)

        return logits, label
