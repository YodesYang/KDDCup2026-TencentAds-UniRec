#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${1:-${REPO_DIR}/data}"

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

python -u "${REPO_DIR}/src/train.py" \
  --data_dir "${DATA_DIR}" \
  --ckpt_dir "${REPO_DIR}/checkpoints/reference" \
  --log_dir "${REPO_DIR}/logs/reference" \
  --seed 1 \
  --valid_split_strategy time \
  --valid_ratio 0.05 \
  --num_epochs 2 \
  --batch_size 128 \
  --lr 0.0001 \
  --sparse_lr 0.05 \
  --action_num 2 \
  --multi_task_loss \
  --click_loss_weight 1.0 \
  --conversion_loss_weight 1.0 \
  --ns_tokenizer_type rankmixer \
  --user_ns_tokens 4 \
  --item_ns_tokens 2 \
  --num_queries 1 \
  --ns_groups_json "${REPO_DIR}/src/ns_groups_vocab_tier.json" \
  --emb_skip_threshold 1000000 \
  --seq_encoder_type swiglu \
  --seq_hash_vocab 500000 \
  --enable_per_domain_buckets \
  --seq_max_lens "seq_a:128,seq_b:128,seq_c:256,seq_d:256" \
  --keep_topk_ckpt 2 \
  --enable_amp \
  --amp_dtype bfloat16 \
  --num_workers 8
