#!/usr/bin/env bash
set -euo pipefail

# Continue pre-training from the strongest available base checkpoint using
# saner warmup defaults and a held-out eval slice.

BASE_CKPT="${BASE_CKPT:-training/checkpoints/step-4000}"
TOKENS="${TOKENS:-training/data/train_tokens.bin}"
OUTPUT_DIR="${OUTPUT_DIR:-training/checkpoints/base-repair-v1}"

EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
SEQ_LEN="${SEQ_LEN:-2048}"
LR="${LR:-1.5e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
EVAL_RATIO="${EVAL_RATIO:-0.01}"
EVAL_EVERY="${EVAL_EVERY:-250}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-32}"

exec accelerate launch training/train.py \
  --tokens "$TOKENS" \
  --output-dir "$OUTPUT_DIR" \
  --resume "$BASE_CKPT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --seq-len "$SEQ_LEN" \
  --lr "$LR" \
  --warmup-ratio "$WARMUP_RATIO" \
  --save-every "$SAVE_EVERY" \
  --eval-ratio "$EVAL_RATIO" \
  --eval-every "$EVAL_EVERY" \
  --eval-max-batches "$EVAL_MAX_BATCHES" \
  "$@"
