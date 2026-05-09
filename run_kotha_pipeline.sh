#!/bin/bash
# Kotha-1 Fine-Tuning Pipeline
# Runs Phase 2 → 3 → 4 sequentially on A100
# Expected total time: ~5-6 hours
# Expected cost: ~$8 at $1.49/hr
#
# Usage: bash run_kotha_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="${PROJ_DIR:-$SCRIPT_DIR}"
cd "$PROJ_DIR"

# Use python3 (python not always symlinked)
alias python=python3
PYTHON=python3

BASE_MODEL="${BASE_MODEL:-training/checkpoints/step-4000}"
CPT_DATA="${CPT_DATA:-finetune/data/cpt_textbook.jsonl}"
SFT_TRAIN_DATA="${SFT_TRAIN_DATA:-finetune/data/kotha_final_balanced_train.jsonl}"
SFT_VAL_DATA="${SFT_VAL_DATA:-finetune/data/kotha_final_balanced_val.jsonl}"

echo "=========================================="
echo " Kotha-1 Fine-Tuning Pipeline"
echo " $(date)"
echo "=========================================="

# Verify files exist
echo ""
echo "[CHECK] Verifying required files..."
for f in \
    "$BASE_MODEL/model.safetensors" \
    "$BASE_MODEL/tokenizer.json" \
    "$CPT_DATA" \
    "$SFT_TRAIN_DATA" \
    "$SFT_VAL_DATA" \
    finetune/phase2_continued_pretrain.py \
    finetune/phase3_full_sft.py; do
    if [ ! -f "$f" ]; then
        echo "MISSING: $f"
        exit 1
    fi
    echo "  OK: $f"
done
echo "[CHECK] All files present."

# Check GPU
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ==========================================
# PHASE 2: Continued Pre-Training
# ==========================================
echo "=========================================="
echo " PHASE 2: Continued Pre-Training"
echo " Start: $(date)"
echo "=========================================="

$PYTHON finetune/phase2_continued_pretrain.py \
    --base-model "$BASE_MODEL" \
    --data "$CPT_DATA" \
    --output checkpoints/kotha-cpt \
    --epochs 2 \
    --lr 1e-5 \
    --batch-size 8 \
    --grad-accum 16 \
    --max-seq-len 2048

echo ""
echo "[PHASE 2 DONE] $(date)"
echo "Checkpoint: checkpoints/kotha-cpt/"
echo ""

# ==========================================
# PHASE 3: Full SFT
# ==========================================
echo "=========================================="
echo " PHASE 3: Full SFT (curated dataset)"
echo " Start: $(date)"
echo "=========================================="

$PYTHON finetune/phase3_full_sft.py \
    --base-model checkpoints/kotha-cpt \
    --train-data "$SFT_TRAIN_DATA" \
    --val-data "$SFT_VAL_DATA" \
    --output checkpoints/kotha-1-sft \
    --epochs 2 \
    --lr 2e-5 \
    --batch-size 4 \
    --grad-accum 8 \
    --max-seq-len 2048

echo ""
echo "[PHASE 3 DONE] $(date)"
echo "Checkpoint: checkpoints/kotha-1-sft/"
echo ""

# ==========================================
# PHASE 4: News LoRA (skip if no news data)
# ==========================================
if [ -f finetune/data/kotha_news_train.jsonl ]; then
    echo "=========================================="
    echo " PHASE 4: News LoRA"
    echo " Start: $(date)"
    echo "=========================================="

    $PYTHON finetune/phase4_news_lora.py \
        --base-model checkpoints/kotha-1-sft \
        --data finetune/data/kotha_news_train.jsonl \
        --output checkpoints/kotha-1-news-lora \
        --epochs 3 \
        --lr 2e-5 \
        --batch-size 4 \
        --grad-accum 4

    echo ""
    echo "[PHASE 4 DONE] $(date)"
else
    echo "[PHASE 4 SKIP] No news data at finetune/data/kotha_news_train.jsonl"
fi

# ==========================================
# DONE
# ==========================================
echo ""
echo "=========================================="
echo " PIPELINE COMPLETE"
echo " $(date)"
echo "=========================================="
echo ""
echo "Checkpoints:"
ls -lhd checkpoints/*/
echo ""
echo "NEXT: Download checkpoints and STOP this instance!"
echo "  rsync -avz -e 'ssh -i ~/.brev/brev.pem' shadeform@\$(hostname -I | awk '{print \$1}'):/home/shadeform/bangla-llm/checkpoints/ ./checkpoints/"
