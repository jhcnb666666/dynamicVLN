#!/usr/bin/env bash
# 在完整 R2R 训练集上继续训练 500 步
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# 从之前的 quick experiment checkpoint 继续训练
RESUME_CKPT="${RESUME_CKPT:-checkpoints/tmp_quick_experiment_0512_164146}"
MODEL_PATH="${MODEL_PATH:-checkpoints/InternVLA-N1-System2}"

# 完整 R2R 训练集
TRAIN_ROOT="/home/ubuntu/dataset/VLN-Trajectory-Data/R2R"
VAL_ROOT="/home/ubuntu/project/StreamVLN/experiments_ext/fixed_eval_subsets/phase0_20260421_230201/teacher_forcing_240"

OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/tmp_full_r2r_continued_$(date +%m%d_%H%M%S)}"

MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-8000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-200}"
MAX_STEPS="${MAX_STEPS:-500}"
NUM_HISTORY="${NUM_HISTORY:-0}"
TRAIN_BS="${TRAIN_BS:-1}"
GRAD_ACC="${GRAD_ACC:-4}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
DTYPE="${DTYPE:-bf16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"

echo "========================================"
echo "完整 R2R 继续训练配置"
echo "========================================"
echo "基础模型:   ${MODEL_PATH}"
echo "恢复 checkpoint: ${RESUME_CKPT}"
echo "训练根目录: ${TRAIN_ROOT}"
echo "测试根目录: ${VAL_ROOT}"
echo "输出目录:   ${OUTPUT_DIR}"
echo "训练样本:   ${MAX_TRAIN_SAMPLES}"
echo "评估样本:   ${MAX_EVAL_SAMPLES}"
echo "训练步数:   ${MAX_STEPS}"
echo "历史帧数:   ${NUM_HISTORY}"
echo "LoRA r/α:   ${LORA_R}/${LORA_ALPHA}"
echo "========================================"

mkdir -p "$OUTPUT_DIR"

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
/home/ubuntu/.conda/envs/internnav/bin/python scripts/train/qwenvl_train/offline_r2r_multiframe_lora_sft_eval.py \
  --model_path "$MODEL_PATH" \
  --train_root "$TRAIN_ROOT" \
  --val_root "$VAL_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --resume_from_checkpoint "$RESUME_CKPT" \
  --max_train_samples "$MAX_TRAIN_SAMPLES" \
  --max_eval_samples "$MAX_EVAL_SAMPLES" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --max_steps "$MAX_STEPS" \
  --num_history "$NUM_HISTORY" \
  --per_device_train_batch_size "$TRAIN_BS" \
  --gradient_accumulation_steps "$GRAD_ACC" \
  --learning_rate "$LR" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --logging_steps 50 \
  --save_strategy no \
  --eval_strategy no \
  --report_to none \
  --attn_implementation "$ATTN_IMPL" \
  --dtype "$DTYPE" \
  --gradient_checkpointing \
  --run_generation_eval \
  --generation_eval_samples "$MAX_EVAL_SAMPLES" \
  --generation_max_new_tokens 6 \
  --dataloader_num_workers 0

echo "========================================"
echo "训练完成！结果保存在: ${OUTPUT_DIR}/offline_eval_metrics.json"
echo "========================================"
cat "${OUTPUT_DIR}/offline_eval_metrics.json"
