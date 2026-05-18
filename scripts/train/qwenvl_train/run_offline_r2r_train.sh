#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-checkpoints/InternVLA-N1-System2}"
TRAIN_ROOT="${TRAIN_ROOT:-/home/ubuntu/dataset/VLN-Trajectory-Data/R2R}"
VAL_ROOT="${VAL_ROOT:-/home/ubuntu/dataset/VLN-Trajectory-Data/R2R/offline_r2r_val_unseen}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/offline_r2r_lora_run1}"

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
conda run -n internnav python scripts/train/qwenvl_train/offline_r2r_lora_sft_eval.py \
  --model_path "$MODEL_PATH" \
  --train_root "$TRAIN_ROOT" \
  --val_root "$VAL_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --max_train_samples "${MAX_TRAIN_SAMPLES:-8000}" \
  --max_eval_samples "${MAX_EVAL_SAMPLES:-1200}" \
  --max_seq_length "${MAX_SEQ_LENGTH:-1024}" \
  --num_train_epochs "${NUM_EPOCHS:-1}" \
  --max_steps "${MAX_STEPS:--1}" \
  --per_device_train_batch_size "${TRAIN_BS:-1}" \
  --per_device_eval_batch_size "${EVAL_BS:-1}" \
  --gradient_accumulation_steps "${GRAD_ACC:-8}" \
  --learning_rate "${LR:-2e-4}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --warmup_ratio "${WARMUP_RATIO:-0.03}" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --save_steps "${SAVE_STEPS:-200}" \
  --eval_steps "${EVAL_STEPS:-100}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
  --eval_strategy "${EVAL_STRATEGY:-steps}" \
  --dataloader_num_workers "${NUM_WORKERS:-0}" \
  --attn_implementation "${ATTN_IMPL:-sdpa}" \
  --dtype "${DTYPE:-bf16}" \
  --lora_r "${LORA_R:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --run_generation_eval \
  --generation_eval_samples "${GEN_EVAL_SAMPLES:-200}" \
  --generation_max_new_tokens "${GEN_MAX_NEW_TOKENS:-6}"
