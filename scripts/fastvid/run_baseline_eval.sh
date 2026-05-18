#!/usr/bin/env bash
# Baseline 评估脚本：不训练，直接加载原始 checkpoint 在测试集上评估
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-checkpoints/InternVLA-N1-System2}"
VAL_ROOT="/home/ubuntu/project/StreamVLN/experiments_ext/fixed_eval_subsets/phase0_20260421_230201/teacher_forcing_240"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/tmp_baseline_eval_$(date +%m%d_%H%M%S)}"

MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-200}"
NUM_HISTORY="${NUM_HISTORY:-0}"
DTYPE="${DTYPE:-bf16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"

echo "========================================"
echo "Baseline 评估配置"
echo "========================================"
echo "模型路径:   ${MODEL_PATH}"
echo "测试根目录: ${VAL_ROOT}"
echo "输出目录:   ${OUTPUT_DIR}"
echo "评估样本:   ${MAX_EVAL_SAMPLES}"
echo "历史帧数:   ${NUM_HISTORY}"
echo "========================================"

mkdir -p "$OUTPUT_DIR"

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
/home/ubuntu/.conda/envs/internnav/bin/python scripts/fastvid/run_baseline_eval.py \
  --model_path "$MODEL_PATH" \
  --val_root "$VAL_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --max_eval_samples "$MAX_EVAL_SAMPLES" \
  --num_history "$NUM_HISTORY" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --attn_implementation "$ATTN_IMPL" \
  --dtype "$DTYPE" \
  --generation_eval_samples "$MAX_EVAL_SAMPLES" \
  --generation_max_new_tokens 6

echo "========================================"
echo "Baseline 评估完成！"
echo "========================================"
cat "${OUTPUT_DIR}/baseline_eval_metrics.json"
