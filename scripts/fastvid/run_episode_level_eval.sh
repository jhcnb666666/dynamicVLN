#!/usr/bin/env bash
# Episode-level VLN 评估
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-checkpoints/InternVLA-N1-System2}"
VAL_ROOT="/home/ubuntu/project/StreamVLN/experiments_ext/fixed_eval_subsets/phase0_20260421_230201/teacher_forcing_240"
NUM_HISTORY="${NUM_HISTORY:-0}"
MAX_EPISODES="${MAX_EPISODES:-}"  # 空表示全部 240 个 episode

echo "========================================"
echo "Episode-level VLN 评估"
echo "========================================"

# 1. Baseline
echo ""
echo ">>> 评估 Baseline（无 LoRA）..."
BASELINE_OUTPUT="checkpoints/tmp_episode_baseline_$(date +%m%d_%H%M%S)"
/home/ubuntu/.conda/envs/internnav/bin/python scripts/fastvid/run_episode_level_eval.py \
  --model_path "$MODEL_PATH" \
  --val_root "$VAL_ROOT" \
  --output_dir "$BASELINE_OUTPUT" \
  --num_history "$NUM_HISTORY" \
  ${MAX_EPISODES:+--max_episodes "$MAX_EPISODES"} \
  --attn_implementation sdpa \
  --dtype bf16

# 2. LoRA 微调模型
echo ""
echo ">>> 评估 LoRA 微调模型..."
LORA_CKPT="${LORA_CKPT:-checkpoints/tmp_quick_experiment_0512_164146}"
LORA_OUTPUT="checkpoints/tmp_episode_lora_$(date +%m%d_%H%M%S)"
/home/ubuntu/.conda/envs/internnav/bin/python scripts/fastvid/run_episode_level_eval.py \
  --model_path "$MODEL_PATH" \
  --lora_checkpoint "$LORA_CKPT" \
  --val_root "$VAL_ROOT" \
  --output_dir "$LORA_OUTPUT" \
  --num_history "$NUM_HISTORY" \
  ${MAX_EPISODES:+--max_episodes "$MAX_EPISODES"} \
  --attn_implementation sdpa \
  --dtype bf16

echo ""
echo "========================================"
echo "评估完成！"
echo "Baseline 结果: ${BASELINE_OUTPUT}/episode_eval_metrics.json"
echo "LoRA 结果:     ${LORA_OUTPUT}/episode_eval_metrics.json"
echo "========================================"

echo ""
echo "【Baseline 汇总】"
cat "${BASELINE_OUTPUT}/episode_eval_metrics.json" | python3 -c "import sys, json; d=json.load(sys.stdin)['summary']; print(json.dumps(d, indent=2))"

echo ""
echo "【LoRA 汇总】"
cat "${LORA_OUTPUT}/episode_eval_metrics.json" | python3 -c "import sys, json; d=json.load(sys.stdin)['summary']; print(json.dumps(d, indent=2))"
