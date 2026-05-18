#!/usr/bin/env python3
"""Episode-level VLN evaluation.

Rolls out the model on full episodes: starting from frame 0, the model predicts
an action, we move to the next frame, and repeat until 'stop' or max steps.
This is then compared against the GT action sequence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from peft import PeftModel

from scripts.train.qwenvl_train.offline_r2r_multiframe_lora_sft_eval import (
    build_user_prompt,
    parse_action_from_text,
    patch_torch_from_numpy,
)

ACTION_MAP = {0: "stop", 1: "forward", 2: "left", 3: "right"}
REVERSE_ACTION_MAP = {v: k for k, v in ACTION_MAP.items()}


def list_rgb_frames(rgb_dir: str) -> List[str]:
    if not os.path.isdir(rgb_dir):
        return []
    frames = sorted([
        os.path.join(rgb_dir, f)
        for f in os.listdir(rgb_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    return frames


def load_episodes(data_root: str, max_episodes: Optional[int]) -> List[Dict]:
    annotation_path = os.path.join(data_root, "annotations.json")
    with open(annotation_path, "r", encoding="utf-8") as f:
        episodes = json.load(f)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    return episodes


def normalize_instruction(instructions) -> str:
    if isinstance(instructions, str):
        return instructions.strip()
    if isinstance(instructions, (list, tuple)) and instructions:
        text = instructions[0]
        if isinstance(text, str):
            return text.strip()
    return "Navigate to the target location."


@torch.no_grad()
def predict_action(model, processor, image_path: str, instruction: str, num_history: int, history_frame_paths: List[str]) -> str:
    images = [Image.open(p).convert("RGB") for p in history_frame_paths]
    images.append(Image.open(image_path).convert("RGB"))

    prompt_messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": build_user_prompt(instruction, num_history)}],
        }
    ]

    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=images, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=6,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        use_cache=True,
    )

    generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return parse_action_from_text(generated_text)


def evaluate_episode(model, processor, episode: Dict, num_history: int) -> Dict:
    video_rel = episode.get("video", "")
    rgb_dir = os.path.join(args.val_root, video_rel, "rgb")
    frames = list_rgb_frames(rgb_dir)
    actions_gt = episode.get("actions", [])
    instruction = normalize_instruction(episode.get("instructions") or episode.get("instruction", ""))

    # actions[0] is -1 (initial state), actions[1] is first action for frame 0 -> 1
    # We predict action based on frame[i], then move to frame[i+1]
    usable_steps = min(len(frames), len(actions_gt) - 1)

    pred_actions = []
    history_frame_paths = []

    for step in range(usable_steps):
        current_frame = frames[step]
        action_text = predict_action(model, processor, current_frame, instruction, num_history, history_frame_paths)
        pred_actions.append(action_text)

        # Update history for next step
        if num_history > 0:
            history_frame_paths.append(current_frame)
            if len(history_frame_paths) > num_history:
                history_frame_paths.pop(0)

    gt_actions = [ACTION_MAP.get(a, "unknown") for a in actions_gt[1:usable_steps + 1]]

    # Compute metrics
    correct = sum(1 for p, g in zip(pred_actions, gt_actions) if p == g)
    action_acc = correct / max(len(gt_actions), 1)

    # Trajectory-level: exact match of full action sequence
    exact_match = pred_actions == gt_actions

    # Stop position: where did the model stop (if at all)
    pred_stop_idx = None
    for i, a in enumerate(pred_actions):
        if a == "stop":
            pred_stop_idx = i
            break

    gt_stop_idx = None
    for i, a in enumerate(gt_actions):
        if a == "stop":
            gt_stop_idx = i
            break

    stop_correct = pred_stop_idx == gt_stop_idx

    return {
        "episode_id": episode.get("id"),
        "num_steps": usable_steps,
        "pred_actions": pred_actions,
        "gt_actions": gt_actions,
        "action_acc": action_acc,
        "exact_match": exact_match,
        "pred_stop_idx": pred_stop_idx,
        "gt_stop_idx": gt_stop_idx,
        "stop_correct": stop_correct,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Episode-level VLN evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Optional LoRA checkpoint to load")
    parser.add_argument("--val_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--num_history", type=int, default=0)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def main():
    global args
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    patch_torch_from_numpy()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    processor = AutoProcessor.from_pretrained(args.model_path)
    print(f"Loading base model from: {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
    )

    if args.lora_checkpoint:
        print(f"Loading LoRA adapter from: {args.lora_checkpoint}")
        model = PeftModel.from_pretrained(model, args.lora_checkpoint)

    model.eval().cuda()

    episodes = load_episodes(args.val_root, args.max_episodes)
    print(f"Evaluating {len(episodes)} episodes...")

    results = []
    total_action_acc = 0
    exact_matches = 0
    stop_corrects = 0

    for i, ep in enumerate(episodes):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(episodes)}")
        res = evaluate_episode(model, processor, ep, args.num_history)
        results.append(res)
        total_action_acc += res["action_acc"]
        if res["exact_match"]:
            exact_matches += 1
        if res["stop_correct"]:
            stop_corrects += 1

    n = len(episodes)
    summary = {
        "model": args.model_path,
        "lora_checkpoint": args.lora_checkpoint,
        "num_episodes": n,
        "num_history": args.num_history,
        "avg_action_acc": total_action_acc / max(n, 1),
        "exact_match_rate": exact_matches / max(n, 1),
        "stop_correct_rate": stop_corrects / max(n, 1),
    }

    print("\n========== Episode-Level VLN Eval Results ==========")
    print(json.dumps(summary, indent=2))

    metrics_path = os.path.join(args.output_dir, "episode_eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2)
    print(f"Saved detailed results to: {metrics_path}")


if __name__ == "__main__":
    main()
