#!/usr/bin/env python3
"""Quick experiment: train a minimal LoRA (or reuse existing) and evaluate
action accuracy under 5 different iterative waypoint-filtering thresholds."""

import argparse
import glob
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLConfig,
    set_seed,
)
from peft import PeftModel

sys.path.insert(0, "/home/ubuntu/project/InternNav-lora")


def patch_torch_from_numpy() -> bool:
    """Patch torch.from_numpy for environments where ndarray conversion is broken."""
    original_from_numpy = torch.from_numpy

    def safe_from_numpy(array):
        try:
            return original_from_numpy(array)
        except TypeError as exc:
            if isinstance(array, np.ndarray) and "expected np.ndarray" in str(exc):
                return torch.tensor(array)
            raise

    torch.from_numpy = safe_from_numpy
    try:
        _ = torch.from_numpy(np.array([1, 2, 3]))
        return True
    except Exception:
        return False


patched = patch_torch_from_numpy()
if patched:
    print("Patched torch.from_numpy with a safe fallback for this environment.")

# Direct-load geometry_utils to avoid uvicorn etc.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "geometry_utils", "/home/ubuntu/project/InternNav-lora/internnav/utils/geometry_utils.py"
)
geometry_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry_utils)
filter_waypoints_iterative_by_cosine_distance = geometry_utils.filter_waypoints_iterative_by_cosine_distance

def trajectory_to_discrete_actions_close_to_goal(trajectory, step_size=0.25, turn_angle_deg=15, lookahead=4):
    """Convert a trajectory to discrete actions (0=stop, 1=forward, 2=left, 3=right)."""
    actions = []
    yaw = 0.0
    pos = trajectory[0]
    turn_angle_rad = np.deg2rad(turn_angle_deg)
    goal = trajectory[-1]

    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    while np.linalg.norm(pos - goal) > 0.2:
        dists = np.linalg.norm(trajectory - pos, axis=1)
        nearest_idx = np.argmin(dists)
        target_idx = min(nearest_idx + lookahead, len(trajectory) - 1)
        target = trajectory[target_idx]
        target_dir = target - pos
        if np.linalg.norm(target_dir) < 1e-6:
            break
        target_yaw = np.arctan2(target_dir[1], target_dir[0])
        delta_yaw = normalize_angle(target_yaw - yaw)
        n_turns = int(round(delta_yaw / turn_angle_rad))
        if n_turns > 0:
            actions += [2] * n_turns
        elif n_turns < 0:
            actions += [3] * (-n_turns)
        yaw = normalize_angle(yaw + n_turns * turn_angle_rad)

        next_pos = pos + step_size * np.array([np.cos(yaw), np.sin(yaw)])
        if np.linalg.norm(next_pos - goal) > np.linalg.norm(pos - goal):
            break
        actions.append(1)
        pos = next_pos

    return actions


ACTION_MAP = {0: "stop", 1: "forward", 2: "left", 3: "right"}
ALLOWED_ACTIONS = tuple(ACTION_MAP.values())


def frame_sort_key(frame_path: str):
    stem = Path(frame_path).stem
    if stem.isdigit():
        return int(stem)
    return stem


def list_rgb_frames(rgb_dir: str) -> List[str]:
    if not os.path.isdir(rgb_dir):
        return []
    frames = [os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    frames.sort(key=frame_sort_key)
    return frames


def normalize_instruction(instructions) -> str:
    if isinstance(instructions, str):
        return instructions.strip()
    if isinstance(instructions, Sequence) and instructions:
        text = instructions[0]
        if isinstance(text, str):
            return text.strip()
    return "Navigate to the target location."


def build_samples(data_root: str, max_samples: Optional[int], seed: int):
    annotation_path = os.path.join(data_root, "annotations.json")
    with open(annotation_path, "r", encoding="utf-8") as f:
        episodes = json.load(f)
    random.Random(seed).shuffle(episodes)
    samples = []
    for ep in episodes:
        actions = ep.get("actions", [])
        if len(actions) <= 1:
            continue
        video_rel = ep.get("video", "")
        rgb_dir = os.path.join(data_root, video_rel, "rgb")
        frames = list_rgb_frames(rgb_dir)
        if not frames:
            continue
        instruction = normalize_instruction(ep.get("instructions") or ep.get("instruction", ""))
        usable_steps = min(len(frames), len(actions) - 1)
        for step in range(usable_steps):
            action_id = actions[step + 1]
            action_text = ACTION_MAP.get(action_id)
            if action_text is None:
                continue
            samples.append({"frame_path": frames[step], "instruction": instruction, "action": action_text, "action_id": action_id})
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    return samples


def build_user_prompt(instruction: str) -> str:
    return (
        "You are an autonomous navigation assistant. "
        f"Instruction: {instruction}\n"
        "Look at the current RGB observation and predict the next action. "
        "Reply with exactly one word from: forward, left, right, stop."
    )


def parse_action_from_text(text: str) -> str:
    text = text.strip().lower()
    for action in ALLOWED_ACTIONS:
        if re.search(rf"\b{action}\b", text):
            return action
    startswith_map = {"f": "forward", "l": "left", "r": "right", "s": "stop"}
    if text:
        return startswith_map.get(text[0], "unknown")
    return "unknown"


@torch.no_grad()
def run_generation_eval(model, processor, eval_samples, max_samples, max_new_tokens):
    if not eval_samples:
        return {"generation_eval_samples": 0, "generation_action_acc": 0.0}
    model.eval()
    subset = eval_samples[:max_samples]
    correct = 0
    total = 0
    for sample in subset:
        image = Image.open(sample["frame_path"]).convert("RGB")
        prompt_messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": build_user_prompt(sample["instruction"])}],
        }]
        prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt_text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, temperature=1.0, top_p=1.0, top_k=50, use_cache=True)
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        pred_action = parse_action_from_text(generated_text)
        if pred_action == sample["action"]:
            correct += 1
        total += 1
    accuracy = correct / max(total, 1)
    return {"generation_eval_samples": float(total), "generation_action_acc": accuracy}


def load_episode_waypoints_and_actions(data_root: str, max_episodes: Optional[int], seed: int):
    """Load episodes with waypoints (from pose) and ground-truth actions."""
    annotation_path = os.path.join(data_root, "annotations.json")
    with open(annotation_path, "r", encoding="utf-8") as f:
        episodes = json.load(f)
    random.Random(seed).shuffle(episodes)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    results = []
    for ep in episodes:
        video_rel = ep.get("video", "")
        pose_dir = os.path.join(data_root, video_rel, "pose")
        if not os.path.isdir(pose_dir):
            continue
        pose_files = sorted(glob.glob(os.path.join(pose_dir, "*.npy")))
        if len(pose_files) < 3:
            continue

        waypoints = []
        for pf in pose_files:
            mat = np.load(pf)
            if mat.shape == (4, 4):
                waypoints.append([mat[0, 3], mat[2, 3]])
            else:
                waypoints.append([0.0, 0.0])
        waypoints = np.array(waypoints, dtype=np.float32)

        actions = ep.get("actions", [])
        if len(actions) <= 1:
            continue

        results.append({
            "episode": ep.get("id", video_rel),
            "waypoints": waypoints,
            "actions": actions[1:],  # skip the first -1 placeholder
        })
    return results


def evaluate_waypoint_thresholds(episodes: List[dict], thresholds: List[float]):
    """For each threshold, filter waypoints iteratively and convert to discrete actions."""
    stats = []
    for thr in thresholds:
        total_actions_pred = 0
        total_actions_correct = 0
        total_waypoints_original = 0
        total_waypoints_filtered = 0

        for ep in episodes:
            wps = ep["waypoints"]
            gt_actions = ep["actions"]

            filtered, final_avg, remaining = filter_waypoints_iterative_by_cosine_distance(wps, avg_threshold=thr)

            # Convert filtered trajectory to discrete actions
            pred_actions = trajectory_to_discrete_actions_close_to_goal(filtered)

            # Align lengths for comparison (use minimum length)
            compare_len = min(len(pred_actions), len(gt_actions))
            if compare_len > 0:
                matches = sum(1 for i in range(compare_len) if pred_actions[i] == gt_actions[i])
                total_actions_correct += matches
                total_actions_pred += compare_len

            total_waypoints_original += len(wps)
            total_waypoints_filtered += remaining

        accuracy = total_actions_correct / max(total_actions_pred, 1)
        retain_ratio = total_waypoints_filtered / max(total_waypoints_original, 1)
        stats.append({
            "threshold": thr,
            "action_accuracy": accuracy,
            "action_compare_samples": total_actions_pred,
            "action_matches": total_actions_correct,
            "waypoints_original": total_waypoints_original,
            "waypoints_filtered": total_waypoints_filtered,
            "waypoints_retain_ratio": retain_ratio,
        })
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="/home/ubuntu/project/InternNav-lora/checkpoints/InternVLA-N1-System2")
    parser.add_argument("--lora_checkpoint", type=str, default="/home/ubuntu/project/InternNav-lora/checkpoints/tmp_offline_lora_1000steps_v5/checkpoint-600")
    parser.add_argument("--val_root", type=str, default="/home/ubuntu/dataset/VLN-Trajectory-Data/R2R/offline_r2r_val_unseen")
    parser.add_argument("--max_eval_samples", type=int, default=256)
    parser.add_argument("--max_episodes", type=int, default=200)
    parser.add_argument("--generation_max_new_tokens", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # 1. Model generation eval (baseline)
    print("=" * 60)
    print("Step 1: Running model generation eval (baseline)")
    print("=" * 60)

    config_path = os.path.join(args.base_model_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    config = Qwen2_5_VLConfig.from_dict(raw_cfg)

    print("Loading base model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.lora_checkpoint)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.base_model_path)

    eval_samples = build_samples(args.val_root, args.max_eval_samples, seed=args.seed + 1)
    gen_metrics = run_generation_eval(
        model=model,
        processor=processor,
        eval_samples=eval_samples,
        max_samples=args.max_eval_samples,
        max_new_tokens=args.generation_max_new_tokens,
    )
    print(f"Baseline generation action accuracy: {gen_metrics['generation_action_acc']:.4f}")

    # 2. Waypoint filtering experiment
    print("\n" + "=" * 60)
    print("Step 2: Evaluating waypoint filtering thresholds")
    print("=" * 60)

    episodes = load_episode_waypoints_and_actions(args.val_root, max_episodes=args.max_episodes, seed=args.seed + 1)
    print(f"Loaded {len(episodes)} episodes with pose+actions for waypoint experiment")

    # 5 threshold combinations aligned with user's config matrix
    # We evaluate both stable and unstable thresholds
    thresholds = [1.50, 1.25, 1.00, 0.75, 0.50]
    threshold_stats = evaluate_waypoint_thresholds(episodes, thresholds)

    # 3. Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nBaseline (model generation action accuracy): {gen_metrics['generation_action_acc']:.4f}")
    print(f"Baseline eval samples: {int(gen_metrics['generation_eval_samples'])}")

    print("\n" + "-" * 60)
    print(f"{'Config':<20} {'Threshold':<12} {'WP Retain%':<12} {'Action Acc':<12} {'Compare Steps'}")
    print("-" * 60)

    config_names = {
        1.50: "Ultra-Aggressive",
        1.25: "Aggressive",
        1.00: "Balanced (default)",
        0.75: "Conservative",
        0.50: "Ultra-Conservative",
    }

    for stat in threshold_stats:
        thr = stat["threshold"]
        name = config_names.get(thr, "Custom")
        print(
            f"{name:<20} {thr:<12.2f} {stat['waypoints_retain_ratio']:<12.3f} "
            f"{stat['action_accuracy']:<12.4f} {stat['action_compare_samples']}"
        )

    # Save detailed results
    out_path = "/tmp/quick_threshold_experiment_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "baseline": gen_metrics,
            "threshold_experiments": threshold_stats,
        }, f, indent=2)
    print(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
