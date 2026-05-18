#!/usr/bin/env python3
"""Standalone eval script: loads a trained LoRA checkpoint and computes
action generation accuracy + waypoint filtering stats with a given threshold."""

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
)

sys.path.insert(0, "/home/ubuntu/project/InternNav-lora")

from peft import PeftModel

ACTION_MAP = {
    0: "stop",
    1: "forward",
    2: "left",
    3: "right",
}
ALLOWED_ACTIONS = tuple(ACTION_MAP.values())


def frame_sort_key(frame_path: str):
    stem = Path(frame_path).stem
    if stem.isdigit():
        return int(stem)
    return stem


def list_rgb_frames(rgb_dir: str) -> List[str]:
    if not os.path.isdir(rgb_dir):
        return []
    frames = [
        os.path.join(rgb_dir, f)
        for f in os.listdir(rgb_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
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
            samples.append({
                "frame_path": frames[step],
                "instruction": instruction,
                "action": action_text,
            })
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
        inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
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


def compute_waypoint_filter_stats(val_root: str, max_eval_samples: int, threshold: float, seed: int):
    annotation_path = os.path.join(val_root, "annotations.json")
    with open(annotation_path, "r", encoding="utf-8") as f:
        episodes = json.load(f)
    random.Random(seed).shuffle(episodes)
    if max_eval_samples is not None:
        episodes = episodes[:max_eval_samples]

    from internnav.utils.geometry_utils import filter_waypoints_by_cosine_distance

    total_original = 0
    total_filtered = 0
    per_episode_deleted = []

    for ep in episodes:
        video_rel = ep.get("video", "")
        pose_dir = os.path.join(val_root, video_rel, "pose")
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
        waypoints_np = np.array(waypoints, dtype=np.float32)
        filtered_np = filter_waypoints_by_cosine_distance(waypoints_np, threshold=threshold)
        original_count = len(waypoints_np)
        filtered_count = len(filtered_np)
        deleted = original_count - filtered_count
        total_original += original_count
        total_filtered += filtered_count
        per_episode_deleted.append(deleted)

    if not per_episode_deleted:
        return {}
    arr = np.array(per_episode_deleted, dtype=np.float32)
    return {
        "waypoint_threshold": threshold,
        "waypoint_episodes": float(len(arr)),
        "waypoint_total_original": float(total_original),
        "waypoint_total_filtered": float(total_filtered),
        "waypoint_total_deleted": float(total_original - total_filtered),
        "waypoint_mean_deleted_per_episode": float(arr.mean()),
        "waypoint_median_deleted_per_episode": float(np.median(arr)),
        "waypoint_min_deleted_per_episode": float(arr.min()),
        "waypoint_max_deleted_per_episode": float(arr.max()),
        "waypoint_std_deleted_per_episode": float(arr.std()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_checkpoint", type=str, required=True)
    parser.add_argument("--val_root", type=str, required=True)
    parser.add_argument("--max_eval_samples", type=int, default=240)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--generation_max_new_tokens", type=int, default=6)
    parser.add_argument("--waypoint_threshold", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load base model with config workaround
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

    print("Running generation eval...")
    gen_metrics = run_generation_eval(
        model=model,
        processor=processor,
        eval_samples=eval_samples,
        max_samples=args.max_eval_samples,
        max_new_tokens=args.generation_max_new_tokens,
    )
    print(gen_metrics)

    print("Running waypoint filter analysis...")
    wp_metrics = compute_waypoint_filter_stats(
        val_root=args.val_root,
        max_eval_samples=args.max_eval_samples,
        threshold=args.waypoint_threshold,
        seed=args.seed + 1,
    )
    print(wp_metrics)

    metrics = {**gen_metrics, **wp_metrics}
    out_path = f"/tmp/eval_wp_t{args.waypoint_threshold}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
