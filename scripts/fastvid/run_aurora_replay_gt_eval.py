#!/usr/bin/env python3
"""AuroraReplay-GT style evaluation for FastVid on InternNav.

Per-step manual decode (avoids model.generate() Qwen2.5-VL bugs).
Ground-truth action prefix is fed to the model at each timestep.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from internnav.model.basemodel.qwen25vl_fastvid import (
    Qwen2_5_VLForConditionalGenerationFastVid,
)
from internnav.model.compression.feature_flags import FastVidConfig
from scripts.train.qwenvl_train.offline_r2r_multiframe_lora_sft_eval import (
    patch_torch_from_numpy,
)

# Action mapping consistent with OfflineR2RSFTDataset
ACTION_ID_TO_TEXT = {0: "stop", 1: "forward", 2: "left", 3: "right"}
ACTION_TEXT_TO_ID = {v: k for k, v in ACTION_ID_TO_TEXT.items()}


def _normalize_actions(raw_actions: List[int]) -> List[int]:
    """Drop leading -1 and keep only valid ids."""
    actions = [int(a) for a in raw_actions if isinstance(a, int)]
    if len(actions) > 0 and actions[0] == -1:
        actions = actions[1:]
    return [a for a in actions if a in ACTION_ID_TO_TEXT]


def _build_action_prefix(actions: List[int]) -> str:
    """Convert action list to space-separated text for GT prefix."""
    return " ".join(ACTION_ID_TO_TEXT[a] for a in actions)


# Regex to find the first valid action token (arrows or English words)
_ACTION_REGEX = re.compile(
    r"stop|forward|left|right|\u2191|\u2190|\u2192|\u25b2",
    re.IGNORECASE,
)


def _parse_first_action(text: str) -> int:
    """Extract first valid action from generated text. Invalid => -1."""
    match = _ACTION_REGEX.search(text)
    if not match:
        return -1
    token = match.group(0).lower()
    if token in ACTION_TEXT_TO_ID:
        return ACTION_TEXT_TO_ID[token]
    # Arrow symbol fallbacks
    if token in {"\u2191", "up"}:
        return 1
    if token in {"\u2190"}:
        return 2
    if token in {"\u2192"}:
        return 3
    if token in {"\u25b2"}:
        return 1  # treat as forward
    return -1


def _select_history_indices(total_previous: int, num_history: Optional[int]) -> List[int]:
    if total_previous <= 0:
        return []
    if num_history is None or num_history <= 0:
        return list(range(total_previous))
    if total_previous <= num_history:
        return list(range(total_previous))
    sampled = torch.linspace(0, total_previous - 1, num_history).long().tolist()
    return [int(v) for v in sampled]


def _load_frames(rgb_dir: str) -> List[str]:
    if not os.path.isdir(rgb_dir):
        return []
    return sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])


def _load_eval_records(dataset_root: str, max_episodes: int) -> List[Dict]:
    anno_path = os.path.join(dataset_root, "annotations.json")
    with open(anno_path, "r", encoding="utf-8") as f:
        annos = json.load(f)

    records = []
    for item in annos:
        instructions = item.get("instructions", item.get("instruction", None))
        if instructions is None:
            continue
        if isinstance(instructions, str):
            instructions = [instructions]
        elif not isinstance(instructions, list):
            continue

        actions = _normalize_actions(item.get("actions", []))
        if len(actions) == 0:
            continue

        video_rel = item.get("video", "")
        rgb_dir = os.path.join(dataset_root, video_rel, "rgb")
        frame_files = _load_frames(rgb_dir)
        if not frame_files:
            continue

        for ins_idx, instruction in enumerate(instructions):
            records.append(
                {
                    "id": item.get("id", len(records)),
                    "ins_idx": ins_idx,
                    "instruction": instruction,
                    "video_rel": video_rel,
                    "actions": actions,
                    "frame_files": frame_files,
                }
            )

    if max_episodes > 0:
        records = records[:max_episodes]
    return records


def _build_step_prompt(
    processor,
    instruction: str,
    images: List[Image.Image],
    num_history: int,
    gt_prefix_text: str,
) -> Dict[str, torch.Tensor]:
    """Build processor inputs for one AuroraReplay-GT step."""
    # User text mirrors OfflineR2RSFTDataset prompt style
    user_text = (
        f"You are an autonomous navigation assistant. "
        f"Instruction: {instruction}\n"
    )
    if num_history > 0:
        user_text += (
            f"These are your historical observations: "
            f"{'<image> ' * num_history}\n"
            "Look at the current RGB observation and predict the next action. "
        )
    else:
        user_text += "Look at the current RGB observation and predict the next action. "
    user_text += "Reply with exactly one word from: forward, left, right, stop."

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": user_text}],
        },
    ]

    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if gt_prefix_text:
        prompt_text += gt_prefix_text

    # Ensure there is a trailing space after prefix so model generates the next token cleanly
    if gt_prefix_text and not prompt_text.endswith(" "):
        prompt_text += " "

    inputs = processor(
        text=[prompt_text],
        images=images,
        return_tensors="pt",
    )
    return inputs


def _manual_decode(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    pixel_values: Optional[torch.Tensor],
    image_grid_thw: Optional[torch.Tensor],
    max_new_tokens: int,
    pad_token_id: int,
    stop_token_ids: List[int],
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Greedy manual decode loop. Returns (generated_ids, full_sequence)."""
    device = next(model.parameters()).device
    generated = []
    past_key_values = None

    # Prefill
    outputs = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        pixel_values=pixel_values.to(device) if pixel_values is not None else None,
        image_grid_thw=image_grid_thw.to(device) if image_grid_thw is not None else None,
        past_key_values=past_key_values,
        use_cache=True,
    )
    next_token_logits = outputs.logits[:, -1, :]
    next_token = next_token_logits.argmax(dim=-1)
    generated.append(next_token.item())
    past_key_values = outputs.past_key_values

    # Decode steps
    for _ in range(max_new_tokens - 1):
        if next_token.item() in stop_token_ids:
            break

        input_ids = next_token.unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        outputs = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            pixel_values=None,
            image_grid_thw=None,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        generated.append(next_token.item())
        past_key_values = outputs.past_key_values

    return torch.tensor(generated, dtype=torch.long), past_key_values


def run_aurora_eval(
    model,
    processor,
    records: List[Dict],
    num_history: int,
    max_new_tokens: int,
    device: torch.device,
    dataset_root: str,
) -> Dict:
    model.eval()
    total_correct = 0
    total_compared = 0
    class_total = {0: 0, 1: 0, 2: 0, 3: 0}
    class_correct = {0: 0, 1: 0, 2: 0, 3: 0}
    invalid_count = 0
    step_latencies: List[float] = []
    original_image_tokens: List[int] = []
    compressed_image_tokens: List[int] = []

    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id or 151643
    stop_token_ids = [
        processor.tokenizer.eos_token_id,
        processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
    ]
    stop_token_ids = [t for t in stop_token_ids if t is not None]

    for record in tqdm(records, desc="AuroraReplay-GT"):
        rgb_dir = os.path.join(dataset_root, record["video_rel"], "rgb")

        frame_files = record["frame_files"]
        gt_actions = record["actions"]
        pred_actions: List[int] = []

        rgb_list: List[Image.Image] = []

        for step_id in range(len(gt_actions)):
            frame_idx = min(step_id, len(frame_files) - 1)
            frame_path = os.path.join(rgb_dir, frame_files[frame_idx])
            try:
                image = Image.open(frame_path).convert("RGB")
            except Exception:
                image = Image.new("RGB", (640, 480), (0, 0, 0))
            rgb_list.append(image)

            history_indices = _select_history_indices(len(rgb_list) - 1, num_history)
            images = [rgb_list[idx] for idx in history_indices] + [rgb_list[-1]]
            history_count = len(history_indices)

            gt_prefix = _build_action_prefix(gt_actions[:step_id])

            inputs = _build_step_prompt(
                processor,
                record["instruction"],
                images,
                history_count,
                gt_prefix,
            )

            step_start = time.perf_counter()
            with torch.no_grad():
                generated_ids, _ = _manual_decode(
                    model,
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    inputs.get("pixel_values"),
                    inputs.get("image_grid_thw"),
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_token_id,
                    stop_token_ids=stop_token_ids,
                )
            step_latencies.append((time.perf_counter() - step_start) * 1000.0)

            generated_text = processor.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            pred_action = _parse_first_action(generated_text)
            pred_actions.append(pred_action)
            if pred_action == -1:
                invalid_count += 1

            # Collect compression stats
            if hasattr(model, "last_compression_stats") and model.last_compression_stats is not None:
                stats = model.last_compression_stats
                original_image_tokens.append(stats["original_image_tokens"])
                compressed_image_tokens.append(stats["compressed_image_tokens"])

        compare_len = min(len(pred_actions), len(gt_actions))
        for gt, pred in zip(gt_actions[:compare_len], pred_actions[:compare_len]):
            class_total[int(gt)] += 1
            if int(gt) == int(pred):
                total_correct += 1
                class_correct[int(gt)] += 1
        total_compared += compare_len

    overall_acc = (total_correct / total_compared) if total_compared > 0 else 0.0
    per_class_acc = {
        str(act): (class_correct[act] / class_total[act]) if class_total[act] > 0 else 0.0
        for act in [0, 1, 2, 3]
    }

    avg_orig = sum(original_image_tokens) / max(len(original_image_tokens), 1)
    avg_comp = sum(compressed_image_tokens) / max(len(compressed_image_tokens), 1)
    reduction = (avg_orig - avg_comp) / max(avg_orig, 1.0) if avg_orig > 0 else 0.0
    avg_latency = sum(step_latencies) / max(len(step_latencies), 1)
    fps = (total_compared / (sum(step_latencies) / 1000.0)) if step_latencies else 0.0

    return {
        "eval_protocol": "AuroraReplay-GT",
        "num_episodes": len(records),
        "num_actions_compared": total_compared,
        "num_actions_correct": total_correct,
        "overall_action_acc": overall_acc,
        "per_class_action_acc": per_class_acc,
        "invalid_prediction_rate": invalid_count / max(total_compared, 1),
        "avg_original_image_tokens": avg_orig,
        "avg_compressed_image_tokens": avg_comp,
        "token_reduction_ratio": reduction,
        "avg_latency_ms": avg_latency,
        "fps": fps,
    }


def main():
    parser = argparse.ArgumentParser(description="AuroraReplay-GT FastVid eval")
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1-System2")
    parser.add_argument("--dataset_root", type=str, default="/home/ubuntu/dataset/VLN-Trajectory-Data/R2R/offline_r2r_val_unseen")
    parser.add_argument("--output_dir", type=str, default="checkpoints/fastvid_aurora_gt")
    parser.add_argument("--num_history", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--max_episodes", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    patch_torch_from_numpy()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(args.model_path)
    records = _load_eval_records(args.dataset_root, args.max_episodes)
    if not records:
        raise RuntimeError("No eval records found.")

    results = {}

    # Baseline
    print("\n========== Baseline (no FastVid) ==========")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    model.eval().to(device)
    baseline_metrics = run_aurora_eval(
        model, processor, records, args.num_history, args.max_new_tokens, device,
        args.dataset_root,
    )
    results["baseline"] = baseline_metrics
    print(json.dumps(baseline_metrics, indent=2))
    del model
    torch.cuda.empty_cache()

    # FastVid ratios
    for ratio in args.ratios:
        print(f"\n========== FastVid keep_ratio={ratio} ==========")
        fastvid_cfg = FastVidConfig(
            enabled=True,
            retention_ratio=ratio,
            dyseg_c=8,
            dyseg_tau=0.9,
            stprune_d=0.4,
            dtm_p=4,
            dtm_beta=0.6,
            score_type="attn_proxy",
            min_tokens_per_frame=4,
        )
        model = Qwen2_5_VLForConditionalGenerationFastVid.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            attn_implementation=args.attn_implementation,
            fastvid_config=fastvid_cfg,
        )
        model.eval().to(device)
        metrics = run_aurora_eval(
            model, processor, records, args.num_history, args.max_new_tokens, device,
            args.dataset_root,
        )
        results[f"fastvid_keep_{ratio}"] = metrics
        print(json.dumps(metrics, indent=2))
        del model
        torch.cuda.empty_cache()

    summary_path = os.path.join(args.output_dir, "aurora_replay_gt_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n========== Summary ==========")
    print(f"{'Config':<25} {'Acc':>8} {'Invalid':>8} {'TokenRed':>10} {'FPS':>8}")
    for name, m in sorted(results.items()):
        acc = m.get("overall_action_acc", 0.0)
        inv = m.get("invalid_prediction_rate", 0.0)
        red = m.get("token_reduction_ratio", 0.0)
        fps = m.get("fps", 0.0)
        print(f"{name:<25} {acc:>8.4f} {inv:>7.2%} {red:>9.2%} {fps:>8.2f}")
    print(f"\nSaved results to: {summary_path}")


if __name__ == "__main__":
    main()
