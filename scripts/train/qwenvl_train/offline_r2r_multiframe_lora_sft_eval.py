#!/usr/bin/env python3
"""Offline R2R multi-frame LoRA SFT + evaluation with FastVid support.

Extends the single-frame script to support historical observations (num_history)
and optional FastVid token compression.

Data format:
- <root>/annotations.json
- <root>/images/<episode>/rgb/*.jpg

Each sample uses num_history historical frames + 1 current frame.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model, PeftModel

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from internnav.model.basemodel.qwen25vl_fastvid import Qwen2_5_VLForConditionalGenerationFastVid
from internnav.model.compression.feature_flags import FastVidConfig


ACTION_MAP = {
    0: "stop",
    1: "forward",
    2: "left",
    3: "right",
}

ALLOWED_ACTIONS = tuple(ACTION_MAP.values())


@dataclass
class OfflineSample:
    frame_path: str
    instruction: str
    action: str
    history_frame_paths: List[str]


def patch_torch_from_numpy() -> bool:
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


def build_samples(
    data_root: str,
    max_samples: Optional[int],
    seed: int,
    num_history: int,
) -> List[OfflineSample]:
    annotation_path = os.path.join(data_root, "annotations.json")
    if not os.path.isfile(annotation_path):
        raise FileNotFoundError(f"annotations.json not found: {annotation_path}")

    with open(annotation_path, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    random.Random(seed).shuffle(episodes)

    samples: List[OfflineSample] = []
    skipped_no_frames = 0
    skipped_short_actions = 0

    for ep in episodes:
        actions = ep.get("actions", [])
        if len(actions) <= 1:
            skipped_short_actions += 1
            continue

        video_rel = ep.get("video", "")
        rgb_dir = os.path.join(data_root, video_rel, "rgb")
        frames = list_rgb_frames(rgb_dir)
        if not frames:
            skipped_no_frames += 1
            continue

        instruction = normalize_instruction(ep.get("instructions") or ep.get("instruction", ""))
        usable_steps = min(len(frames), len(actions) - 1)

        for step in range(usable_steps):
            action_id = actions[step + 1]
            action_text = ACTION_MAP.get(action_id)
            if action_text is None:
                continue

            # Collect historical frames
            if num_history > 0:
                if step > 0:
                    start_idx = max(0, step - num_history)
                    history_paths = frames[start_idx:step]
                else:
                    history_paths = []
                # Pad to num_history (repeat first available, or current frame if none)
                fill_frame = history_paths[0] if history_paths else frames[step]
                while len(history_paths) < num_history:
                    history_paths.insert(0, fill_frame)
            else:
                history_paths = []

            samples.append(
                OfflineSample(
                    frame_path=frames[step],
                    instruction=instruction,
                    action=action_text,
                    history_frame_paths=history_paths,
                )
            )
            if max_samples is not None and len(samples) >= max_samples:
                print(f"Reached max_samples={max_samples} for {data_root}")
                print(f"Skipped episodes: no_frames={skipped_no_frames}, short_actions={skipped_short_actions}")
                return samples

    print(f"Collected {len(samples)} samples from {data_root}")
    print(f"Skipped episodes: no_frames={skipped_no_frames}, short_actions={skipped_short_actions}")
    return samples


def build_user_prompt(instruction: str, num_history: int) -> str:
    base = (
        "You are an autonomous navigation assistant. "
        f"Instruction: {instruction}\n"
    )
    if num_history > 0:
        base += (
            f"These are your historical observations: {'<image> ' * num_history}\n"
            "Current observation: <image>\n"
            "Look at the current RGB observation and predict the next action. "
        )
    else:
        base += "Look at the current RGB observation and predict the next action. "
    base += "Reply with exactly one word from: forward, left, right, stop."
    return base


class OfflineR2RSFTDataset(Dataset):
    def __init__(self, samples: List[OfflineSample], processor, max_seq_length: int, num_history: int):
        self.samples = samples
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.num_history = num_history

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        images = [Image.open(p).convert("RGB") for p in sample.history_frame_paths]
        images.append(Image.open(sample.frame_path).convert("RGB"))

        user_prompt = build_user_prompt(sample.instruction, self.num_history)
        prompt_messages = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": user_prompt}],
            }
        ]

        full_messages = prompt_messages + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample.action}],
            }
        ]

        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Use the same processor path for both prompt and full text so that
        # special-token handling (e.g. BOS) is identical and prompt_len aligns
        # exactly with input_ids.
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=images,
            return_tensors="pt",
        )
        prompt_ids = prompt_inputs["input_ids"][0]

        model_inputs = self.processor(
            text=[full_text],
            images=images,
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"][0][: self.max_seq_length]
        attention_mask = model_inputs["attention_mask"][0][: self.max_seq_length]

        labels = input_ids.clone()
        prompt_len = min(prompt_ids.shape[0], labels.shape[0])
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": model_inputs["pixel_values"],
            "image_grid_thw": model_inputs["image_grid_thw"],
        }


@dataclass
class QwenVLDataCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        max_len = max(f["input_ids"].shape[0] for f in features)

        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, feature in enumerate(features):
            seq_len = feature["input_ids"].shape[0]
            input_ids[i, :seq_len] = feature["input_ids"]
            attention_mask[i, :seq_len] = feature["attention_mask"]
            labels[i, :seq_len] = feature["labels"]

        pixel_values = torch.cat([f["pixel_values"] for f in features], dim=0)
        image_grid_thw = torch.cat([f["image_grid_thw"] for f in features], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


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
def run_generation_eval(
    model,
    processor,
    eval_samples: List[OfflineSample],
    max_samples: int,
    max_new_tokens: int,
    num_history: int,
) -> Dict[str, float]:
    if not eval_samples:
        return {"generation_eval_samples": 0, "generation_action_acc": 0.0}

    model.eval()
    subset = eval_samples[:max_samples]

    correct = 0
    total = 0

    for sample in subset:
        images = [Image.open(p).convert("RGB") for p in sample.history_frame_paths]
        images.append(Image.open(sample.frame_path).convert("RGB"))

        prompt_messages = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": build_user_prompt(sample.instruction, num_history)}],
            }
        ]

        prompt_text = processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[prompt_text],
            images=images,
            return_tensors="pt",
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_p=1.0,
            top_k=50,
            use_cache=True,
        )

        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        pred_action = parse_action_from_text(generated_text)

        if pred_action == sample.action:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    return {
        "generation_eval_samples": float(total),
        "generation_action_acc": accuracy,
    }


def load_model_with_lora(args):
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    fastvid_cfg = FastVidConfig(
        enabled=args.fastvid_enabled,
        retention_ratio=args.fastvid_keep_ratio,
        dyseg_c=args.fastvid_dyseg_c,
        dyseg_tau=args.fastvid_dyseg_tau,
        stprune_d=args.fastvid_stprune_d,
        dtm_p=args.fastvid_dtm_p,
        dtm_beta=args.fastvid_dtm_beta,
        score_type=args.fastvid_score_type,
        min_tokens_per_frame=args.fastvid_min_tokens_per_frame,
    )

    if fastvid_cfg.enabled:
        print(f"Loading Qwen2_5_VLForConditionalGenerationFastVid with FastVid config: {fastvid_cfg.to_dict()}")
        model = Qwen2_5_VLForConditionalGenerationFastVid.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            attn_implementation=args.attn_implementation,
            fastvid_config=fastvid_cfg,
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            attn_implementation=args.attn_implementation,
        )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.resume_from_checkpoint:
        print(f"Resuming from LoRA checkpoint: {args.resume_from_checkpoint}")
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
        model.print_trainable_parameters()
        return model

    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Offline R2R multi-frame LoRA SFT + eval with FastVid")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--val_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_train_samples", type=int, default=8000)
    parser.add_argument("--max_eval_samples", type=int, default=1200)
    parser.add_argument("--max_seq_length", type=int, default=1024)

    parser.add_argument("--num_history", type=int, default=8, help="Number of historical frames per sample. 0 = single frame.")

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--eval_strategy", type=str, default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to existing LoRA adapter to resume training from")

    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "eager"])
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # FastVid arguments
    parser.add_argument("--fastvid_enabled", action="store_true", help="Enable FastVid token compression")
    parser.add_argument("--fastvid_keep_ratio", type=float, default=0.5)
    parser.add_argument("--fastvid_dyseg_c", type=int, default=8)
    parser.add_argument("--fastvid_dyseg_tau", type=float, default=0.9)
    parser.add_argument("--fastvid_stprune_d", type=float, default=0.4)
    parser.add_argument("--fastvid_dtm_p", type=int, default=4)
    parser.add_argument("--fastvid_dtm_beta", type=float, default=0.6)
    parser.add_argument("--fastvid_score_type", type=str, default="attn_proxy")
    parser.add_argument("--fastvid_min_tokens_per_frame", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--run_generation_eval", action="store_true")
    parser.add_argument("--generation_eval_samples", type=int, default=200)
    parser.add_argument("--generation_max_new_tokens", type=int, default=6)

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    patched = patch_torch_from_numpy()
    if patched:
        print("Patched torch.from_numpy with a safe fallback for this environment.")

    set_seed(args.seed)

    processor = AutoProcessor.from_pretrained(args.model_path)

    train_samples = build_samples(args.train_root, args.max_train_samples, seed=args.seed, num_history=args.num_history)
    eval_samples = build_samples(args.val_root, args.max_eval_samples, seed=args.seed + 1, num_history=args.num_history)

    if not train_samples:
        raise RuntimeError("No train samples found. Check --train_root.")
    if not eval_samples:
        raise RuntimeError("No eval samples found. Check --val_root.")

    train_dataset = OfflineR2RSFTDataset(train_samples, processor=processor, max_seq_length=args.max_seq_length, num_history=args.num_history)
    eval_dataset = OfflineR2RSFTDataset(eval_samples, processor=processor, max_seq_length=args.max_seq_length, num_history=args.num_history)

    model = load_model_with_lora(args)

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    data_collator = QwenVLDataCollator(pad_token_id=pad_token_id)

    use_bf16 = args.dtype == "bf16"
    use_fp16 = args.dtype == "fp16"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        report_to=args.report_to,
        dataloader_num_workers=args.dataloader_num_workers,
        label_names=["labels"],
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )

    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    metrics: Dict[str, float] = dict(train_result.metrics)

    if args.eval_strategy != "no":
        eval_metrics = trainer.evaluate()
        metrics.update({f"trainer_{k}": v for k, v in eval_metrics.items()})

    if args.run_generation_eval:
        gen_metrics = run_generation_eval(
            model=trainer.model,
            processor=processor,
            eval_samples=eval_samples,
            max_samples=args.generation_eval_samples,
            max_new_tokens=args.generation_max_new_tokens,
            num_history=args.num_history,
        )
        metrics.update(gen_metrics)

    metrics_path = os.path.join(args.output_dir, "offline_eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training and evaluation finished.")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
