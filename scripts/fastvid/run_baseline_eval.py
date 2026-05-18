#!/usr/bin/env python3
"""Baseline evaluation: load pretrained checkpoint without LoRA and evaluate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.train.qwenvl_train.offline_r2r_multiframe_lora_sft_eval import (
    build_samples,
    OfflineR2RSFTDataset,
    QwenVLDataCollator,
    run_generation_eval,
    patch_torch_from_numpy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline eval without training")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--val_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--num_history", type=int, default=0)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--generation_eval_samples", type=int, default=200)
    parser.add_argument("--generation_max_new_tokens", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    patch_torch_from_numpy()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    processor = AutoProcessor.from_pretrained(args.model_path)

    print(f"Loading baseline model from: {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    model.eval().cuda()
    print("Model loaded.")

    print(f"Building eval samples from: {args.val_root}")
    eval_samples = build_samples(args.val_root, args.max_eval_samples, seed=43, num_history=args.num_history)
    if not eval_samples:
        raise RuntimeError("No eval samples found.")
    print(f"Eval samples: {len(eval_samples)}")

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    eval_dataset = OfflineR2RSFTDataset(
        eval_samples, processor=processor, max_seq_length=args.max_seq_length, num_history=args.num_history
    )

    print("Running generation eval...")
    gen_metrics = run_generation_eval(
        model=model,
        processor=processor,
        eval_samples=eval_samples,
        max_samples=args.generation_eval_samples,
        max_new_tokens=args.generation_max_new_tokens,
        num_history=args.num_history,
    )

    metrics = {
        "model": "baseline (no LoRA)",
        "val_root": args.val_root,
        "num_history": args.num_history,
        **gen_metrics,
    }

    metrics_path = os.path.join(args.output_dir, "baseline_eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n========== Baseline Eval Results ==========")
    print(json.dumps(metrics, indent=2))
    print(f"Saved to: {metrics_path}")


if __name__ == "__main__":
    main()
