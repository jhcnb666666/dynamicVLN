#!/usr/bin/env python3
"""Training-Free FastVid evaluation on InternNav baseline.

Loads the baseline Qwen2.5-VL checkpoint, enables FastVid compression at
various keep ratios, and evaluates next-token action accuracy on the R2R
validation set with multi-frame historical observations.

Evaluation uses next-token logits (teacher-forcing style) rather than
model.generate(), because FastVid compression modifies input_ids length and
can interact poorly with Qwen2.5-VL's generation caching / position encoding.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from internnav.model.basemodel.qwen25vl_fastvid import Qwen2_5_VLForConditionalGenerationFastVid
from internnav.model.compression.feature_flags import FastVidConfig
from scripts.train.qwenvl_train.offline_r2r_multiframe_lora_sft_eval import (
    build_samples,
    OfflineR2RSFTDataset,
    QwenVLDataCollator,
    patch_torch_from_numpy,
)


def run_next_token_eval(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate action accuracy using next-token prediction (teacher-forcing)."""
    model.eval()
    correct = 0
    total = 0
    total_original_tokens = 0
    total_compressed_tokens = 0
    total_samples = 0
    start_time = time.time()

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits

            # Retrieve compressed labels stored by the custom forward.
            # The model receives original labels, compresses them internally,
            # and stores the result here so we can compute accuracy without
            # relying on the parent loss function (which sees labels=None).
            compressed_labels = getattr(model, "last_compressed_labels", None)
            if compressed_labels is None:
                # Fallback to original labels for baseline (no compression)
                compressed_labels = batch.get("labels")

            if compressed_labels is not None:
                # Next-token prediction: predict token at position t using logits at t-1
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = compressed_labels[..., 1:].contiguous()

                valid_mask = shift_labels != -100
                if valid_mask.any():
                    preds = shift_logits.argmax(dim=-1)
                    correct += (preds[valid_mask] == shift_labels[valid_mask]).sum().item()
                    total += valid_mask.sum().item()

            # Collect compression stats from the last forward
            if hasattr(model, "last_compression_stats") and model.last_compression_stats is not None:
                stats = model.last_compression_stats
                total_original_tokens += stats["original_image_tokens"]
                total_compressed_tokens += stats["compressed_image_tokens"]

            total_samples += batch["input_ids"].shape[0]

    elapsed = time.time() - start_time
    accuracy = correct / max(total, 1)
    avg_orig = total_original_tokens / max(total_samples, 1)
    avg_comp = total_compressed_tokens / max(total_samples, 1)
    reduction = (avg_orig - avg_comp) / max(avg_orig, 1.0) if avg_orig > 0 else 0.0

    return {
        "next_token_action_acc": accuracy,
        "eval_samples": float(total_samples),
        "eval_tokens_total": float(total),
        "eval_time_sec": elapsed,
        "samples_per_sec": total_samples / max(elapsed, 1e-6),
        "avg_original_image_tokens": avg_orig,
        "avg_compressed_image_tokens": avg_comp,
        "token_reduction_ratio": reduction,
    }


def main():
    parser = argparse.ArgumentParser(description="Training-Free FastVid eval")
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1-System2")
    parser.add_argument("--val_root", type=str, default="/home/ubuntu/dataset/VLN-Trajectory-Data/R2R/offline_r2r_val_unseen")
    parser.add_argument("--output_dir", type=str, default="checkpoints/fastvid_training_free")
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    patch_torch_from_numpy()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    processor = AutoProcessor.from_pretrained(args.model_path)
    eval_samples = build_samples(args.val_root, args.max_eval_samples, seed=43, num_history=args.num_history)
    if not eval_samples:
        raise RuntimeError("No eval samples found.")

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    eval_dataset = OfflineR2RSFTDataset(
        eval_samples, processor=processor, max_seq_length=1024, num_history=args.num_history
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=QwenVLDataCollator(pad_token_id=pad_token_id),
        shuffle=False,
    )

    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Baseline (no compression)
    print("\n========== Baseline (no FastVid) ==========")
    model = Qwen2_5_VLForConditionalGenerationFastVid.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        fastvid_config=FastVidConfig(enabled=False),
    )
    model.eval().to(device)
    baseline_metrics = run_next_token_eval(model, eval_dataloader, device)
    results["baseline"] = baseline_metrics
    print(json.dumps(baseline_metrics, indent=2))
    del model
    torch.cuda.empty_cache()

    # FastVid at each keep ratio
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
        metrics = run_next_token_eval(model, eval_dataloader, device)
        results[f"fastvid_keep_{ratio}"] = metrics
        print(json.dumps(metrics, indent=2))
        del model
        torch.cuda.empty_cache()

    summary_path = os.path.join(args.output_dir, "training_free_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n========== Summary ==========")
    print(f"{'Config':<20} {'Acc':>8} {'TokenRed':>10} {'Samples/s':>10}")
    for name, m in sorted(results.items()):
        acc = m.get("next_token_action_acc", 0.0)
        red = m.get("token_reduction_ratio", 0.0)
        sps = m.get("samples_per_sec", 0.0)
        print(f"{name:<20} {acc:>8.4f} {red:>9.2%} {sps:>10.2f}")
    print(f"\nSaved results to: {summary_path}")


if __name__ == "__main__":
    main()
