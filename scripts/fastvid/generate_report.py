#!/usr/bin/env python3
"""Generate comparison report for FastVid experiments on InternNav."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional


def load_json(path: str) -> Optional[Dict]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate FastVid comparison report")
    parser.add_argument("--training_free_results", type=str, default="checkpoints/fastvid_training_free/training_free_results.json")
    parser.add_argument("--training_aware_summary", type=str, default="checkpoints/fastvid_training_aware/training_aware_summary.json")
    parser.add_argument("--output_path", type=str, default="checkpoints/fastvid_report.md")
    args = parser.parse_args()

    tf_data = load_json(args.training_free_results) or {}
    ta_data = load_json(args.training_aware_summary) or {}

    lines = [
        "# FastVid on InternNav - Experiment Report",
        "",
        "## Training-Free Results",
        "",
        "| Config | Action Acc | Token Reduction | Samples/sec |",
        "|--------|-----------:|----------------:|------------:|",
    ]

    for name, metrics in sorted(tf_data.items()):
        acc = metrics.get("generation_action_acc", 0.0)
        red = metrics.get("token_reduction_ratio", 0.0)
        sps = metrics.get("samples_per_sec", 0.0)
        lines.append(f"| {name} | {acc:.4f} | {red:.2%} | {sps:.2f} |")

    lines.extend([
        "",
        "## Training-Aware Results",
        "",
        "| Config | Action Acc | Train Loss | Status |",
        "|--------|-----------:|-----------:|--------|",
    ])

    for name, info in sorted(ta_data.items()):
        metrics = info.get("metrics", {})
        acc = metrics.get("generation_action_acc", 0.0)
        loss = metrics.get("train_loss", 0.0)
        status = "ok" if info.get("exit_code") == 0 else "failed"
        lines.append(f"| {name} | {acc:.4f} | {loss:.4f} | {status} |")

    lines.extend([
        "",
        "## Notes",
        "",
        "- Baseline: InternVLA-N1-System2 (Qwen2.5-VL) without compression.",
        "- Training-Free: baseline checkpoint + FastVid compression, no fine-tuning.",
        "- Training-Aware: LoRA SFT under FastVid-compressed input distribution.",
        "- Eval metric: generation_action_acc (exact match of predicted action token).",
        "- Token reduction is computed on visual tokens only (image_token_id count).",
    ])

    report_text = "\n".join(lines) + "\n"
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"Saved report to: {args.output_path}")


if __name__ == "__main__":
    main()
