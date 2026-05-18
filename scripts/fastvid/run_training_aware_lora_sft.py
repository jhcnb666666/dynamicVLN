#!/usr/bin/env python3
"""Training-Aware FastVid LoRA SFT suite for InternNav.

Trains one LoRA adapter per FastVid keep ratio on the R2R training set with
multi-frame historical observations.  After training, each checkpoint is
evaluated with generation action accuracy under the matching FastVid config.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

DEFAULT_RATIOS = [0.3, 0.5, 0.7]
DEFAULT_LORA_R = 64
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_MAX_STEPS = 1000


def build_train_command(
    base_script: Path,
    model_path: str,
    train_root: str,
    val_root: str,
    output_dir: str,
    ratio: float,
    num_history: int,
    max_steps: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    max_train_samples: int,
    max_eval_samples: int,
) -> List[str]:
    return [
        sys.executable,
        str(base_script),
        "--model_path", model_path,
        "--train_root", train_root,
        "--val_root", val_root,
        "--output_dir", output_dir,
        "--num_history", str(num_history),
        "--fastvid_enabled",
        "--fastvid_keep_ratio", str(ratio),
        "--max_steps", str(max_steps),
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "8",
        "--learning_rate", "1e-4",
        "--lora_r", str(lora_r),
        "--lora_alpha", str(lora_alpha),
        "--lora_dropout", str(lora_dropout),
        "--lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "--save_strategy", "no",
        "--eval_strategy", "no",
        "--report_to", "none",
        "--run_generation_eval",
        "--generation_eval_samples", "200",
        "--generation_max_new_tokens", "6",
        "--max_train_samples", str(max_train_samples),
        "--max_eval_samples", str(max_eval_samples),
        "--attn_implementation", "sdpa",
        "--dtype", "bf16",
    ]


def main():
    parser = argparse.ArgumentParser(description="Training-Aware FastVid LoRA SFT suite")
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1-System2")
    parser.add_argument("--train_root", type=str, default="/home/ubuntu/dataset/VLN-Trajectory-Data/R2R")
    parser.add_argument("--val_root", type=str, default="/home/ubuntu/dataset/VLN-Trajectory-Data/R2R/offline_r2r_val_unseen")
    parser.add_argument("--output_dir", type=str, default="checkpoints/fastvid_training_aware")
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--max_train_samples", type=int, default=8000)
    parser.add_argument("--max_eval_samples", type=int, default=1200)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    base_script = Path(project_root) / "scripts" / "train" / "qwenvl_train" / "offline_r2r_multiframe_lora_sft_eval.py"
    if not base_script.is_file():
        raise FileNotFoundError(f"Base training script not found: {base_script}")

    os.makedirs(args.output_dir, exist_ok=True)
    suite_results: Dict[str, Dict] = {}

    for ratio in args.ratios:
        ratio_tag = f"keep_{ratio}"
        run_output_dir = os.path.join(args.output_dir, ratio_tag)
        print(f"\n========== Training FastVid {ratio_tag} ==========")

        cmd = build_train_command(
            base_script=base_script,
            model_path=args.model_path,
            train_root=args.train_root,
            val_root=args.val_root,
            output_dir=run_output_dir,
            ratio=ratio,
            num_history=args.num_history,
            max_steps=args.max_steps,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
        )

        print("Command:", " ".join(cmd))
        if args.dry_run:
            print("[dry-run] skipping execution")
            suite_results[ratio_tag] = {"status": "dry_run"}
            continue

        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"

        proc = subprocess.run(cmd, env=env, cwd=str(project_root))
        suite_results[ratio_tag] = {"exit_code": proc.returncode}

        metrics_path = os.path.join(run_output_dir, "offline_eval_metrics.json")
        if os.path.isfile(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                suite_results[ratio_tag]["metrics"] = json.load(f)

    summary_path = os.path.join(args.output_dir, "training_aware_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(suite_results, f, indent=2)

    print(f"\nSaved training-aware summary to: {summary_path}")


if __name__ == "__main__":
    main()
