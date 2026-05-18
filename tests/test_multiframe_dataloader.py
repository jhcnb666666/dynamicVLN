#!/usr/bin/env python3
"""Quick test for multi-frame data loader."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoProcessor

from scripts.train.qwenvl_train.offline_r2r_multiframe_lora_sft_eval import (
    patch_torch_from_numpy,
    build_samples,
    OfflineR2RSFTDataset,
    QwenVLDataCollator,
)


def test_build_samples():
    data_root = "/home/ubuntu/dataset/VLN-Trajectory-Data/R2R"
    samples = build_samples(data_root, max_samples=10, seed=42, num_history=4)
    assert len(samples) > 0
    for s in samples:
        assert len(s.history_frame_paths) <= 4
        assert Path(s.frame_path).exists()
        for p in s.history_frame_paths:
            assert Path(p).exists()
    print(f"[PASS] build_samples: collected {len(samples)} samples with history")


def test_dataset_and_collator():
    data_root = "/home/ubuntu/dataset/VLN-Trajectory-Data/R2R"
    model_path = "checkpoints/InternVLA-N1-System2"
    processor = AutoProcessor.from_pretrained(model_path)

    samples = build_samples(data_root, max_samples=4, seed=42, num_history=2)
    dataset = OfflineR2RSFTDataset(samples, processor=processor, max_seq_length=512, num_history=2)

    items = [dataset[i] for i in range(len(dataset))]
    for item in items:
        assert "pixel_values" in item
        assert "image_grid_thw" in item
        # num_images = num_history + 1 = 3
        assert item["image_grid_thw"].shape[0] == 3, f"Expected 3 images, got {item['image_grid_thw'].shape[0]}"
        print(f"  sample: input_ids={item['input_ids'].shape}, images={item['image_grid_thw'].shape}")

    collator = QwenVLDataCollator(pad_token_id=processor.tokenizer.pad_token_id)
    batch = collator(items)
    assert batch["input_ids"].shape[0] == len(items)
    # pixel_values is flattened patches; image_grid_thw has one row per image
    assert batch["image_grid_thw"].shape[0] == len(items) * 3  # 3 images per sample
    total_patches = batch["image_grid_thw"][:, 0] * batch["image_grid_thw"][:, 1] * batch["image_grid_thw"][:, 2]
    total_patches = total_patches.sum().item()
    assert batch["pixel_values"].shape[0] == total_patches
    print(f"[PASS] collator: batch input_ids={batch['input_ids'].shape}, pixel_values={batch['pixel_values'].shape}")


def main():
    patch_torch_from_numpy()
    test_build_samples()
    test_dataset_and_collator()
    print("\nAll multi-frame dataloader tests passed!")


if __name__ == "__main__":
    main()
