#!/usr/bin/env python3
"""Smoke tests for Qwen2_5_VLForConditionalGenerationFastVid."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoProcessor, AutoTokenizer

from internnav.model.basemodel.qwen25vl_fastvid import Qwen2_5_VLForConditionalGenerationFastVid
from internnav.model.compression.feature_flags import FastVidConfig


def test_forward_with_compression():
    model_path = "checkpoints/InternVLA-N1-System2"
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    # Build a small randomly-initialized model for shape testing
    from transformers import Qwen2_5_VLConfig
    config = Qwen2_5_VLConfig.from_pretrained(model_path)
    # Reduce layers for faster init but keep head dimensions matching rope_scaling
    config.num_hidden_layers = 2
    # visual_config is a dict or object; set hidden_size small to skip heavy ViT init
    if hasattr(config, "visual_config") and config.visual_config is not None:
        if isinstance(config.visual_config, dict):
            config.visual_config["hidden_size"] = 256
            config.visual_config["num_hidden_layers"] = 2
        else:
            config.visual_config.hidden_size = 256
            config.visual_config.num_hidden_layers = 2

    fastvid_cfg = FastVidConfig(enabled=True, retention_ratio=0.5, min_tokens_per_frame=4)
    model = Qwen2_5_VLForConditionalGenerationFastVid(config, fastvid_config=fastvid_cfg)
    model.eval()

    # Since we skipped visual_config, we need to mock the visual tower behavior
    # We'll directly test _compress_for_batch and build inputs_embeds logic
    batch_size = 2
    num_images_per_sample = 3
    tokens_per_image = 49  # 14x14 grid, merge_size=2 -> 49
    hidden_dim = config.hidden_size
    total_tokens = batch_size * num_images_per_sample * tokens_per_image

    image_embeds = torch.randn(total_tokens, hidden_dim)
    image_grid_thw = torch.tensor([[1, 14, 14]] * (batch_size * num_images_per_sample))

    # Build input_ids with image_token_id placeholders
    image_token_id = int(config.image_token_id)
    pad_token_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    for b in range(batch_size):
        # text: "<|im_start|>user\n<image> what?<|im_end|>\n<|im_start|>assistant\nforward<|im_end|>"
        # Simplified: just pad tokens + image tokens + pad tokens
        text_len = 10
        ids = [pad_token_id] * 5 + [image_token_id] * (num_images_per_sample * tokens_per_image) + [pad_token_id] * 5
        am = [1] * len(ids)
        lbl = [-100] * 5 + [-100] * (num_images_per_sample * tokens_per_image) + list(range(len(ids) - 5, len(ids)))
        input_ids_list.append(ids)
        attention_mask_list.append(am)
        labels_list.append(lbl)

    max_len = max(len(x) for x in input_ids_list)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    for b in range(batch_size):
        length = len(input_ids_list[b])
        input_ids[b, :length] = torch.tensor(input_ids_list[b], dtype=torch.long)
        attention_mask[b, :length] = torch.tensor(attention_mask_list[b], dtype=torch.long)
        labels[b, :length] = torch.tensor(labels_list[b], dtype=torch.long)

    original_image_token_count = (input_ids == image_token_id).sum().item()
    print(f"Original image tokens: {original_image_token_count}")

    compressed_embeds, new_input_ids, new_attention_mask, new_labels = model._compress_for_batch(
        image_embeds, image_grid_thw, input_ids, attention_mask, labels
    )

    new_image_token_count = (new_input_ids == image_token_id).sum().item()
    print(f"Compressed image tokens: {new_image_token_count}")
    assert new_image_token_count < original_image_token_count
    assert compressed_embeds.shape[0] == new_image_token_count
    assert compressed_embeds.shape[1] == hidden_dim

    # Verify labels alignment: non-image text labels should be preserved
    for b in range(batch_size):
        old_text_labels = [labels[b, i].item() for i in range(labels.shape[1]) if labels[b, i].item() != -100]
        new_text_labels = [new_labels[b, i].item() for i in range(new_labels.shape[1]) if new_labels[b, i].item() != -100]
        # After compression, image tokens are still -100, text labels should match
        assert old_text_labels == new_text_labels, f"Label mismatch at batch {b}"

    print("[PASS] _compress_for_batch shape and label preservation")

    # Test full forward (with mocked visual to avoid heavy init)
    # We bypass visual by providing inputs_embeds directly, but that's the non-fastvid path.
    # For the fastvid path, we need visual to return our image_embeds.
    # We'll monkey-patch model.visual for the test.
    class MockVisual(torch.nn.Module):
        def __init__(self, output):
            super().__init__()
            self.output = output
            self.dtype = torch.float32
        def forward(self, pixel_values, grid_thw=None):
            return self.output

    model.visual = MockVisual(image_embeds)
    dummy_pixel_values = torch.randn(1)  # placeholder, visual ignores it

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        pixel_values=dummy_pixel_values,
        image_grid_thw=image_grid_thw,
    )

    assert outputs.logits is not None
    assert outputs.logits.shape[0] == batch_size
    # Sequence length should be reduced
    assert outputs.logits.shape[1] == new_input_ids.shape[1]
    print(f"[PASS] forward with FastVid: logits shape {outputs.logits.shape}")

    # Test disabled path
    model.fastvid_config.enabled = False
    outputs2 = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        pixel_values=dummy_pixel_values,
        image_grid_thw=image_grid_thw,
    )
    assert outputs2.logits.shape[1] == input_ids.shape[1]
    print(f"[PASS] forward without FastVid: logits shape {outputs2.logits.shape}")


def main():
    test_forward_with_compression()
    print("\nAll Qwen2.5-VL FastVid tests passed!")


if __name__ == "__main__":
    main()
