#!/usr/bin/env python3
"""Test that Qwen2_5_VLForConditionalGenerationFastVid can load from_pretrained."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoProcessor

from internnav.model.basemodel.qwen25vl_fastvid import Qwen2_5_VLForConditionalGenerationFastVid
from internnav.model.compression.feature_flags import FastVidConfig
from scripts.train.qwenvl_train.offline_r2r_multiframe_lora_sft_eval import patch_torch_from_numpy

patch_torch_from_numpy()


def test_load_baseline():
    model_path = "checkpoints/InternVLA-N1-System2"
    cfg = FastVidConfig(enabled=False)
    model = Qwen2_5_VLForConditionalGenerationFastVid.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        fastvid_config=cfg,
    )
    assert model.fastvid_config.enabled is False
    assert hasattr(model, "last_compression_stats")
    print("[PASS] Loaded baseline model")
    del model


def test_load_fastvid():
    model_path = "checkpoints/InternVLA-N1-System2"
    cfg = FastVidConfig(enabled=True, retention_ratio=0.5)
    model = Qwen2_5_VLForConditionalGenerationFastVid.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        fastvid_config=cfg,
    )
    assert model.fastvid_config.enabled is True
    assert model.fastvid_config.retention_ratio == 0.5
    print("[PASS] Loaded FastVid model")
    del model


def test_generate_one_sample():
    """Run a single sample through the model with FastVid enabled."""
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_path = "checkpoints/InternVLA-N1-System2"
    processor = AutoProcessor.from_pretrained(model_path)
    cfg = FastVidConfig(enabled=True, retention_ratio=0.5)
    model = Qwen2_5_VLForConditionalGenerationFastVid.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        fastvid_config=cfg,
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    from PIL import Image
    # Use a dummy image
    dummy_image = Image.new("RGB", (224, 224), color="red")
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image> what?<|im_end|>\n<|im_start|>assistant\n"

    inputs = processor(text=[prompt], images=[dummy_image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=5,
            use_cache=True,
        )

    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"[PASS] Generated text: {generated_text[:50]}...")
    print(f"  last_compression_stats: {model.last_compression_stats}")
    del model


def main():
    test_load_baseline()
    test_load_fastvid()
    test_generate_one_sample()
    print("\nAll from_pretrained tests passed!")


if __name__ == "__main__":
    main()
