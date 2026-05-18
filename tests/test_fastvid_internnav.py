#!/usr/bin/env python3
"""Smoke tests for FastVid compression module in InternNav."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from internnav.model.compression.fastvid import (
    compress_frames_by_grid,
    apply_fastvid_compression,
    _compress_frame_sequence,
)
from internnav.model.compression.feature_flags import FastVidConfig


def test_feature_flags():
    cfg = FastVidConfig(enabled=True, retention_ratio=0.5)
    assert cfg.enabled is True
    assert cfg.retention_ratio == 0.5
    d = cfg.to_dict()
    assert d["enabled"] is True
    cfg2 = FastVidConfig.from_dict(d)
    assert cfg2.retention_ratio == 0.5
    print("[PASS] feature_flags")


def test_compress_frame_sequence():
    num_frames = 4
    tokens_per_frame = 64
    hidden_dim = 768
    frames = torch.randn(num_frames, tokens_per_frame, hidden_dim)
    cfg = FastVidConfig(enabled=True, retention_ratio=0.5, min_tokens_per_frame=4)
    compressed_frames, sizes, meta = _compress_frame_sequence(
        frames,
        retention_ratio=cfg.retention_ratio,
        dyseg_c=cfg.dyseg_c,
        dyseg_tau=cfg.dyseg_tau,
        stprune_d=cfg.stprune_d,
        dtm_p=cfg.dtm_p,
        dtm_beta=cfg.dtm_beta,
        score_type=cfg.score_type,
        min_tokens_per_frame=cfg.min_tokens_per_frame,
    )
    assert len(compressed_frames) == num_frames
    assert len(sizes) == num_frames
    for s in sizes:
        assert s <= tokens_per_frame
        assert s >= cfg.min_tokens_per_frame
    print(f"[PASS] compress_frame_sequence: {tokens_per_frame} -> {sizes}")


def test_compress_frames_by_grid():
    num_frames = 3
    tokens_per_frame = 196  # 14x14 patches with merge_size=2 -> 49? Actually 14*14=196, /4=49
    # Wait, Qwen2.5-VL uses merge_size=2, so 14x14 grid -> 7x7=49 tokens
    # Let's use 49 tokens per frame for realism
    tokens_per_frame = 49
    hidden_dim = 3584  # Qwen2.5-VL-7B visual hidden size
    total_patches = num_frames * tokens_per_frame

    image_embeds = torch.randn(total_patches, hidden_dim)
    # grid_thw: [num_frames, 3] -> (t, h, w)
    # For a 224x224 image with patch_size=14, grid is (1, 14, 14)
    # tokens = 1*14*14 // 4 = 49
    image_grid_thw = torch.tensor([[1, 14, 14], [1, 14, 14], [1, 14, 14]])

    cfg = FastVidConfig(enabled=True, retention_ratio=0.5, min_tokens_per_frame=4)
    compressed_embeds, compressed_sizes = compress_frames_by_grid(
        image_embeds, image_grid_thw, cfg
    )

    assert len(compressed_sizes) == num_frames
    total_compressed = sum(compressed_sizes)
    assert compressed_embeds.shape[0] == total_compressed
    assert compressed_embeds.shape[1] == hidden_dim
    for s in compressed_sizes:
        assert s <= tokens_per_frame
        assert s >= cfg.min_tokens_per_frame
    reduction = 1.0 - total_compressed / total_patches
    print(f"[PASS] compress_frames_by_grid: {total_patches} -> {total_compressed} (reduction={reduction:.2%}), sizes={compressed_sizes}")


def test_compress_frames_by_grid_disabled():
    num_frames = 2
    tokens_per_frame = 49
    hidden_dim = 3584
    total_patches = num_frames * tokens_per_frame
    image_embeds = torch.randn(total_patches, hidden_dim)
    image_grid_thw = torch.tensor([[1, 14, 14], [1, 14, 14]])
    cfg = FastVidConfig(enabled=False)
    compressed_embeds, compressed_sizes = compress_frames_by_grid(
        image_embeds, image_grid_thw, cfg
    )
    assert compressed_embeds.shape[0] == total_patches
    assert compressed_sizes == []
    print("[PASS] compress_frames_by_grid_disabled")


def test_apply_fastvid_compression():
    batch_size = 2
    num_frames = 4
    tokens_per_frame = 32
    hidden_dim = 768
    image_features = [torch.randn(num_frames, tokens_per_frame, hidden_dim) for _ in range(batch_size)]
    memory_features = [torch.randn(2, num_frames * tokens_per_frame, hidden_dim) for _ in range(batch_size)]
    cfg = FastVidConfig(enabled=True, retention_ratio=0.7)
    comp_img, comp_mem, stats = apply_fastvid_compression(image_features, memory_features, cfg)
    assert len(comp_img) == batch_size
    assert len(comp_mem) == batch_size
    assert stats["method"] == "fastvid"
    print(f"[PASS] apply_fastvid_compression: stats={stats['retention_ratio']}")


def test_extreme_keep_ratio():
    num_frames = 2
    tokens_per_frame = 16
    hidden_dim = 768
    frames = torch.randn(num_frames, tokens_per_frame, hidden_dim)
    cfg = FastVidConfig(enabled=True, retention_ratio=0.1, min_tokens_per_frame=4)
    compressed_frames, sizes, meta = _compress_frame_sequence(
        frames,
        retention_ratio=cfg.retention_ratio,
        dyseg_c=cfg.dyseg_c,
        dyseg_tau=cfg.dyseg_tau,
        stprune_d=cfg.stprune_d,
        dtm_p=cfg.dtm_p,
        dtm_beta=cfg.dtm_beta,
        score_type=cfg.score_type,
        min_tokens_per_frame=cfg.min_tokens_per_frame,
    )
    for s in sizes:
        assert s >= cfg.min_tokens_per_frame
    print(f"[PASS] extreme_keep_ratio: sizes={sizes}")


def main():
    test_feature_flags()
    test_compress_frame_sequence()
    test_compress_frames_by_grid()
    test_compress_frames_by_grid_disabled()
    test_apply_fastvid_compression()
    test_extreme_keep_ratio()
    print("\nAll FastVid smoke tests passed!")


if __name__ == "__main__":
    main()
