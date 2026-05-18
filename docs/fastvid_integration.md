# FastVid Video Token Compression Integration

This document describes how FastVid video token compression has been integrated into InternNav, a Qwen2.5-VL based VLN agent. FastVid was originally developed for StreamVLN; this migration adapts it to work with Qwen2.5-VL's architecture and InternNav's data pipeline.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [File Map](#file-map)
3. [Key Design Decisions](#key-design-decisions)
4. [Bug Fixes & Lessons Learned](#bug-fixes--lessons-learned)
5. [Usage](#usage)
6. [Known Limitations](#known-limitations)

---

## Architecture Overview

FastVid compresses visual tokens **after** the vision tower projects image patches into the LLM's embedding space, but **before** they are scattered into the input token sequence. This design minimizes intrusion into the base model.

```
Images → Processor → pixel_values + image_grid_thw
                          ↓
                   self.visual(pixel_values, grid_thw)
                          ↓
              [total_visual_tokens, hidden_dim]
                          ↓
            ┌─────────────────────────────┐
            │   FastVid Compression         │
            │   - Attention token selection │
            │   - Dynamic segmentation      │
            │   - Density-based merging     │
            └─────────────────────────────┘
                          ↓
              [compressed_tokens, hidden_dim]
                          ↓
           Rebuild inputs_embeds with compressed
           visual tokens, adjusting input_ids,
           attention_mask, and labels.
                          ↓
           1D causal position_ids (bypass mrope)
                          ↓
           LLM forward (super().forward)
```

### Why 1D Causal Position IDs?

Qwen2.5-VL uses 3D multimodal RoPE (`get_rope_index`) which depends on `image_grid_thw`. After FastVid compression changes token counts per frame, `get_rope_index` breaks with shape mismatches. We bypass it entirely by constructing simple 1D causal `position_ids`:

```python
position_ids = torch.arange(seq_length).view(1, 1, -1).expand(3, batch_size, seq_length)
```

This is compatible with `attn_implementation="sdpa"`.

---

## File Map

| File | Purpose |
|------|---------|
| `internnav/model/compression/fastvid.py` | Core FastVid algorithm: token scoring, dynamic segmentation, DTM |
| `internnav/model/compression/feature_flags.py` | `FastVidConfig` dataclass with hyperparameters |
| `internnav/model/basemodel/qwen25vl_fastvid.py` | `Qwen2_5_VLForConditionalGenerationFastVid` subclass |
| `scripts/fastvid/run_aurora_replay_gt_eval.py` | AuroraReplay-GT style evaluation (manual decode) |
| `scripts/train/qwenvl_train/offline_r2r_multiframe_lora_sft_eval.py` | Multi-frame dataset with history frames |

---

## Key Design Decisions

### 1. Compression Entry Point

The custom `forward()` intercepts calls when `pixel_values` is present and `past_key_values` is `None` (prefill). During cached generation (decode), `past_key_values` is not `None`, so compression is skipped automatically.

### 2. Spatial Merge Size Division

Qwen2.5-VL's vision tower merges `spatial_merge_size × spatial_merge_size` patches into one token (default `merge_size=2`). The raw grid `t*h*w` must be divided by `merge_size^2` (= 4) to get the actual visual token count per frame:

```python
tokens_per_frame = image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2] // 4
```

### 3. Label Alignment for SFT

`OfflineR2RSFTDataset` masks prompt tokens with `-100`. To ensure perfect alignment between `prompt_len` and `input_ids`, both must use the **same processor path**:

```python
prompt_inputs = processor(text=[prompt_text], images=images, return_tensors="pt")
prompt_ids = prompt_inputs["input_ids"][0]
```

Using `tokenizer(..., add_special_tokens=False)` caused an off-by-one mismatch, leaving prompt tokens unmasked and corrupting accuracy metrics.

### 4. Manual Decode for Inference

`model.generate()` crashes on Qwen2.5-VL subclasses due to `get_rope_index` shape mismatches during decoding. We use a **manual decode loop** instead:

```python
past_key_values = None
for _ in range(max_new_tokens):
    out = model(input_ids=..., attention_mask=...,
                pixel_values=pixel_values if past_key_values is None else None,
                past_key_values=past_key_values, use_cache=True)
    next_token = out.logits[:, -1, :].argmax(dim=-1)
    past_key_values = out.past_key_values
```

---

## Bug Fixes & Lessons Learned

### Bug 1: `tokens_per_frame` Missing `spatial_merge_size` Division

**Symptom**: `compress_frames_by_grid` sliced `image_embeds` with `t*h*w` instead of `t*h*w//4`, causing fallback to no-compression or `torch.stack` shape mismatch.

**Fix**: Added `spatial_merge_size` parameter (default 2) and divide by `merge_size^2`:
```python
tokens_per_frame = (
    image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
    // (spatial_merge_size * spatial_merge_size)
).tolist()
```

### Bug 2: Missing `past_key_values` Guard in Custom Forward

**Symptom**: `model.generate()` crashed during cached generation with `IndexError` in `get_rope_index`.

**Fix**: Added `is_prefill` check requiring `past_key_values is None` (or `get_seq_length() == 0`):
```python
is_prefill = (
    pixel_values is not None
    and image_grid_thw is not None
    and (past_key_values is None or past_key_values.get_seq_length() == 0)
)
```

### Bug 3: `labels` Length Mismatch in Eval

**Symptom**: After compression, `logits` length differed from `batch["labels"]` length, breaking accuracy calculation.

**Fix**: The custom `forward()` stores compressed labels as `self.last_compressed_labels` and passes `labels=None` to `super().forward()` (to avoid parent loss computation). Eval scripts retrieve labels from `model.last_compressed_labels`.

### Bug 4: Action Parser Failing on Repeated Arrow Symbols

**Symptom**: Model generated `←←←←←←←←←` without spaces; `split()` treated it as one word, failing regex match.

**Fix**: Switched from `text.split()` to `re.search()` with a compiled regex:
```python
_ACTION_REGEX = re.compile(r"stop|forward|left|right|\u2191|\u2190|\u2192|\u25b2", re.IGNORECASE)
```

---

## Usage

### Training-Free Evaluation

Run AuroraReplay-GT style evaluation on the R2R validation set:

```bash
cd /home/ubuntu/project/InternNav-lora
python scripts/fastvid/run_aurora_replay_gt_eval.py \
    --model_path checkpoints/InternVLA-N1-System2 \
    --dataset_root /home/ubuntu/dataset/VLN-Trajectory-Data/R2R/offline_r2r_val_unseen \
    --num_history 3 \
    --ratios 0.3 0.5 0.7 \
    --max_episodes 50 \
    --max_new_tokens 10
```

This will evaluate baseline and FastVid at keep ratios 0.3, 0.5, 0.7, producing:
- `checkpoints/fastvid_aurora_gt/aurora_replay_gt_results.json`
- Console summary with accuracy, invalid rate, token reduction, and FPS.

### Training-Aware LoRA SFT (Phase 5)

To be implemented. The plan is to load `Qwen2_5_VLForConditionalGenerationFastVid`, apply LoRA adapters to target modules, and fine-tune on the multi-frame SFT dataset with FastVid compression active during training.

---

## Known Limitations

1. **`model.generate()` not supported**: Due to Qwen2.5-VL `get_rope_index` incompatibility with custom subclasses, always use the manual decode loop for inference.
2. **Video inputs not supported**: Only RGB images are handled; `pixel_values_videos` is ignored.
3. **Action vocabulary**: Current parser expects `forward/left/right/stop` or arrow symbols (`↑←→`). The model rarely predicts `stop` or `forward` in training-free mode (likely due to checkpoint/domain gap).
4. **Performance gap vs StreamVLN reference**: Reference reports ~33.6% baseline accuracy on TF240 with StreamVLN; InternNav baseline achieves ~28% on the same data. This is expected because InternVLA-N1-System2 is a different checkpoint with a different prompt format.

---

## Sample Results (20-episode test)

| Config | Acc | Invalid | TokenRed | FPS |
|--------|-----|---------|----------|-----|
| Baseline | 29.36% | 16.03% | 0.00% | 4.45 |
| FastVid keep=0.3 | 29.36% | 16.76% | 70.37% | 5.12 |
| FastVid keep=0.5 | 30.45% | 14.79% | 50.61% | 4.90 |
| FastVid keep=0.7 | 30.75% | 14.71% | 29.63% | 4.77 |

**Observations:**
- Training-free FastVid preserves baseline accuracy within ±1.4pp.
- Token reduction scales with keep ratio as expected.
- FastVid 0.3 achieves the highest FPS boost (~15%) due to the smallest per-step token count.
- Invalid rate remains stable (~15–17%) across all configurations.

*Note: 20 episodes (~1,366 actions) is a smoke test. Run on 200+ episodes for publication-grade metrics.*
