import torch
import numpy as np

# Patch torch.from_numpy for buggy numpy/torch build
_original_from_numpy = torch.from_numpy
def _patched_from_numpy(obj):
    try:
        return _original_from_numpy(obj)
    except TypeError:
        return torch.tensor(obj)
torch.from_numpy = _patched_from_numpy

from transformers import AutoProcessor
from internnav.model.basemodel.qwen25vl_fastvid import Qwen2_5_VLForConditionalGenerationFastVid
from internnav.model.compression.fastvid import FastVidConfig
from scripts.train.qwenvl_train.offline_r2r_multiframe_lora_sft_eval import build_samples, OfflineR2RSFTDataset, QwenVLDataCollator

# Load 1 sample
data_root = '/home/ubuntu/dataset/VLN-Trajectory-Data/R2R'
samples = build_samples(data_root, max_samples=1, seed=42, num_history=3)
print('Loaded', len(samples), 'samples')

processor = AutoProcessor.from_pretrained('checkpoints/InternVLA-N1-System2')
pad_token_id = processor.tokenizer.pad_token_id or 151643

# Build dataset
dataset = OfflineR2RSFTDataset(samples, processor, max_seq_length=2048, num_history=3)
collator = QwenVLDataCollator(pad_token_id=pad_token_id)

# Get a batch
batch = collator([dataset[0]])
print('batch keys:', batch.keys())
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f'{k}: shape={v.shape}, dtype={v.dtype}')

# Count image tokens
image_token_id = 151655
for i in range(batch['input_ids'].shape[0]):
    n = (batch['input_ids'][i] == image_token_id).sum().item()
    print(f'Sample {i}: image_token count = {n}')

# Count total pixels from grid
grid = batch['image_grid_thw']
total_pixels = (grid[:,0] * grid[:,1] * grid[:,2]).sum().item()
print(f'Total pixels from grid: {total_pixels}')
print(f'Expected visual tokens (//4): {total_pixels // 4}')

# Load FastVid model
fastvid_cfg = FastVidConfig(enabled=True, retention_ratio=0.5)
model = Qwen2_5_VLForConditionalGenerationFastVid.from_pretrained(
    'checkpoints/InternVLA-N1-System2',
    fastvid_config=fastvid_cfg,
    torch_dtype=torch.bfloat16,
    device_map='cuda:0',
    attn_implementation='sdpa'
)

# Move batch to device
device = torch.device('cuda:0')
for k in ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'labels']:
    if k in batch:
        batch[k] = batch[k].to(device)

# DEBUG: manually run compression steps
print('\n--- Manual compression debug ---')
image_embeds = model.visual(batch['pixel_values'].type(model.visual.dtype), grid_thw=batch['image_grid_thw'])
print(f'image_embeds shape: {image_embeds.shape}')

from internnav.model.compression.fastvid import compress_frames_by_grid

# Try compress_frames_by_grid directly
compressed, sizes = compress_frames_by_grid(image_embeds, batch['image_grid_thw'], fastvid_cfg)
print(f'compressed shape: {compressed.shape}')
print(f'sizes: {sizes}')

# Try _compress_for_batch
compressed_embeds, new_input_ids, new_attention_mask, new_labels = model._compress_for_batch(
    image_embeds=image_embeds,
    image_grid_thw=batch['image_grid_thw'],
    input_ids=batch['input_ids'],
    attention_mask=batch['attention_mask'],
    labels=batch['labels'],
)
print(f'compressed_embeds shape: {compressed_embeds.shape}')
print(f'new_input_ids shape: {new_input_ids.shape}')
print(f'new image_token count: {(new_input_ids == image_token_id).sum().item()}')

print('\n--- Forward debug ---')
with torch.no_grad():
    out = model(**batch)
print('Forward success!')
