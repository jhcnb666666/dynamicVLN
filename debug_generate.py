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
processor = AutoProcessor.from_pretrained('checkpoints/InternVLA-N1-System2')
pad_token_id = processor.tokenizer.pad_token_id or 151643

dataset = OfflineR2RSFTDataset(samples, processor, max_seq_length=2048, num_history=3)
model_input = dataset[0]

# Find action text from labels
labels = model_input['labels']
action_mask = labels != -100
action_token_ids = labels[action_mask].tolist()
action_text = processor.tokenizer.decode(action_token_ids, skip_special_tokens=True)
print('Action text:', repr(action_text))

# Build generate inputs: prompt only
prompt_end = action_mask.nonzero(as_tuple=True)[0][0].item()
prompt_input_ids = model_input['input_ids'][:prompt_end].unsqueeze(0)
prompt_attention_mask = model_input['attention_mask'][:prompt_end].unsqueeze(0)
pixel_values = model_input['pixel_values']
image_grid_thw = model_input['image_grid_thw']

print(f'prompt length: {prompt_end}')

# Load FastVid model
fastvid_cfg = FastVidConfig(enabled=True, retention_ratio=0.5)
model = Qwen2_5_VLForConditionalGenerationFastVid.from_pretrained(
    'checkpoints/InternVLA-N1-System2',
    fastvid_config=fastvid_cfg,
    torch_dtype=torch.bfloat16,
    device_map='cuda:0',
    attn_implementation='sdpa'
)

device = torch.device('cuda:0')
gen_inputs = {
    'input_ids': prompt_input_ids.to(device),
    'attention_mask': prompt_attention_mask.to(device),
    'pixel_values': pixel_values.to(device),
    'image_grid_thw': image_grid_thw.to(device),
}

print('\n--- Generating ---')
with torch.no_grad():
    generated_ids = model.generate(
        **gen_inputs,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=pad_token_id,
    )
print('Generated ids shape:', generated_ids.shape)
generated_text = processor.tokenizer.decode(generated_ids[0, prompt_end:], skip_special_tokens=True)
print('Generated text:', repr(generated_text))
