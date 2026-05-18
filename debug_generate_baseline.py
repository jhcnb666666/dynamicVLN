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

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from scripts.train.qwenvl_train.offline_r2r_multiframe_lora_sft_eval import build_samples, OfflineR2RSFTDataset, QwenVLDataCollator

# Load 1 sample
data_root = '/home/ubuntu/dataset/VLN-Trajectory-Data/R2R'
samples = build_samples(data_root, max_samples=1, seed=42, num_history=3)
processor = AutoProcessor.from_pretrained('checkpoints/InternVLA-N1-System2')
pad_token_id = processor.tokenizer.pad_token_id or 151643

dataset = OfflineR2RSFTDataset(samples, processor, max_seq_length=2048, num_history=3)
model_input = dataset[0]

labels = model_input['labels']
action_mask = labels != -100
prompt_end = action_mask.nonzero(as_tuple=True)[0][0].item()
prompt_input_ids = model_input['input_ids'][:prompt_end].unsqueeze(0)
prompt_attention_mask = model_input['attention_mask'][:prompt_end].unsqueeze(0)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'checkpoints/InternVLA-N1-System2',
    torch_dtype=torch.bfloat16,
    device_map='cuda:0',
    attn_implementation='sdpa'
)

device = torch.device('cuda:0')
gen_inputs = {
    'input_ids': prompt_input_ids.to(device),
    'attention_mask': prompt_attention_mask.to(device),
    'pixel_values': model_input['pixel_values'].to(device),
    'image_grid_thw': model_input['image_grid_thw'].to(device),
}

print('--- Baseline Generating ---')
with torch.no_grad():
    generated_ids = model.generate(
        **gen_inputs,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=pad_token_id,
    )
print('Generated ids shape:', generated_ids.shape)
