import torch
import numpy as np
_original_from_numpy = torch.from_numpy
def _patched_from_numpy(obj):
    try:
        return _original_from_numpy(obj)
    except TypeError:
        return torch.tensor(obj)
torch.from_numpy = _patched_from_numpy

from transformers import AutoProcessor
from scripts.train.qwenvl_train.offline_r2r_multiframe_lora_sft_eval import build_samples, OfflineR2RSFTDataset

data_root = '/home/ubuntu/dataset/VLN-Trajectory-Data/R2R'
samples = build_samples(data_root, max_samples=1, seed=42, num_history=3)
processor = AutoProcessor.from_pretrained('checkpoints/InternVLA-N1-System2')
dataset = OfflineR2RSFTDataset(samples, processor, max_seq_length=2048, num_history=3)

item = dataset[0]
print('input_ids shape:', item['input_ids'].shape)
print('labels shape:', item['labels'].shape)

# Count non -100 labels
non_mask = (item['labels'] != -100).sum().item()
print('non -100 labels:', non_mask)

# Show first few non -100 positions and their decoded text
non_mask_positions = (item['labels'] != -100).nonzero(as_tuple=True)[0]
if len(non_mask_positions) > 0:
    print('first non -100 pos:', non_mask_positions[0].item())
    print('last non -100 pos:', non_mask_positions[-1].item())
    
# Decode non -100 tokens
non_mask_token_ids = item['labels'][item['labels'] != -100]
text = processor.tokenizer.decode(non_mask_token_ids, skip_special_tokens=False)
print('decoded non -100 text:', repr(text))

# Verify prompt alignment
prompt_text = processor.apply_chat_template(
    [{"role": "user", "content": [{"type": "image"} for _ in range(4)] + [{"type": "text", "text": "test"}]}],
    tokenize=False,
    add_generation_prompt=True,
)
prompt_inputs = processor(text=[prompt_text], images=[Image.new('RGB', (64,64)) for _ in range(4)], return_tensors="pt")
print('prompt_ids length:', prompt_inputs['input_ids'].shape[1])
