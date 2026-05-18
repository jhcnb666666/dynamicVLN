import torch
import numpy as np
_original_from_numpy = torch.from_numpy
def _patched_from_numpy(obj):
    try:
        return _original_from_numpy(obj)
    except TypeError:
        return torch.tensor(obj)
torch.from_numpy = _patched_from_numpy

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image

processor = AutoProcessor.from_pretrained('checkpoints/InternVLA-N1-System2')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'checkpoints/InternVLA-N1-System2',
    torch_dtype=torch.bfloat16,
    device_map='cuda:0',
    attn_implementation='sdpa'
)

# Build a simple prompt
images = [Image.new('RGB', (64, 64), color=(100, 150, 200)) for _ in range(4)]
user_text = (
    "You are an autonomous navigation assistant. "
    "Instruction: Walk straight and turn left.\n"
    "These are your historical observations: <image> <image> <image>\n"
    "Look at the current RGB observation and predict the next action. "
    "Reply with exactly one word from: forward, left, right, stop."
)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": user_text}]},
]
prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print('Prompt text:')
print(repr(prompt_text))
print()

inputs = processor(text=[prompt_text], images=images, return_tensors="pt")
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        print(f'{k}: {v.shape}')

device = torch.device('cuda:0')
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)
pixel_values = inputs['pixel_values'].to(device)
image_grid_thw = inputs['image_grid_thw'].to(device)

# Manual decode 5 tokens
with torch.no_grad():
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        use_cache=True,
    )
    next_token = out.logits[:, -1, :].argmax(dim=-1)
    print('First generated token:', next_token.item(), repr(processor.tokenizer.decode([next_token.item()])))

    past_key_values = out.past_key_values
    for i in range(4):
        out = model(
            input_ids=next_token.unsqueeze(0),
            attention_mask=torch.ones_like(next_token.unsqueeze(0)),
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_token = out.logits[:, -1, :].argmax(dim=-1)
        print(f'Token {i+2}:', next_token.item(), repr(processor.tokenizer.decode([next_token.item()])))
        past_key_values = out.past_key_values
