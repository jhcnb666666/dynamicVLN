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
from PIL import Image

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'checkpoints/InternVLA-N1-System2',
    torch_dtype='auto',
    device_map='cpu'
)
processor = AutoProcessor.from_pretrained('checkpoints/InternVLA-N1-System2')

img = Image.new('RGB', (640, 480), color=(100, 150, 200))
inputs = processor(text='<image> Describe the image.', images=[img], return_tensors='pt')

print('pixel_values shape:', inputs['pixel_values'].shape)
print('image_grid_thw:', inputs['image_grid_thw'])
print('input_ids shape:', inputs['input_ids'].shape)

image_embeds = model.visual(inputs['pixel_values'], grid_thw=inputs['image_grid_thw'])
print('image_embeds shape:', image_embeds.shape)

image_token_id = 151655
n_image_tokens = (inputs['input_ids'] == image_token_id).sum().item()
print('n_image_tokens in input_ids:', n_image_tokens)
print('match:', image_embeds.shape[0] == n_image_tokens)

t, h, w = inputs['image_grid_thw'][0].tolist()
print(f'grid_thw: t={t}, h={h}, w={w}')
print(f't*h*w = {t*h*w}')
print(f't*h*w//4 = {t*h*w//4}')
