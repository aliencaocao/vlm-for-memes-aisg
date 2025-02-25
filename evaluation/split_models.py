import safetensors
from safetensors.torch import save_file
import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

MODEL_NAME = 'models/llava-v1.6-mistral-7b-hf'
processor = LlavaNextProcessor.from_pretrained(MODEL_NAME)
model = LlavaNextForConditionalGeneration.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True, torch_dtype=torch.float16)
for param in model.parameters(): param.requires_grad = False
model.language_model.save_pretrained(MODEL_NAME + '-llm', safe_serialization=True)
processor.save_pretrained(MODEL_NAME + '-llm')
model.vision_tower.save_pretrained(MODEL_NAME + '-CLIP', safe_serialization=True)
safetensors.torch.save_model(model.multi_modal_projector, MODEL_NAME + '-CLIP/' + MODEL_NAME + '-PROJ.safetensors')
newline = model.image_newline.data
save_file({'image_newline': newline}, MODEL_NAME + '-CLIP/newline.safetensors')
