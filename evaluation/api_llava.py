import os
import sys
import traceback
from base64 import b64decode
from io import BytesIO
from typing import Union
from contextlib import contextmanager
import ast

import fastapi
import orjson
import torch
import transformers
from PIL import Image
from pydantic import BaseModel
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers.generation import GenerateDecoderOnlyOutput

from utils import prepare_torch_deterministic
from llava.mm_utils import resize_and_pad_image, select_best_resolution
from llava.utils import disable_torch_init

model_path = os.environ.get('MODEL_PATH', None) or '../LLaVA/checkpoints/llava-1.6-mistral-7b-r128-a256-lr1e4-with-sg-merged-hf'
transformers.logging.set_verbosity_error()
min_p = 0.1
temperature = 0.9
max_new_tokens = 512

app = fastapi.FastAPI()


class InferenceRequest(BaseModel):
    im_path: str
    image_b64: str
    ocr_text: str = ''
    ocr_lang: str = ''
    sample: bool = False


prepare_torch_deterministic()
disable_torch_init()

processor = LlavaNextProcessor.from_pretrained(model_path, local_files_only=False)
model = LlavaNextForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, local_files_only=False, torch_dtype=torch.float16)
model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# model.config.image_grid_pinpoints = ast.literal_eval(model.config.image_grid_pinpoints)  # needed for llava-hf/ models
prompt_prefix = "[INST] <image>\n"
prompt_postfix = " [/INST]"
# model._setup_cache(StaticCache, max_batch_size=1, max_cache_len=max_new_tokens)
# torch.compile(model, mode="reduce-overhead", fullgraph=True)

yes_tokens = [i for i in range(processor.tokenizer.vocab_size) if processor.tokenizer.decode(i).lower().strip() == 'yes']  # all possible yes tokens
yes_tokens_tensor = torch.tensor(yes_tokens, device=model.device, dtype=torch.int64)
no_tokens = [i for i in range(processor.tokenizer.vocab_size) if processor.tokenizer.decode(i).lower().strip() == 'no']  # all possible no tokens
no_tokens_tensor = torch.tensor(no_tokens, device=model.device, dtype=torch.int64)
yes_no_tokens = yes_tokens + no_tokens
yes_no_tokens_tensor = torch.tensor(yes_no_tokens, device=model.device, dtype=torch.int64)
ocr_langs = {'cmn': 'Chinese', 'zsm': 'Malay', 'tam': 'Tamil'}


def pprint(x: str):
    print(f'[LLaVA API] {x}', file=sys.stderr)


def process_request(request: InferenceRequest) -> tuple[str, Image.Image, str, bool]:
    with BytesIO(b64decode(request.image_b64)) as f:
        image = Image.open(f).convert('RGB')

    im_path = request.im_path
    sample = request.sample
    ocr_text = request.ocr_text
    ocr_lang = request.ocr_lang
    ocr_prompt = ''
    if ocr_text:
        if ocr_lang:
            ocr_lang = ocr_langs[ocr_lang]
            ocr_prompt = f'The text translated from {ocr_lang} says:'
        else:
            ocr_prompt = 'The text says:'
        ocr_prompt += f'\n{ocr_text}'

    prompt = (
        f'You are a professional content moderator. Analyze this meme in the context of Singapore society. {ocr_prompt}\n\n'
        f'Output a YAML in English using tab for indentation that contains description, the victim groups and methods of attack if any. Think through the information you just provided and label the meme as harmful using "Yes" or "No".'
        f'Do not include any other explanation outside the YAML.'
    )

    return im_path, image, prompt_prefix + prompt + prompt_postfix, sample


def process_responses(im_paths: list[str], model_output: GenerateDecoderOnlyOutput) -> list[dict[str, Union[str, bool, float]]]:
    global TOTAL_TOKENS
    fn_output = []
    logit_len = len(model_output.scores)

    for idx, im_path in zip(range(len(model_output.sequences)), im_paths):
        sequences_list = model_output.sequences[idx][-logit_len:].tolist()  # different in official impl that this includes input ids so need to take only the new tokens
        # pprint(processor.tokenizer.decode(sequences_list, skip_special_tokens=True))
        # pprint(sequences_list)
        # pprint(f'token count: {len(sequences_list)}')
        try:
            yes_no_pos = None
            for x in yes_no_tokens:
                try:
                    pos = sequences_list[::-1].index(x)
                    if yes_no_pos is None or pos < yes_no_pos:
                        yes_no_pos = pos
                except ValueError:
                    continue
            if yes_no_pos is None:
                pprint('Model output format error')
                fn_output.append({'im_path': im_path, 'success': False, 'score': 0.5, 'tokens': '', 'no_tokens': len(sequences_list), 'error': 'Model output format error'})
                continue

            yes_no_pos = len(sequences_list) - 1 - yes_no_pos
            yes_no_tok = sequences_list[yes_no_pos]
            softmax_scores = model_output.scores[yes_no_pos][idx].softmax(dim=0)
            label_bool = yes_no_tok in yes_tokens
            score = softmax_scores.index_select(-1, yes_tokens_tensor if label_bool else no_tokens_tensor).sum() / softmax_scores.index_select(0, yes_no_tokens_tensor).sum()
            if not label_bool:
                score = 1 - score
            fn_output.append({'im_path': im_path, 'success': True, 'score': score, 'no_tokens': len(sequences_list), 'tokens': processor.tokenizer.decode(sequences_list, skip_special_tokens=True)})

        except Exception:
            pprint('Error processing response')
            traceback.print_exc()
            fn_output.append({'im_path': im_path, 'success': False, 'score': 0.5, 'no_tokens': len(sequences_list), 'tokens': '', 'error': 'Error processing response'})  # default to 0.5 if there's an error, class will be 0 but no effect on AUROC

    return fn_output


def resize_images(images: list[Image.Image]) -> list[Image.Image]:
    res = []
    grid_pinpoints = model.config.image_grid_pinpoints
    for image in images:
        best_resolution = select_best_resolution(image.size, grid_pinpoints)
        image_padded = resize_and_pad_image(image, best_resolution)
        res.append(image_padded)
    return res


@contextmanager
def sdap_backend():
    if float('.'.join(torch.__version__.split('.')[:2])) >= 2.4:
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
            yield
    else:
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
            yield


@app.post('/inference')
def inference(requests: list[InferenceRequest]) -> list[dict[str, Union[str, bool, float]]]:
    processed = [process_request(x) for x in requests]
    sample_batches = []
    no_sample_batches = []

    for im_path, images, prompts, sample in processed:
        if sample:
            sample_batches.append((im_path, images, prompts))
        else:
            no_sample_batches.append((im_path, images, prompts))

    def infer_batch(im_paths: list[str], images: list[Image.Image], prompts: list[str], sample: bool) -> list[dict[str, Union[str, bool, float]]]:
        try:
            images = resize_images(images)
            inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(model.device)
            with sdap_backend():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    do_sample=False,  # turn off here to align with best eval results
                    temperature=temperature if sample else None,
                    min_p=min_p if sample else None,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
        except Exception as e:
            pprint('Error processing batch')
            traceback.print_exc()
            return [{'im_path': im_path, 'success': False, 'score': 0.5, 'no_tokens': 0, 'error': 'Error processing batch'} for im_path in im_paths]
        return process_responses(im_paths, output)

    results = []
    if sample_batches:
        results.extend(infer_batch(*zip(*sample_batches), sample=True))
    if no_sample_batches:
        results.extend(infer_batch(*zip(*no_sample_batches), sample=False))
    return results


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'healthy'}
