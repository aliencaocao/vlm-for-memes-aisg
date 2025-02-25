import time

start = time.perf_counter()
TIME_LIMIT_SECONDS = 60 * 60 * 3 - 600  # 3 hours with 10min headroom

import os

try:
    os.makedirs(os.environ['NUMBA_CACHE_DIR'], exist_ok=True)
    os.makedirs(os.environ['HF_HOME'], exist_ok=True)
    os.makedirs(os.environ['PYTORCH_KERNEL_CACHE_PATH'], exist_ok=True)
    os.makedirs(os.environ['XDG_CONFIG_HOME'], exist_ok=True)
    os.makedirs(os.environ['XDG_CACHE_HOME'], exist_ok=True)
except:
    pass

import atexit
import signal
import sys
from base64 import b64encode
from threading import Thread

import requests
import transformers
import orjson
from tqdm import tqdm
from utils import create_batch, empty_cache, cleanup, InferenceServerManager

signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))  # to trigger cleanup on SIGTERM (ctrl c)
atexit.register(cleanup)

with open('sg_acronyms.json', 'r') as f:
    abbrv_dict = orjson.loads(f.read())


def timing(timings: dict[str, float], ref: str, label: str, human_label: str):
    timings[label] = time.perf_counter()
    print(f'[TIMING] {human_label}: {timings[label] - timings[ref]:.4f}s', file=sys.stderr)


def expand_abbrv(s):
    words = s.split()
    for i, word in enumerate(words):
        words[i] = abbrv_dict.get(word, word)
    return ' '.join(words)


class ForceRetryException(Exception):
    pass


if __name__ == '__main__':
    llava_max_soft_retry_per_batch = 2  # max 2 retries before restarting runtime
    llava_max_hard_retry_per_batch = 1  # max 1 retries after restarting runtime
    translation_batch_size = 24
    llava_batch_size = 1  # make sure same as in api_llava_exllama.py!

    print('Imports complete', file=sys.stderr)
    transformers.logging.set_verbosity_error()
    timings = {'start': time.perf_counter()}
    server_manager = InferenceServerManager()

    server_manager.start_server('OCR')
    translator_loader_thread = Thread(target=server_manager.start_server, args=('Translation',))
    translator_loader_thread.start()
    image_paths = [x.strip() for x in sys.stdin if x.strip()]
    ds_size = len(image_paths)
    # fill with default first
    ocr_result: list[dict] = [{'text': '', 'lang': 'en', 'file': x} for x in image_paths]
    _ocr_result = server_manager.request('OCR', {'im_paths': image_paths}, req_timeout_first=1 * ds_size, req_timeout=0.5 * ds_size)
    if not _ocr_result:
        print('OCR failed to return any result, using defaults', file=sys.stderr)
    else:
        ocr_result = _ocr_result
    server_manager.stop_server('OCR')
    timing(timings, 'start', 'ocr', 'OCR')

    ocr_translate = {  # convert between PaddleClas langauge codes and SeamlessM4T language codes
        'latin': 'zsm',
        'ta': 'tam',
        'chinese_cht': 'cmn',
        'en': ''  # not translated or error or no text detected
    }
    # llava_input: dict[str, tuple[str, str]] = {x: ('', '') for x in image_paths}  # fake input for testing
    llava_input: dict[str, tuple[str, str]] = {x['file']: (ocr_translate[x['lang']], expand_abbrv(x['text'])) for x in ocr_result}  # filename: [langauge, ocr_text]

    translator_loader_thread.join()
    for lang in ['chinese_cht', 'latin', 'ta']:
        translate_input = [b for b in create_batch([x for x in ocr_result if x['lang'] == lang], batch_size=translation_batch_size)]
        for batch in tqdm(translate_input, desc=f'Translation {lang}'):
            error_results = ['Translation failed. Please ignore this translation and read the text from the image directly.'] * len(batch)  # default in case fail
            translation_retry_counter = 0
            while translation_retry_counter <= 1:  # only allow 1 retry
                results: list[str] = server_manager.request('Translation', {'prompts': [x['text'] for x in batch], 'src_lang': ocr_translate[lang]}, req_timeout_first=10 * translation_batch_size, req_timeout=1 * translation_batch_size)
                if results == error_results:
                    translation_retry_counter += 1
                    if translation_retry_counter <= 1:
                        print('Translation has failed, SOFT retrying', file=sys.stderr)
                    else:
                        print('Translation has failed after 1 retry, giving up', file=sys.stderr)
                else:
                    break
            for x, r in zip(batch, results):
                llava_input[x['file']] = (ocr_translate[lang], r.replace('<unk>', ''))  # remove <unk> tokens as it causes bug in LLAVA image encoding https://github.com/huggingface/transformers/issues/29835
    server_manager.stop_server('Translation')
    timing(timings, 'ocr', 'translation', 'Translation')
    empty_cache()

    output: dict[str, float] = {path: 0.5 for path in image_paths}  # initialize all result to 0.5 first in case some got missed out
    batches = create_batch(list(llava_input.items()), batch_size=llava_batch_size)

    # only run here so highest chance to keep the result
    server_manager.start_server('Benchmark')
    server_manager.request('Benchmark', dict(), req_timeout_first=5, req_timeout=5)
    server_manager.stop_server('Benchmark')
    empty_cache()

    server_manager.start_server('LLaVA')
    TOTAL_TOKENS = 0
    all_requests = []
    for batch in tqdm(batches, desc='LLaVA Inference'):
        if time.perf_counter() - start > TIME_LIMIT_SECONDS:
            print('TIMEOUT', file=sys.stderr)
            break
        request_body = []
        for im_path, (ocr_lang, ocr_text) in batch:
            with open(im_path, 'rb') as f:
                im_b64 = b64encode(f.read()).decode('ascii')
            req = {
                'im_path': im_path,
                'image_b64': im_b64,
                'ocr_text': ocr_text,
                'ocr_lang': ocr_lang,
                'sample': False
            }
            request_body.append(req)
        all_requests.extend(request_body)

        img_soft_retry_counter = 0
        img_hard_retry_counter = 0
        res = []

        while img_hard_retry_counter <= llava_max_hard_retry_per_batch:
            try:
                _res: list[dict] = server_manager.request('LLaVA', request_body, req_timeout_first=60 * llava_batch_size, req_timeout=30 * llava_batch_size)
                if _res:
                    TOTAL_TOKENS += sum(int(x['no_tokens']) for x in _res)
                    success = {x['im_path']: x['success'] for x in _res}
                    res += [x for x in _res if x['success']]  # save the successful ones only
                else:  # it can return None
                    success = {x['im_path']: False for x in request_body}
                if all(success.values()):
                    initial_request = False
                    break
                else:  # retry failed samples and change sample to be True, successful ones will not be retried
                    request_body = [x for x in request_body if not success[x['im_path']]]
                    request_body = [{**x, 'sample': True} for x in request_body]
                    if img_hard_retry_counter >= llava_max_hard_retry_per_batch:
                        tqdm.write(f'[LLaVA] Inference failed for sample after {llava_max_hard_retry_per_batch} HARD retries', file=sys.stderr)
                        break
                    else:
                        if img_soft_retry_counter < llava_max_soft_retry_per_batch:
                            tqdm.write('[LLaVA] Inference failed for sample, SOFT retrying', file=sys.stderr)
                            img_soft_retry_counter += 1
                        else:
                            tqdm.write(f'[LLaVA] Inference failed for sample after {llava_max_soft_retry_per_batch} SOFT retries, HARD retrying', file=sys.stderr)
                            raise ForceRetryException()

            except (requests.exceptions.Timeout, ForceRetryException) as e:
                if isinstance(e, requests.exceptions.Timeout):
                    tqdm.write(f'[LLaVA] Inference timeout for batch size {llava_batch_size}, HARD retrying', file=sys.stderr)
                img_soft_retry_counter = 0
                img_hard_retry_counter += 1

                tqdm.write('[LLaVA] Restarting server', file=sys.stderr)
                server_manager.stop_server('LLaVA')
                empty_cache()
                server_manager.start_server('LLaVA')

        if len(res) != len(batch):
            tqdm.write(f'[LLaVA] Inference failed for {len(batch) - len(res)} samples, skipping', file=sys.stderr)
        for r in res: output[r['im_path']] = r['score']

    timing(timings, 'translation', 'llava', 'LLaVA Inference')
    print('LLaVA total tokens:', TOTAL_TOKENS, file=sys.stderr)
    # save llava requests
    with open('llava_requests.json', 'wb') as f:
        f.write(orjson.dumps(all_requests))

    for file_path in image_paths:
        score = output[file_path]
        print(f'{score:.4f}\t{int(score > .5)}')

    timing(timings, 'start', 'end', 'Total Time')
    server_manager.stop_server('LLaVA')
