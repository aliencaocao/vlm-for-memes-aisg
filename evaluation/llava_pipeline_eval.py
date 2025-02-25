import atexit
import signal
import sys
import os
import time
from base64 import b64decode
from typing import Union

import requests
import transformers
import orjson
from tqdm import tqdm
from utils import create_batch, empty_cache, cleanup, InferenceServerManager

signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))  # to trigger cleanup on SIGTERM (ctrl c)
#atexit.register(cleanup)

with open('llava_requests.json', 'rb') as f:
    llava_requests = orjson.loads(f.read())


output_fname = 'llava_pipeline_output.json'
if os.path.isfile(output_fname) and os.path.getsize(output_fname) > 0:  # already run some, resume
    with open(output_fname, 'rb') as f:
        output = orjson.loads(f.read())
else:
    output: dict[str, dict[str, Union[str, float]]] = {}

print(f'Resuming {len(output)}/{len(llava_requests)}')
llava_requests = [x for x in llava_requests if x['im_path'] not in output]
if not llava_requests:
    exit(0)


class ForceRetryException(Exception):
    pass


if __name__ == '__main__':
    llava_max_soft_retry_per_batch = 2  # max 2 retries before restarting runtime
    llava_max_hard_retry_per_batch = 1  # max 1 retries after restarting runtime
    llava_batch_size = 1  # make sure same as in api_llava_exllama.py!

    print('Imports complete', file=sys.stderr)
    transformers.logging.set_verbosity_error()
    server_manager = InferenceServerManager()

    server_manager.start_server('LLaVA')
    TOTAL_TOKENS = 0
    all_requests = []
    for request_body in tqdm(create_batch(llava_requests, llava_batch_size), desc='LLaVA Inference'):
        img_soft_retry_counter = 0
        img_hard_retry_counter = 0
        res = []

        while img_hard_retry_counter <= llava_max_hard_retry_per_batch:
            try:
                _res: list[dict] = server_manager.request('LLaVA', request_body)
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
                        res += [x for x in _res if not success[x['im_path']]]  # save the failed ones after max retries
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
                #
                # tqdm.write('[LLaVA] Restarting server', file=sys.stderr)
                # server_manager.stop_server('LLaVA')
                # empty_cache()
                # server_manager.start_server('LLaVA')

        if len(res) != len(request_body):
            tqdm.write(f'[LLaVA] Inference failed for {len(request_body) - len(res)} samples, skipping', file=sys.stderr)
        for r in res:
            output[r['im_path']] = {'score': r['score'], 'tokens': r['tokens']}

        # save every batch to prevent data loss
        with open(output_fname, 'wb') as f:
            f.write(orjson.dumps(output))

    print('LLaVA total tokens:', TOTAL_TOKENS, file=sys.stderr)

    server_manager.stop_server('LLaVA')
