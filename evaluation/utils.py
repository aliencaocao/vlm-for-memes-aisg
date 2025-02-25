import gc
import os
import sys
import time
from dataclasses import dataclass
from subprocess import Popen, call
from threading import Thread
from typing import Any, Optional

import requests
import torch


def create_batch(samples: list, batch_size: int = 32) -> list:
    batches = []
    for i in range(0, len(samples), batch_size):
        batches.append(samples[i:i + batch_size])
    return batches


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args) -> object:
        Thread.join(self, *args)
        return self._return


def prepare_torch_deterministic():
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    torch.set_grad_enabled(False)
    torch.inference_mode(True)


def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()


def cleanup(proc_name: str = 'uvicorn'):
    call("ps ax | grep '" + proc_name + "' | awk -F ' ' '{print $1}' | xargs kill -9", shell=True, stdout=sys.stderr)


@dataclass
class InferenceServer:
    name: str
    fname: str
    port: int
    start_timeout: int
    _first_request: bool = True
    session: Optional[requests.Session] = None
    proc: Optional[Popen] = None


class InferenceServerManager:
    servers = {
        'LLaVA': InferenceServer(
            name='LLaVA',
            fname='api_llava',
            port=8080,
            start_timeout=999999,
        ),
        'qwen': InferenceServer(
            name='Qwen',
            fname='api_qwen',
            port=8079,
            start_timeout=999999,
        ),
        'OCR': InferenceServer(
            name='OCR',
            fname='api_ocr',
            port=8081,
            start_timeout=60,
        ),
        'Translation': InferenceServer(
            name='Translation',
            fname='api_translation',
            port=8082,
            start_timeout=60,
        ),
        'Benchmark': InferenceServer(
            name='Benchmark',
            fname='api_benchmark',
            port=8083,
            start_timeout=10,
        )
    }

    def stop_server(self, server_name: str) -> bool:
        server = self.servers[server_name]
        if server.proc is not None and server.session is not None:
            server.session.close()
            server.session = None

            server.proc.kill()
            # cleanup(server.fname)
            server.proc = None

            print(f'[{server_name}] Server killed', file=sys.stderr)
            return True
        return False

    def start_server(self, server_name: str) -> bool:
        server = self.servers[server_name]
        if server_name == 'qwen':  # doing same node multi gpu cycling using eval manager, so need cycle ports
            server.port += int(os.getenv('CUDA_VISIBLE_DEVICES', 0))
        if server.proc is not None:
            print(f'[{server.name}] Server already running', file=sys.stderr)
            return True

        start_time = time.perf_counter()
        print(f'[{server.name}] Loading {server.name} server', file=sys.stderr)
        proc = Popen(
            [f'{sys.executable} -m uvicorn {server.fname}:app --host 127.0.0.1 --port {str(server.port)} --log-level warning'],
            start_new_session=True, shell=True, stdout=sys.stderr
        )

        # Wait for server to start
        while time.perf_counter() - start_time < server.start_timeout:
            try:
                requests.get(f'http://localhost:{str(server.port)}/health').json()
                print(f'[{server.name}] Server loaded in {time.perf_counter() - start_time:.4f}s', file=sys.stderr)
                server.proc = proc
                server.session = requests.Session()
                server._first_request = True
                return True
            except requests.exceptions.ConnectionError:
                time.sleep(1)

        print(f'[{server.name}] Failed to start', file=sys.stderr)
        proc.kill()
        # cleanup()
        raise TimeoutError(f'Failed to start {server.name} server')

    def request(self, server_name: str, json, req_timeout_first: Optional[float] = None, req_timeout: Optional[float] = None) -> Any:
        server = self.servers[server_name]
        session = server.session
        port = server.port
        timeout = req_timeout_first if server._first_request and req_timeout_first else req_timeout
        if session is None:
            return None

        try:
            server._first_request = False
            return session.post(
                url=f'http://localhost:{port}/inference',
                json=json,
                timeout=timeout,
            ).json()
        except requests.exceptions.Timeout:
            print(f'[{server_name}] Timeout during inference', file=sys.stderr)
            return None
