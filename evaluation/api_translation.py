import sys
from typing import Literal

import fastapi
from pydantic import BaseModel

from translation import SeamlessM4TTranslation
from utils import prepare_torch_deterministic

app = fastapi.FastAPI()


class TranslationInferenceRequest(BaseModel):
    prompts: list[str]
    src_lang: Literal['cmn', 'zsm', 'tam']


prepare_torch_deterministic()
translator = SeamlessM4TTranslation()


def pprint(x: str):
    print(f'[Translation API] {x}', file=sys.stderr)


@app.post('/inference')
def inference_translation(requests: TranslationInferenceRequest) -> list[str]:
    results = translator.translate(texts=requests.prompts, src_lang=requests.src_lang)
    return results


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'healthy'}
