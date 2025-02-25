import sys

import fastapi
from pydantic import BaseModel

from ocr import OCR, OCRResult

app = fastapi.FastAPI()


class OCRInferenceRequest(BaseModel):
    im_paths: list[str]


ocr = OCR()


def pprint_ocr(x: str):
    print(f'[OCR API] {x}', file=sys.stderr)


@app.post('/inference')
def inference_ocr(requests: OCRInferenceRequest) -> list[OCRResult]:
    global ocr
    ocr_result = ocr.run_batch(requests.im_paths)
    return ocr_result


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'healthy'}
