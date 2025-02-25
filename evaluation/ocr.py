import logging
import os
import sys
import time
import traceback
from collections import Counter
from functools import partial
from typing import Literal, Optional

import cv2
import numpy as np
import paddle
import paddleclas
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException
from paddleclas.deploy.python.postprocess import build_postprocess as paddle_build_postprocess
from paddleclas.deploy.python.preprocess import create_operators as paddle_create_operators
from paddleocr import PaddleOCR
from pydantic import BaseModel
from tqdm import tqdm
import regex

logging.getLogger("ppcls").disabled = True
logging.getLogger("ppocr").disabled = True


class OCRResult(BaseModel):
    text: str
    lang: Literal['latin', 'chinese_cht', 'ta', 'en']
    file: str


def sorted_boxes(dt_boxes):
    """
    Copied from https://github.com/PaddlePaddle/PaddleOCR/blob/52cf1e1bdd2a4422bd6232e924bf26b85a47afda/tools/infer/predict_system.py#L123 due to bug in multiprocessing used by SGLang that causes importing it to fail
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def get_rotate_crop_image(img, points):
    """
    Copied from https://github.com/PaddlePaddle/PaddleOCR/blob/52cf1e1bdd2a4422bd6232e924bf26b85a47afda/tools/infer/utility.py#L602 due to bug in multiprocessing used by SGLang that causes importing it to fail
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


class OCR:
    def __init__(self, models_path: str = '/paddle', paddleclas_batch_size: int = 1024, paddleocr_batch_size: int = 1024, use_gpu: bool = True) -> None:
        print('[OCR] Loading models', file=sys.stderr)
        self.paddleclas_batch_size = paddleclas_batch_size
        self.paddleocr_batch_size = paddleocr_batch_size

        self.lang_classifier = paddleclas.PaddleClas(  # cannot specify model_name here else it forces to download model in root dir
            inference_model_dir=os.path.join(models_path, 'language_classification_infer'),
            use_gpu=use_gpu,
        )
        # override the config to correct the labels to languages
        lang_classifier_config = paddleclas.paddleclas.init_config(
            model_type='pulc',
            model_name='language_classification',
            inference_model_dir=os.path.join(models_path, 'language_classification_infer'),
            use_gpu=use_gpu,  # cannot use fp16 as it only works with TRT
            batch_size=paddleclas_batch_size
        )
        self.lang_classifier._config = lang_classifier_config
        self.lang_classifier.predictor.preprocess_ops = paddle_create_operators(lang_classifier_config["PreProcess"]["transform_ops"])
        self.lang_classifier.predictor.postprocess = paddle_build_postprocess(lang_classifier_config["PostProcess"])

        # self.layout_engine = None

        paddle_ocr = partial(
            PaddleOCR,
            use_angle_cls=False,
            use_gpu=use_gpu,
            show_log=False,
            max_batch_size=self.paddleocr_batch_size,
            use_dilation=True,  # improves accuracy
            det_db_score_mode='slow',  # improves accuracy
            rec_batch_num=self.paddleocr_batch_size,
        )
        paddle_langs = (
            # clas_lang, ocr_lang, det_model, rec_model
            ('en', "en", "en_PP-OCRv3_det_infer", "en_PP-OCRv4_rec_infer"),
            ('chinese_cht', "ch", "ch_PP-OCRv4_det_infer", "ch_PP-OCRv4_rec_infer"),
            ('latin', "ms", "Multilingual_PP-OCRv3_det_infer", "latin_PP-OCRv3_rec_infer"),
            ('ta', "ta", "Multilingual_PP-OCRv3_det_infer", "ta_PP-OCRv4_rec_infer")
        )
        self.ocr_models = {
            paddleclas_lang: paddle_ocr(
                lang=paddleocr_lang,
                det_model_dir=os.path.join(models_path, det_model),
                rec_model_dir=os.path.join(models_path, rec_model),
            ) for paddleclas_lang, paddleocr_lang, det_model, rec_model in paddle_langs
        }
        self.ocr_models['initial_ocr'] = paddle_ocr(
            det_model_dir=os.path.join(models_path, 'Multilingual_PP-OCRv3_det_infer'),
            rec_model_dir=os.path.join(models_path, 'en_PP-OCRv4_rec_infer'),
        )
        print('[OCR] Models loaded', file=sys.stderr)

        self.try_count: int = 1

    def get_boxes(self, image_path: str) -> Optional[list[np.ndarray]]:
        im = cv2.imread(image_path)
        boxes = self.ocr_models['initial_ocr'].text_detector(im)[0]
        if not len(boxes): return None
        boxes = sorted_boxes(boxes)  # ensure top left to bottom, left to right
        cropped_boxes = [get_rotate_crop_image(im, box) for box in boxes]
        return cropped_boxes

    def get_lang(self, cropped_boxes: list[np.ndarray]) -> Literal['latin', 'chinese_cht', 'ta', 'en']:
        lang_result = self.lang_classifier.predictor.predict(cropped_boxes)
        lang_result = [r['label_names'][0] for r in lang_result]
        predominant_lang = Counter(lang_result).most_common(1)[0][0]
        predominant_lang = predominant_lang if predominant_lang in self.ocr_models else 'en'  # Default to en
        return predominant_lang

    def get_layout(self, image_path: str, predominant_lang: Literal['latin', 'en', 'chinese_cht', 'ta']) -> list[dict]:
        pass

    def run_ocr(self, cropped_boxes: list[np.ndarray], predominant_lang: Literal['latin', 'en', 'chinese_cht', 'ta']) -> tuple[str, str]:
        final_ocr = self.ocr_models[predominant_lang]
        final_boxes = final_ocr.text_recognizer(cropped_boxes)[0]  # uses text_recognizer to force batching
        final_boxes = [i for i in final_boxes if i[1] > 0.5]  # remove boxes with low confidence
        final_txts = [box[0] for box in final_boxes]
        ocr_result = " ".join(final_txts)

        if predominant_lang == 'latin':
            p_en, p_id = 0., 0.
            try:
                for x in detect_langs(ocr_result):
                    if x.lang == 'en':
                        p_en = x.prob
                    elif x.lang == 'id':
                        p_id = x.prob
            except LangDetectException as e:
                print(f'[OCR] [TRY {self.try_count}] Error detecting language', file=sys.stderr)
                if self.try_count == 3:  # if it fails twice, default to en and proceed with OCR without raising
                    p_en = 1.  # default to en
                else:
                    raise e
            if p_en > p_id:
                return self.run_ocr(cropped_boxes, 'en')
        elif predominant_lang == 'chinese_cht':
            if regex.search(r'\p{Han}', ocr_result) is None:  # if it dont even contain chinese then means its a FP on chinese, and paddleocr like to FP latin for chinese, so re-run ocr with english
                return self.run_ocr(cropped_boxes, 'en')

        return ocr_result, predominant_lang

    def run_single(self, im_path: str) -> OCRResult:
        try:
            cropped_boxes = self.get_boxes(im_path)
            if cropped_boxes is None:
                return OCRResult(text='', lang='en', file=im_path)  # if no boxes are found, return empty string and default to en. On LLaVa side, if lang=en and ocr_text=empty, it will be not add ocr prompt
            predominant_lang = self.get_lang(cropped_boxes.copy())  # copy to avoid modifying the original list as lang_classifier does normalization & HWC to CHW on them in-place, affecting OCR next
            layout = self.get_layout(im_path, predominant_lang)
            ocr_result, predominant_lang = self.run_ocr(cropped_boxes, predominant_lang)
            self.try_count = 1
            return OCRResult(
                text=ocr_result,
                lang=predominant_lang,
                file=im_path
            )
        except Exception as e:
            r = OCRResult(text='', lang='en', file=im_path)
            while self.try_count < 3:  # retry up to 2 more times
                print(f'[OCR] [TRY {self.try_count}] Error processing image', file=sys.stderr)
                traceback.print_exc()
                self.try_count += 1
                try:
                    r = self.run_single(im_path)
                except Exception as e:
                    traceback.print_exc()
                else:
                    break
            self.try_count = 1
            return r

    def run_batch(self, batch: list[str]) -> list[OCRResult]:
        results = []
        for im_path in tqdm(batch, desc='OCR'):
            results.append(self.run_single(im_path))
        return results


if __name__ == '__main__':
    import gc

    image_paths = [x.strip() for x in sys.stdin if x.strip()]
    s = time.perf_counter()
    ocr = OCR(models_path='models/paddle')
    print(f'Init time: {time.perf_counter() - s:.4f}s')
    s = time.perf_counter()
    ocr_result = ocr.run_batch(image_paths)
    print(ocr_result)
    print(f'OCR time: {time.perf_counter() - s:.4f}s')
    del ocr
    gc.collect()
    paddle.device.cuda.empty_cache()
