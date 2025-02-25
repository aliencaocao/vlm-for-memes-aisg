import os
import sys
import traceback
from typing import Literal

import torch
from fairseq2.assets import InProcAssetMetadataProvider, asset_store
from seamless_communication.inference import Modality, Translator

from ocr import OCRResult

asset_store.env_resolvers.clear()
asset_store.env_resolvers.append(lambda: "demo")


class SeamlessM4TTranslation:
    def __init__(self, model_path: str = 'seamless-m4t-v2-large', use_cuda: bool = True, compile: bool = True):
        print('[TRANSLATION] Loading model', file=sys.stderr)
        CHECKPOINTS_PATH = os.path.join(os.getcwd(), model_path)
        demo_metadata = [
            {
                "name": "seamlessM4T_v2_large@demo",
                "checkpoint": f"file://{CHECKPOINTS_PATH}/seamlessM4T_v2_large-fp16.pt",
            }]
        asset_store.metadata_providers.append(InProcAssetMetadataProvider(demo_metadata))
        self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        self.model = Translator("seamlessM4T_v2_large",
                                vocoder_name_or_card=None,
                                device=torch.device(self.device),
                                dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                                apply_mintox=False,
                                input_modality=Modality.TEXT,
                                output_modality=Modality.TEXT)

        print('[TRANSLATION] Model loaded', file=sys.stderr)
        if compile: self.compile()

    def compile(self):
        self.model.model = torch.compile(self.model.model, fullgraph=True, mode='max-autotune-no-cudagraphs')
        print(f'[TRANSLATION] Will compile model', file=sys.stderr)

    def translate(self, texts: list[str], src_lang: Literal['cmn', 'zsm', 'tam']) -> list[str]:
        bs = len(texts)
        try:
            text_output, _ = self.model.predict(
                input=texts,
                task_str="T2TT",
                tgt_lang='eng',
                src_lang=src_lang,
            )
            return [str(t) for t in text_output]
        except Exception as e:
            print(f'[TRANSLATION] Error translating for batch of size {bs} due to error {e}', file=sys.stderr)
            traceback.print_exc()
            return ['Translation failed. Please ignore this translation and read the text from the image directly.'] * bs  # this is a prompt for model


if __name__ == '__main__':
    from tqdm import tqdm
    from utils import create_batch

    ocr_result = [
        OCRResult(text='hello', lang='en', file='1.img'),
        OCRResult(text='大家好', lang='chinese_cht', file='2.img'),
        OCRResult(text='வணக்கம் வணக்கம்', lang='ta', file='3.img'),
        OCRResult(text='hello', lang='en', file='4.img'),
        OCRResult(text='你好', lang='chinese_cht', file='5.img'),
        OCRResult(text='வணக்கம்', lang='ta', file='6.img'),
    ]

    translator = SeamlessM4TTranslation(model_path='seamless-m4t-v2-large', use_cuda=True, compile=True)
    ocr_translate = {
        'latin': 'zsm',
        'ta': 'tam',
        'chinese_cht': 'cmn',
        'en': 'en'
    }
    llava_input: dict[str, tuple[str, str]] = {x.file: (ocr_translate[x.lang], x.text) for x in ocr_result}  # filename: [langauge, ocr_text]
    translate_input = []
    translation_batch_size = 24
    # group by languages to improve batched efficiency
    for lang in ['chinese_cht', 'latin', 'ta']:
        translate_input = [b for b in create_batch([dict(x) for x in ocr_result if x.lang == lang], batch_size=translation_batch_size)]
        for batch in tqdm(translate_input, desc='Translation Batched'):
            results = translator.translate([x['text'] for x in batch], ocr_translate[lang])
            for x, r in zip(batch, results):
                llava_input[x['file']] = (ocr_translate[lang], r)
    del translator
    print(llava_input)
