from asyncio import run
import os
from pathlib import Path

import orjson
import pandas as pd
from tqdm.contrib.concurrent import process_map

from gpt import get_responses, DatasetImage
from image_utils import process_single_image

root = Path('memecap')
img_root = root / 'memes' / 'memes' / 'memes'


def process_image():
    """Only needs to be ran once, not needed for getting gpt labels"""
    all_images = list(img_root.rglob('*.*'))
    with open(root / 'memes-trainval.json', 'r') as f:
        data_trainval = f.read().replace('NaN', 'null')
        data_trainval = orjson.loads(data_trainval)
    df_trainval = pd.DataFrame(data_trainval)
    with open(root / 'memes-test.json', 'r') as f:
        data_test = f.read().replace('NaN', 'null')
        data_test = orjson.loads(data_test)
    df_test = pd.DataFrame(data_test)
    df = pd.concat([df_trainval, df_test])
    df.drop(columns=['category', 'title', 'url'], inplace=True)
    df.dropna(inplace=True)
    df.set_index('post_id', inplace=True)
    df = df[['img_fname', 'img_captions', 'meme_captions', 'metaphors']]
    df.columns = ['img', 'img_description', 'meme_description', 'metaphors']
    df['img'] = df['img'].apply(lambda x: str(img_root / x))
    json_data = df.to_dict(orient='index')
    with open(root / 'metadata.json', 'wb') as f:
        f.write(orjson.dumps(json_data))
    missing_metadata_images = set(all_images) - set(df['img'].apply(Path))
    for img in missing_metadata_images:
        os.remove(img)
    all_images = list(img_root.rglob('*.*'))  # refresh list
    assert len(all_images) == len(df)
    process_map(process_single_image, all_images, chunksize=1)


if __name__ == '__main__':
    with open(root / 'metadata.json', 'rb') as f:
        data = orjson.loads(f.read())
    processed = [DatasetImage(
        id=k,
        path=v['img'],
        label=False, # since memecap was filtered to have no offensive memes
        lang='en',  # since /r/memes is english
        # text= seems like img_description, meme_description, and metaphors are not direct transcriptions rather they are human descriptions
    ) for k, v in data.items()]
    run(get_responses('memecap', processed, output_path=root / 'labels.csv'))
