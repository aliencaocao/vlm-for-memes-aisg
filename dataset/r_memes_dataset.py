from pathlib import Path
from asyncio import run

import pandas as pd
from tqdm.contrib.concurrent import process_map

from gpt import get_responses, DatasetImage
from image_utils import process_single_image

root = Path('r_memes_dataset')
img_root = root / 'images'


def process_image():
    """Only needs to be ran once, not needed for getting gpt labels"""
    all_images = list(img_root.rglob('*.*'))
    processed = process_map(process_single_image, all_images, chunksize=1)
    processed = pd.DataFrame.from_records(processed, columns=['image_id', 'img'])
    processed.dropna(inplace=True)  # drop error ones
    processed = processed.set_index('image_id')
    processed.to_csv(root / 'metadata.csv', index=True)


if __name__ == '__main__':
    processed = pd.read_csv(root / 'metadata.csv')
    processed['image_id'] = processed['image_id'].astype(str)
    processed = [DatasetImage(
        id=x['image_id'],
        path=x['img'],
        lang='en', # assume reddit is English
    ) for _, x in processed.iterrows()]
    run(get_responses('r_memes', processed, output_path=root / 'labels.csv'))
