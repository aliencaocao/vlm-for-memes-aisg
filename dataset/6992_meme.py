from pathlib import Path
from asyncio import run

import pandas as pd
from tqdm.contrib.concurrent import process_map

from image_utils import process_single_image
from gpt import get_responses, DatasetImage

root = Path('6992_meme')
img_root = root / 'images'


def process_image():
    all_images = list(img_root.rglob('*.*'))
    df = pd.read_csv(root / 'metadata.csv', index_col=0)
    df.drop(columns=['text_ocr', 'overall_sentiment'], inplace=True)
    df.columns = ['img', 'text']
    img_id = df['img'].apply(lambda x: x.split('.')[0])
    df.index = img_id
    df.index.name = 'image_id'

    processed = process_map(process_single_image, all_images, chunksize=1)
    df_processed = pd.DataFrame.from_records(processed, columns=['image_id', 'img'])
    df_processed.dropna(inplace=True)  # drop error ones
    df_processed = df_processed.set_index('image_id')
    df_processed = df_processed.join(df['text'], on='image_id')
    df_processed.to_csv(root / 'metadata.csv', index=True)


if __name__ == '__main__':
    processed = pd.read_csv(root / 'metadata.csv')
    processed['image_id'] = processed['image_id'].astype(str)
    processed = [DatasetImage(
        id=x['image_id'],
        path=x['img'],
        text=x['text'],
    ) for _, x in processed.iterrows()]
    run(get_responses('6992_meme', processed, output_path=root / 'labels.csv'))
