from pathlib import Path
from asyncio import run

import pandas as pd
from tqdm.contrib.concurrent import process_map

from gpt import get_responses, DatasetImage
from image_utils import process_single_image

root = Path('memes_classified_and_labelled')


def process_image():
    all_images = list(root.rglob('*.png'))
    processed = process_map(process_single_image, all_images, chunksize=1)
    df = pd.DataFrame.from_records(processed, columns=['image_id', 'img'])
    df.to_csv(root / 'metadata.csv', index=False)


if __name__ == '__main__':
    processed = pd.read_csv(root / 'metadata.csv')
    processed['image_id'] = processed['image_id'].astype(str)
    processed = [DatasetImage(
        id=x['image_id'],
        path=x['img'],
        lang='en' # TODO: check 'section' to see if the subreddit is English speaking
    ) for _, x in processed.iterrows()]
    run(get_responses('meme_classified_and_labeled', processed, output_path=root / 'labels.csv'))
