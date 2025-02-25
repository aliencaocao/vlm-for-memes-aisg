import os
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from gpt import get_responses, DatasetImage
from asyncio import run

root = Path('indian_memes')
img_root = root / 'memes'


def process_image():
    """Only needs to be ran once, not needed for getting gpt labels"""
    all_images = list(img_root.glob('*'))
    processed = []
    for img in tqdm(all_images):
        # convert to png if its not already using PIL
        pil_img = Image.open(img)
        if pil_img.size[0] > 500 or pil_img.size[1] > 500 or img.suffix != '.png':
            if pil_img.size[0] > 500 or pil_img.size[1] > 500:
                pil_img.thumbnail((500, 500))
            pil_img.save(''.join(str(img).split('.')[:-1]) + '.png', optimize=True)
            pil_img.close()
            if img.suffix != '.png':
                os.remove(img)
        processed.append((img.stem, img.stem + '.png'))
        df = pd.DataFrame.from_records(processed, columns=['image_id', 'img'])
        df['img'] = df['img'].apply(lambda x: str(img_root / x))
        df.to_csv(root / 'metadata.csv', index=False)

if __name__ == '__main__':
    processed = pd.read_csv(root / 'metadata.csv')
    processed['image_id'] = processed['image_id'].astype(str)
    processed = [DatasetImage(
        id=x['image_id'],
        lang='en', # since all memes are in english
        path=x['img'],
    ) for _, x in processed.iterrows()]
    #get_responses('indian_meme', processed, output_path=root / 'labels.csv', is_async=False)
    run(get_responses('indian_meme', processed, output_path=root / 'labels.csv', is_async=True))
