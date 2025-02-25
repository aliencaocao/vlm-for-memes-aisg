from gpt import get_responses, DatasetImage
import os
from pathlib import Path
from asyncio import run

from PIL import Image
from tqdm import tqdm

root = Path('reddit_memes_dataset')
img_root = root / 'memes' / 'memes'


def process_image():
    """Only needs to be ran once, not needed for getting gpt labels"""
    all_images = list(img_root.glob('*'))

    for img in tqdm(all_images, total=len(all_images)):
        # convert to png if its not already using PIL
        pil_img = Image.open(img)
        if pil_img.size[0] > 500 or pil_img.size[1] > 500 or img.suffix != '.png':
            if pil_img.size[0] > 500 or pil_img.size[1] > 500:
                pil_img.thumbnail((500, 500))
            pil_img.save(''.join(str(img).split('.')[:-1]) + '.png', optimize=True)
            pil_img.close()
            if img.suffix != '.png':
                os.remove(img)


if __name__ == '__main__':
    processed = [DatasetImage(
        id=img.stem,
        path=str(img),
        lang='en', # assume reddit is English
    ) for img in img_root.glob('*')]
    run(get_responses('reddit_memes', processed, output_path=str(root / 'labels.csv')))
