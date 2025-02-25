from asyncio import run
from gpt import get_responses, DatasetImage
import glob
import os
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

root = Path('MultiOFF_Dataset')
img_root = root / 'Labelled Images'
labels = root / 'Split Dataset'


def process_image():
    """Only needs to be ran once, not needed for getting gpt labels"""
    dfs = []
    for csv in glob.glob(str(labels) + '/*.csv'):
        dfs.append(pd.read_csv(csv))
    df = pd.concat(dfs)
    df.columns = ['img', 'sentence', 'label']
    img_id = df['img'].apply(lambda x: x.split('.')[0])
    df.index = img_id

    all_images = df['img'].apply(lambda x: img_root / x).tolist()
    for img in tqdm(all_images):
        img_new = str(img).replace('.jpg.png', '.png')  # some files is like .jpg.png which means its actually png and it is not reflected in csv so need manual rename
        if '.jpg.png' in str(img):
            os.rename(img, img_new)
        img = img_new
        # convert to png if its not already using PIL
        pil_img = Image.open(img)
        if pil_img.size[0] > 500 or pil_img.size[1] > 500 or img.split('.')[-1] != 'png':
            if pil_img.size[0] > 500 or pil_img.size[1] > 500:
                pil_img.thumbnail((500, 500))
            pil_img.save(''.join(img.split('.')[:-1]) + '.png', optimize=True)
            pil_img.close()
            if img.split('.')[-1] != 'png':
                os.remove(img)


def gen_labels():
    """Only needs to be ran once, not needed for getting gpt labels"""
    new_df = df.copy()
    files = list(glob.glob(str(img_root) + '/*.png'))
    for file in files:
        new_df.loc[file.split(os.sep)[-1].split('.')[0], 'img'] = file
    new_df.index.name = 'image_id'
    new_df.to_csv(root / 'metadata.csv')


if __name__ == '__main__':
    df = pd.read_csv(root / 'metadata.csv', index_col=0)
    processed = [DatasetImage(
        id=str(idx),
        path=str(row['img']),
        text=str(row['sentence']),
        label=row['label'] == 'offensive'
    ) for idx, row in df.iterrows()]
    output_csv = root / 'labels.csv'

    run(get_responses('MultiOFF', processed, output_path=output_csv))

    # df_gpt = pd.read_csv(output_csv, index_col=0)
    # label_score = {'offensive': 1., 'Non-offensiv': 0.}  # Non-offensiv is not a typo, the original labels was spelt like this
    # for idx, row in df_gpt.iterrows():
    #     df_gpt.loc[idx, 'human_label'] = label_score[df.loc[idx, 'label']]  # override gpt's verdict in human_label
