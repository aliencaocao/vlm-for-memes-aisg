import argparse
import os
import random
import sys
from pathlib import Path

import orjson
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_images', help='Path prefix to the images', type=str, default='../dataset')
parser.add_argument('--output_file', help='Path to the output JSON file - will merge if exists', type=str, default='sg_context_output.json')
args = parser.parse_args()

root = args.path_to_images / Path('sg_context')
img_root = root / 'img'
print('Using img root:', img_root)

def check_image_exist(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{path} not found')
    return True


imgs = []
with open(root / 'context.json', 'r') as f:
    data = orjson.loads(f.read())
    for v in data.values():
        if v['cover']:
            v['img'] = str(img_root / v['cover'])
            check_image_exist(v['img'])
        else:
            v['img'] = None
        if v['imgs']:
            imgs.extend([{'path': str(img_root / x['path']), 'text': x['txt']} for x in v['imgs'] if len(x['txt']) > 20])
        v['text'] = v['body']
        del v['cover'], v['imgs'], v['links'], v['body']
for img in imgs:
    fixed_fname = img['path']
    if '\'' in fixed_fname:
        fixed_fname = fixed_fname.replace('\'', '')
        print(f'Fixed {img["path"]} to {fixed_fname}')
    if sys.platform != 'win32':  # on non windows, there are issues with unicode filenames getting literally mapped to ascii
        for c in img['path']:
            if ord(c) > 128:  # fname is unicode wrongly encoded as ascii
                fixed_fname = fixed_fname.replace(c, '#U' + f'{ord(c):04x}')
                print(fixed_fname)
    img['path'] = fixed_fname
    check_image_exist(img['path'])
ques_templates = ['What is {title}?', 'What is the story behind {title}?', 'Can you explain {title} to me?', 'What exactly does {title} entail?', 'Could you provide some insight into {title}?', 'Could you elaborate on {title} for me?', 'Can you give me a detailed rundown of {title}?', 'I\'m curious about {title}, could you shed some light on it?', 'Explain {title} in detail.']
img_questions = ['What is in this image?', 'What is the story behind this image?', 'Can you explain this image to me?', 'What exactly does this image entail?', 'Could you provide some insight into this image?', 'Could you elaborate on this image for me?', 'Can you give me a detailed rundown of this image?', 'I\'m curious about this image, could you shed some light on it?', 'Explain this image in detail.']

train_data = []
with_img = 0
without_img = 0
img_only = 0
for d in data.values():
    if d['img']:
        human_prompt = '<image>\n' + random.choice(img_questions)
        answer = f'This is an image of {d["title"]}. {d["text"]}'
    else:
        human_prompt = random.choice(ques_templates).format(title=d['title'])
        answer = d['text'].replace('alt=', '')
    conv = [{'from': 'human', 'value': human_prompt}, {'from': 'gpt', 'value': answer}]
    if d['img']:
        train_data.append({
            'id': d['title'],
            'image': d['img'],
            'text': '',  # no OCR for SG context things
            'conversations': conv})
        with_img += 1
    else:
        train_data.append({
            'id': d['title'],
            'text': '',
            'conversations': conv})
        without_img += 1

for img in imgs:
    human_prompt = '<image>\nDescribe this image in a short sentence.'
    answer = img['text'].replace('alt=', '')
    conv = [{'from': 'human', 'value': human_prompt}, {'from': 'gpt', 'value': answer}]
    train_data.append({
        'id': Path(img['path']).stem,
        'image': img['path'],
        'text': '',  # no OCR for SG context things
        'conversations': conv})
    img_only += 1

print(f'With image: {with_img}, Without image: {without_img}, Image only: {img_only}')

df = pd.DataFrame(train_data)

if os.path.exists(args.output_file):
    old_df = pd.read_json(args.output_file)
    df = pd.concat([old_df, df], ignore_index=True)

df.to_json(args.output_file, orient='records')
