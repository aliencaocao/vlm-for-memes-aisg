import argparse
import json
import os
import sys

import pandas as pd


def import_csv(file_path: str = 'output.csv') -> pd.DataFrame:
    # get 2nd line of csv file
    with open(file_path, 'r') as f:
        for i, l in enumerate(f):
            if i: break
    # use index_col=False if last character in row is comma
    kw = {'index_col': False} if l.strip()[-1] == ',' else {}

    df = pd.read_csv(file_path, **kw)
    assert {'id', 'image', 'response', 'text'}.issubset(set(df.columns.values.tolist())), f'Invalid CSV file: {file_path}'
    return df


def check_images(df: pd.DataFrame, path_to_images: str = '.') -> bool:
    found_df = df.image.apply(lambda x: os.path.exists(os.path.join(path_to_images, x)) and os.path.getsize(os.path.join(path_to_images, x)) > 100)
    if not found_df.all():
        print(df.image[~found_df].tolist())
        return False
    return True


def convert_to_yaml(json_str: str) -> str:
    data = json.loads(json_str)
    result_str = ''
    for k, v in data.items():
        result_str += k + ':'
        if isinstance(v, list):
            for i in v:
                result_str += '\n\t- ' + i + '\n'
        else:
            result_str += ' ' + v + '\n'
    return result_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help='Path to the CSV file', type=str, default='output.csv')
    parser.add_argument('--output_file', help='Path to the output JSON file - will merge if exists', type=str, default='output.json')
    parser.add_argument('--path_to_images', help='Path prefix to the images', type=str, default='.')
    args = parser.parse_args()

    df = import_csv(args.input_file)

    if not check_images(df, args.path_to_images):
        print('Not all files in CSV can be found', args.input_file, args.path_to_images, df.iloc[0].image, file=sys.stderr)
        exit(1)

    newline = '\n'
    df = df.assign(conversations=df.apply(lambda row: [{
        'from': 'human',
        'value': '<image>\n'
                 f'You are a professional content moderator. Analyze this meme in the context of Singapore society. {(" The text in this meme is:" + newline + str(row.text)) if str(row.text) else ""}\n\n'
                 f'Output a YAML in English using tab for indention that contains description, the victim groups and methods of attack if any. Think through the information you just provided and label the meme as harmful using "Yes" or "No". Do not include any other explanation outside the YAML.'
    }, {
        'from': 'gpt',
        'value': convert_to_yaml(row.response)
    }], axis=1))

    del df['response']

    if os.path.exists(args.output_file):
        old_df = pd.read_json(args.output_file)
        df = pd.concat([old_df, df], ignore_index=True)

    df.to_json(args.output_file, orient='records')
