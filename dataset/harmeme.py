from gpt import get_responses, DatasetImage
from os import listdir, path
import json
from collections import namedtuple
from asyncio import run

HatefulMeme = namedtuple('HatefulMeme', ('id', 'img', 'label', 'text'))

def gen_dataset(folder_path: str) -> list[HatefulMeme]:
	subfolders = ('data', 'Covid_Dataset')

	entries = []
	for subfolder in subfolders:
		data_path = path.join(folder_path, subfolder, 'datasets/memes/defaults')
		annotations_path = path.join(data_path, 'annotations')
		for fname in listdir(annotations_path):
			if fname[-6:] != '.jsonl':
				continue
			with open(path.join(annotations_path, fname), 'r') as f:
				for l in f:
					data = json.loads(l)
					if 'id' in data and 'image' in data and 'labels' in data and 'text' in data:
						label = [l for l in data['labels'] if l [-8:] == ' harmful'][0]
						entries.append(HatefulMeme(
							id=data['id'],
							img=path.join(data_path, 'images', data['image']),
							label=label,
							text=data['text'],
						))
	return entries

def get_gpt(folder_path: str, api_key: str = None):
	entries = gen_dataset(folder_path)
	entries = [DatasetImage(
		id=x.id,
		path=x.img,
		text=x.text,
		label='not' in x.label
	) for x in entries]
	task = run(get_responses(folder_path, entries, api_key, output_path=path.join(folder_path, 'output.csv')))

if __name__ == '__main__':
	folder_path = 'harmeme'
	get_gpt(folder_path)
