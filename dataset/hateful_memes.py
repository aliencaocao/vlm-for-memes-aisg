from gpt import get_responses, DatasetImage
from os import listdir, path
import json
from collections import namedtuple
from asyncio import run
from csv import reader

HatefulMeme = namedtuple('HatefulMeme', ('id', 'img', 'label', 'text'))

def gen_dataset(folder_path: str) -> list[HatefulMeme]:
	entries = []
	for fname in listdir(folder_path):
		if fname[-6:] != '.jsonl':
			continue
		with open(path.join(folder_path, fname), 'r') as f:
			for l in f:
				data = json.loads(l)
				if 'id' in data and 'img' in data and 'label' in data and 'text' in data:
					entries.append(HatefulMeme(
						id=data['id'],
						img=path.join(folder_path, data['img']),
						label=data['label'],
						text=data['text'],
					))
	return entries

def get_gpt(folder_path: str, api_key: str = None):
	entries = gen_dataset(folder_path)
	entries = [DatasetImage(
		id=x.id,
		path=x.img,
		text=x.text,
		label=x.label
	) for x in entries]
	task = run(get_responses(folder_path, entries, api_key, output_path=path.join(folder_path, 'output.csv')))

def match_gpt(folder_path: str, output_path: str):
	entries = { i.id: i for i in gen_dataset(folder_path) }

	total, correct = 0, 0

	with open(output_path, 'r', newline='') as f:
		csv_reader = reader(f)
		next(csv_reader)

		for img_id, response, finish_reason in csv_reader:
			img_id = str(img_id)
			if 'sorry' in response or img_id not in entries:
				continue
			total += 1
			if 'not offensive' in response.lower() or 'non-offensive' in response.lower() or 'not inherently offensive' in response.lower() or response[:2] == 'No':
				if entries[img_id].label == 0:
					correct += 1
			else:
				if entries[img_id].label == 1:
					correct += 1
			print(img_id)

	print(total, correct, correct / total)

if __name__ == '__main__':
	folder_path = 'hateful_memes'
	get_gpt(folder_path)
