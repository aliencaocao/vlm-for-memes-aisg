from gpt import get_responses, DatasetImage
from os import path
from collections import namedtuple
from asyncio import run
from csv import reader

TamilTrollMeme = namedtuple('TamilTrollMeme', ('id', 'img', 'label', 'text'))

def gen_dataset(folder_path: str) -> list[TamilTrollMeme]:
	entries = []

	with open(path.join(folder_path, 'train_captions.csv'), 'r', newline='') as f:
		csv_reader = reader(f)
		next(csv_reader)
		for entry in csv_reader:
			entries.append(TamilTrollMeme(
				id=entry[0],
				img=path.join(folder_path, 'training_img/uploaded_tamil_memes', entry[1]),
				label=int(entry[1][:3] != 'not'),
				text=entry[2],
			))

	test_set = {}
	with open(path.join(folder_path, 'test_captions.csv'), 'r', newline='') as f:
		csv_reader = reader(f)
		next(csv_reader)
		for entry in csv_reader:
			test_set[entry[0]] = entry[1:]

	with open(path.join(folder_path, 'gold_labels_for_test.csv'), 'r', newline='') as f:
		csv_reader = reader(f)
		next(csv_reader)
		for entry in csv_reader:
			if entry[1] != test_set[entry[0]][0]:
				pass
			entries.append(TamilTrollMeme(
				id=entry[0],
				img=path.join(folder_path, 'test_img/test_img', entry[1]),
				label=int(entry[2] == 'troll'),
				text=test_set[entry[0]][1],
			))

	return entries

def get_gpt(folder_path: str, api_key: str = None):
	entries = gen_dataset(path.join(folder_path, 'Tamil_troll_memes'))
	entries = [DatasetImage(
		id=x.id,
		path=x.img,
		text=x.text,
		# label is "troll" vs "not troll", unreliable
	) for x in entries]
	task = run(get_responses(folder_path, entries, api_key, output_path=path.join(folder_path, 'output.csv')))

if __name__ == '__main__':
	folder_path = 'tamil_troll'
	get_gpt(folder_path)
