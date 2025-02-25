from gpt import get_responses, DatasetImage
from os import path
from collections import namedtuple
from asyncio import run
from csv import reader

MAMIMeme = namedtuple('MAMIMeme', ('id', 'img', 'label', 'text', 'misogynous', 'shaming', 'stereotype', 'objectification', 'violence'))

def dataset(folder_path: str, csv_path: str, has_headers: bool) -> list[MAMIMeme]:
	with open(csv_path, 'r', newline='') as f:
		csv_reader = reader(f, delimiter='\t')
		if has_headers:
			next(csv_reader)

		entries = []
		for entry in csv_reader:
			file_name, *labels = entry
			transcription = ''
			if len(labels) > 5:
				transcription = labels[-1]
				labels = labels[:-1]
			labels = [int(i) for i in labels]
			entries.append(MAMIMeme(
				id=file_name.split('.')[0],
				img=path.join(folder_path, file_name),
				label=any(labels),
				text=transcription,
				misogynous=labels[0],
				shaming=labels[1],
				stereotype=labels[2],
				objectification=labels[3],
				violence=labels[4]
			))
		return entries

def gen_dataset(folder_path: str) -> list[MAMIMeme]:
	subfolders = (
		('TRAINING', 'TRAINING/training.csv', True),
		('Users/fersiniel/Desktop/MAMI - TO LABEL/TRIAL DATASET/', 'Users/fersiniel/Desktop/MAMI - TO LABEL/TRIAL DATASET/trial.csv', True),
		('test', 'test_labels.txt', False)
	)

	entries = []
	for subfolder in subfolders:
		entries.extend(dataset(
			path.join(folder_path, subfolder[0]),
			path.join(folder_path, subfolder[1]),
			subfolder[2]
		))
	return entries

def get_gpt(folder_path: str, api_key: str = None):
	entries = gen_dataset(folder_path)
	gpt_input = []
	for x in entries:
		method_of_attack = []
		if x.shaming:
			method_of_attack.append('shaming')
		if x.stereotype:
			method_of_attack.append('stereotype')
		if x.objectification:
			method_of_attack.append('objectification')
		if x.violence:
			method_of_attack.append('violence')

		gpt_input.append(DatasetImage(
			id=x.id,
			path=x.img,
			text=x.text,
			label=x.misogynous,
			victim_group=('women',) if x.misogynous else None,
			method_of_attack=tuple(method_of_attack) or None
		))
	
	task = run(get_responses(folder_path, gpt_input, api_key, output_path=path.join(folder_path, 'output.csv')))

if __name__ == '__main__':
	folder_path = 'mami'
	get_gpt(folder_path)
