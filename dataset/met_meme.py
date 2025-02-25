from gpt import get_responses, DatasetImage
from os import path
from collections import namedtuple
from asyncio import run
from csv import DictReader

METMeme = namedtuple('METMeme', ('id', 'img', 'lang', 'label', 'sentiment', 'sentiment_degree', 'intention', 'offensiveness'))

def dataset(folder_path: str, csv_path: str) -> list[METMeme]:
	decoded = ''
	with open(csv_path, 'rb') as f:
		while chunk := f.read(1):
			try:
				decoded += chunk.decode('ascii')
			except UnicodeDecodeError:
				pass

	csv_reader = DictReader(decoded.splitlines())

	entries = []
	for entry in csv_reader:
		img = entry['file_name'] if 'file_name' in entry else entry['images_name'].replace('_', '- ')
		entries.append(METMeme(
			id=img.split('.')[0],
			img=path.join(folder_path, img),
			label=int(entry['offensiveness detection'].split('(')[0]) / 3,
			sentiment=entry['sentiment category'],
			sentiment_degree=entry['sentiment degree'],
			intention=entry['intention detection'],
			offensiveness=entry['offensiveness detection'],
			lang='ch' if csv_path == 'label_C.csv' else 'en'
		))
	return entries

def gen_dataset(folder_path: str) -> list[METMeme]:
	subfolders = (
		('Cimages/Cimages/Cimages', 'label_C.csv'),
		('Eimages/Eimages/Eimages', 'label_E.csv'),
	)

	entries = []
	for subfolder in subfolders:
		entries.extend(dataset(
			path.join(folder_path, subfolder[0]),
			path.join(folder_path, subfolder[1])
		))
	return entries

def get_gpt(folder_path: str, api_key: str = None):
	entries = gen_dataset(folder_path)
	entries = [DatasetImage(
		id=x.id,
		lang=x.lang,
		path=x.img,
		label=x.label,
		method_of_attack=x.intention
	) for x in entries]
	task = run(get_responses(folder_path, entries, api_key, output_path=path.join(folder_path, 'output.csv')))

if __name__ == '__main__':
	folder_path = 'met_meme'
	get_gpt(folder_path)
