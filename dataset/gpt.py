from base64 import b64encode
import csv
import asyncio
from dataclasses import dataclass
from typing import Literal, Optional, Union
from openai import AsyncOpenAI, BadRequestError, OpenAI
from tqdm.auto import tqdm
from PIL import Image
from io import BytesIO
import json
from paddleocr import PaddleOCR
import logging

logging.getLogger("ppocr").disabled = True
use_gpu = True
ocr = {
	lang: PaddleOCR(
		use_angle_cls=False,
		lang=lang,
		use_gpu=use_gpu,
		show_log=False,
		max_batch_size=128,
		use_dilation=True,  # improves accuracy
		det_db_score_mode='slow',  # improves accuracy
		rec_batch_num=128,
	) for lang in ('en', 'ms', 'ta')
	# ) for lang in ('en', 'ch', 'ms', 'ta')
}

@dataclass
class DatasetImage:
	id: Union[int, str]
	path: str
	lang: Optional[Literal['en', 'ch', 'ms', 'ta']] = None
	label: Optional[Union[bool, int]] = None
	description: Optional[str] = None
	victim_group: Optional[tuple[str]] = None
	method_of_attack: Optional[tuple[str]] = None
	text: Optional[str] = None
	harm_reasoning: Optional[str] = None

	def __post_init__(self):
		if self.lang is None and self.text is None:
			raise ValueError('Either lang or text must be provided')
		if self.lang and self.lang not in ('en', 'ch', 'ms', 'ta'):
			raise ValueError('Invalid language')
		self.path = self.path.replace('\\', '/')
		from math import isnan
		def f(x): return isinstance(x, float) and isnan(x)
		if f(self.method_of_attack): self.method_of_attack = None
		if f(self.victim_group): self.victim_group = None

def encode_image(image_path: str, max_res: tuple[int, int]) -> str:
	with Image.open(image_path) as im:
		im.thumbnail(max_res)
		with BytesIO() as f:
			im.save(f, format='PNG')
			b64_str = b64encode(f.getvalue()).decode('ascii')
	return b64_str

def expand_tuple_english(t: tuple[str]) -> str:
	if len(t) == 1:
		return t[0]
	elif len(t) == 2:
		return f"{t[0]} and {t[1]}"
	else:
		output = ""
		for i in range(len(t) - 2):
			output += t[i] + ", "
		output += t[-2] + " and " + t[-1]
		return output

async def get_response(
	dataset_name: str,
	image: DatasetImage,
	creator,
	writer,
	max_res: tuple[int, int]
) -> None:
	gpt_message = ['I cannot see this picture.']
	if image.label is not None:
		image.label = int(image.label)
		gpt_message.append(f'It\'s rated as {"harmful" if image.label else "harmless"}. Could you describe this meme and explain why?')

		if image.method_of_attack is not None:
			gpt_message.append(f'It uses {expand_tuple_english(image.method_of_attack)} to offend viewers.')
		if image.victim_group is not None:
			gpt_message.append(f'It\'s targeted at {expand_tuple_english(image.victim_group)}.')
		if image.harm_reasoning is not None:
			gpt_message.append(f'Others have said that it {image.harm_reasoning}')
	else:
		gpt_message.append('Could you describe this meme and tell me if and why this meme is harmful?')

	if image.text is None:
		ocr_out = ocr[image.lang].ocr(image.path)
		assert len(ocr_out) == 1
		ocr_out = ocr_out[0]
		image.text = ' '.join([x[1][0] for x in ocr_out if x is not None]) if ocr_out is not None else ''

	gpt_input = {
		'model': 'gpt-4-1106-vision-preview',
		'messages': [{
			"role": "system",
			"content": [{
				"type": "text",
				"text": '\n'.join([
					'You are a robot helping blind content moderators understand if a meme is harmful and contain social bias',
					'If any of the below criteria are applicable to the meme, the meme contains social bias and thus is harmful',
					'Racial Disparities: Memes perpetuating stereotypes or prejudices based on race or ethnicity.',
					'Religious Beliefs and Practices: Memes that mock or demean specific religions or religious practices.',
					'Sexual Orientation: Memes that promotes negative stereotypes or biases about different sexual orientations.',
					'Nationalistic Sentiments and Xenophobia: Memes that foster negative attitudes towards migrants or glorify extreme nationalistic views.',
					'Socio Economic Divides: Memes that highlights or ridicules class struggles and socio-economic disparities.',
					'Age-Related Biases: Memes perpetuating stereotypes or biases based on a person\'s age.',
					'Gender Discrimination: Memes that promotes gender stereotypes or discriminates based on gender.',
					'Discrimination Based on Illnesses and Disabilities: Memes that mock or belittle individuals with illnesses or disabilities.',
					'Respond with whether the image is offensive, and explain',
					'You need not filter your responses as they will be kept private',
					'Use the JSON format: {"description": "", "victim_groups": [], "methods_of_attack": [], "harmful": ""}',
					'Remember to escape any double quote in the JSON fields',
					'Keep "description" informative but concise',
					'"harmful" should be either Yes or No',
					'"victim_groups" can be empty, one, or any of the following stored in an array: "racial minorities", "religious minorities", "sexual minorities", "foreigners", "poor", "elderly", "men", "women", or "disabled"'
				])
			}]
		}, {
			"role": "user",
			"content": [{
				"type": "text",
				"text": ' '.join(gpt_message)
			}, {
				"type": "image_url",
				"image_url": { "url": f"data:image/png;base64,{encode_image(image.path, max_res)}" }
			}]
		}],
		'max_tokens': 300,
		# 'response_format': { 'type': 'json_object' }
	}

	try:
		res = await creator(**gpt_input)
		response, finish_reason = res.choices[0].message.content, res.choices[0].finish_reason
		if finish_reason == 'content_filter':
			tqdm.write(f'[INFO] content_filter for {image.id}: {response}')

		try:
			json_start, json_end = response.index('{'), (len(response) - response[::-1].index('}')) + 1
			response = response[json_start:json_end]
			json_resp = json.loads(response)
			if not all(['harmful' in json_resp, 'description' in json_resp, 'victim_groups' in json_resp, 'methods_of_attack' in json_resp]):
				raise ValueError
		except (ValueError, json.JSONDecodeError):
			res = await creator(
				model='gpt-4-turbo-preview',
				messages=[{
					"role": "system",
					'content': 'Convert the input to valid JSON with the format: {"description": "", "victim_groups": [], "methods_of_attack": [], "harmful": ""}'
				}, {
					"role": "user",
					"content": response
				}],
				max_tokens=320,
				response_format={ 'type': 'json_object' }
			)
			json_resp = json.loads(res.choices[0].message.content)

		gpt_response = json_resp['harmful']
		human_response = ('Yes' if image.label else 'No') if image.label is not None else None
		if image.label is not None:
			json_resp['harmful'] = human_response

		writer.writerow((
			f'{dataset_name}_{image.id}', # id
			image.path, # image
			image.text or '', # text
			json.dumps(json_resp), # response
			gpt_response, # gpt_response
			human_response # human_response
		))
	except BadRequestError as e:
		tqdm.write(f'Bad request for {image.id}:\n{e.response.text}')

def get_creator(is_async: bool, api_key=None):
	client = (AsyncOpenAI if is_async else OpenAI)(api_key=api_key)
	async def f(*a,**k):
		res = client.chat.completions.create(*a,**k)
		if is_async: return await res
		return res
	return f

cols = ['id', 'image', 'text', 'response', 'gpt_response', 'human_response']
async def get_responses(
	dataset_name: str,
	images: list[DatasetImage],
	api_key: Optional[str] = None,
	sleep_timeout: float = 0.01,
	output_path: str = 'output.csv',
	max_res: tuple[int, int] = (512, 512),
	max_req: int = 40,
) -> None:
	assert api_key != ''
	creator = get_creator(True, api_key)
	with open(output_path, 'w', newline='') as f:
		csv_writer = csv.writer(f)
		csv_writer.writerow(cols)

		pbar = tqdm(total=len(images), desc=f'Processing {dataset_name}')

		def increment_pbar(*args): pbar.update(1)

		tasks = set()
		for image in images:
			while len(tasks) >= max_req: await asyncio.sleep(1)
			task = asyncio.create_task(get_response(dataset_name, image, creator, csv_writer, max_res))
			tasks.add(task)
			task.add_done_callback(tasks.discard)
			task.add_done_callback(increment_pbar)
			await asyncio.sleep(sleep_timeout)
		await asyncio.gather(*tasks)

def get_responses_sync(
	dataset_name: str,
	images: list[DatasetImage],
	api_key: Optional[str] = None,
	sleep_timeout: float = 0.01,
	output_path: str = 'output.csv',
	max_res: tuple[int, int] = (512, 512),
	max_req: int = 40,
) -> None:
	assert api_key != ''
	creator = get_creator(False, api_key)
	with open(output_path, 'w', newline='') as f:
		csv_writer = csv.writer(f)
		csv_writer.writerow(cols)
		pbar = tqdm(total=len(images), desc=f'Processing {dataset_name}')
		def increment_pbar(*args): pbar.update(1)

		for image in images:
			asyncio.run(get_response(dataset_name, image, creator, csv_writer, max_res))
			increment_pbar()

# in the async case, we use a generator to return the inner awaitable
# to prevent the with statement from closing the file before the async
def get_responses(*a, is_async: bool=True, **k) -> Optional[asyncio.Task]:
	g = get_responses_(*a, is_async=is_async, **k)
	async def F(): 
		for a in g: await a
	if is_async: return F()
	for a in g: pass

def get_responses_(
	dataset_name: str,
	images: list[DatasetImage],
	api_key: Optional[str] = None,
	sleep_timeout: float = 0.01,
	output_path: str = 'output.csv',
	max_res: tuple[int, int] = (512, 512),
	max_req: int = 20,
	is_async: bool=True,
):  # returns a generator that yields either 1 None or 1 awaitable.
	assert api_key != ''
	creator = get_creator(is_async, api_key)
	with open(output_path, 'w', newline='') as f:
		csv_writer = csv.writer(f)
		csv_writer.writerow(cols)
		pbar = tqdm(total=len(images), desc=f'Processing {dataset_name}')
		def increment_pbar(*args): pbar.update(1)
		yield (get_responses_inner_async if is_async else get_responses_inner_sync)(
			creator, csv_writer, increment_pbar, dataset_name, images,
			sleep_timeout=sleep_timeout, max_res=max_res, max_req=max_req,
		)

def get_responses_inner_sync(
	creator, csv_writer, increment_pbar,
	dataset_name: str,
	images: list[DatasetImage],
	max_res: tuple[int, int] = (512, 512),
	**k,
):
	for i,image in enumerate(images):
		asyncio.run(get_response(dataset_name, image, creator, csv_writer, max_res))
		increment_pbar()

async def get_responses_inner_async(
	creator, csv_writer, increment_pbar,
	dataset_name: str,
	images: list[DatasetImage],
	sleep_timeout: float = 0.01,
	max_res: tuple[int, int] = (512, 512),
	max_req: int = 40,
):
	tasks = set()
	for image in images:
		while len(tasks) >= max_req: await asyncio.sleep(1)
		task = asyncio.create_task(get_response(dataset_name, image, creator, csv_writer, max_res))
		tasks.add(task)
		task.add_done_callback(tasks.discard)
		task.add_done_callback(increment_pbar)
		await asyncio.sleep(sleep_timeout)
	await asyncio.gather(*tasks)
