# Training

## Using DoRA (optional)

1. Unisntall PEFT (`pip uninstall peft`)
2. Install PEFT with DoRA (`pip install git+https://github.com/huggingface/peft@096fe537370cf8a2cb55cc9bd05c7812ca919405`)
3. Update `llava/train/train.py` line 869 with patched version

## Downloading train data
run in `download-zips.sh` /scraping

## Converting train data
Run in /training:
```sh
for f in ../dataset/results/*.csv; do python gen_train_data.py --input_file "$f" --path_to_images ../dataset --output_file train_without_sg.json; done
cp train_without_sg.json train_with_sg.json
python sg_context.py --path_to_images ../dataset --output_file train_with_sg.json
```
This data is NOT deduplicated, and NOT train-val split.

- `train_with_sg_dedup`: deduplicated  training data with SG context memes and wikipedia.
- `train_without_sg_dedup`: deduplicated training data with SG context memes but without sg wikipedia.
- `train_without_sg_memes`: deduplicated training data with only non-sg memes. Without sg wikipedia.


## Training command

https://github.com/haotian-liu/LLaVA#train

https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune_task_lora.sh

Ensure per_device_train_batch_size * num_devices * gradient_accumulation_steps = 16

Note that `deepspeed` requires GPU0 to be in `CUDA_VISIBLE_DEVICES`

Run ../LLaVA/train.sh in ../LLaVA.

- Ensure that `--image_folder` is the same as the one used in generation
- Change the GPUs used with `--include` or `--exclude` before the script path (https://www.deepspeed.ai/getting-started/#resource-configuration-single-node)  
  _e.g. `deepspeed --exclude localhost:2 llava/train/train_mem.py ...`_
- Omitted --eval_data_path ../training/test.json due to bug, and now hard coded into the fork

## Merging LoRA Weights

```sh
python scripts/merge_lora_weights.py --model-path ./checkpoints/llava-1.6-mistral-7b-r128-a256-lr1e4-with-sg-no-eval-sampling-lora --model-base liuhaotian/llava-v1.6-mistral-7b --save-model-path ./checkpoints/llava-1.6-mistral-7b-r128-a256-lr1e4-with-sg-no-eval-sampling-lora-merged
python ../submission/convert_to_hf.py --model lora-llava-merged --output_path lora-llava-merged-hf
```

You may need to create a `image_newline.safetensors` file:

```python
import safetensors.torch
s01 = safetensors.torch.safe_open('../submission/models/llava-v1.6-mistral-7b/model-00001-of-00004.safetensors', framework="pt", device="cpu")
safetensors.torch.save_file({'image_newline': s01.get_tensor('model.image_newline')}, 'image_newline.safetensors')
```