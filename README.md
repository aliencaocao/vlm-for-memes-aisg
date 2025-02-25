# Detecting Offensive Memes with Social Biases in Singapore Context Using Multimodal Large Language Models

## Introduction
This repository contains the code and data for the paper "Detecting Offensive Memes with Social Biases in Singapore Context Using Multimodal Large Language Models".

## Dataset
Dataset is open-sourced on [HuggingFace Hub](https://huggingface.co/datasets/aliencaocao/multimodal_meme_classification_singapore).

## Repository Structure
- `dataset`: contains preprocessing scripts to scrap/convert data into an intermediary CSV format. This CSV is then converted to JSON files for SFT. For convince, the dataset uploaded to HF is already in the final form. It also contain some post processing scripts that convert from LLaVA SFT JSON format to ShareGPT format (used by LLaMA-Factory).
- `evaluation`: evaluation scripts for model standalone and pipeline (OCR and translation). Implemented in isolated FastAPI servers for easy deployment. Also contain a JSON storing a dictionary of Singapore-specific acronyms which is used during inference. Part of the code is modified from the original [LLaVA repository](https://github.com/haotian-liu/LLaVA).
  - `evaluation/outputs`: contains raw outputs for each model/pipeline tested. Directly used for scoring.
- `training`: contains 2 submodules, `LLaVA` and `qwen2-vl`. `LLaVA` is a modified version of the original LLaVA repository with fixes on training code. `qwen2-vl` contains a fork of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) with modifications, and various training config and bash scripts. It can train both Qwen2-VL and LLaVA, but we only used it to train Qwen2-VL.

## Models
|                                                   Model                                                   | AUROC  | Accuracy |
|:---------------------------------------------------------------------------------------------------------:|:------:|:--------:|
| [LLaVA-NeXT Mistral 7B](https://huggingface.co/aliencaocao/llava-1.6-mistral-7b-offensive-meme-singapore) | 0.7345 |  0.7259  |
|       [Qwen2-VL 7B](https://huggingface.co/aliencaocao/qwen2-vl-7b-rslora-offensive-meme-singapore)       | 0.8192 |  0.8043  |

For W&B training logs, please contact Billy at aliencaocao@gmail.com. For other variants and pipeline mode metrics, please refer to our paper.


## Citation
If you find this repository useful, please cite our paper:
```bibtex
TBC
```