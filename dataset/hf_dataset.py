import os
from datasets import load_dataset, DatasetDict, Features, Value, Image, Sequence, ClassLabel

# -----------------------------------------------------------------------------
# Step 1. Define the mapping for your split JSON files.
# -----------------------------------------------------------------------------
data_files = {
    "test": "test_sharegpt.json",
    "train_without_sg_memes": "train_without_sg_memes_sharegpt.json",
    "train_without_sg_wiki": "train_without_sg_dedup_sharegpt.json",
    "train_with_sg": "train_with_sg_dedup_sharegpt.json"
}

# -----------------------------------------------------------------------------
# Step 2. Process each example to (a) fix the image paths (if needed) and
# (b) add a new column "sub_dataset" which we derive from the image path.
#
# In this example we assume that each JSON record has an "image" field with a
# relative path of the form "sub_dataset/some/other/folders/image.png". We then
# set the "sub_dataset" field to be the first folder.
# -----------------------------------------------------------------------------
def process_example(example):
    if "images" in example and isinstance(example["images"], list) and len(example["images"]) > 0:
        # e.g. "ig_dover_poly/2022-03-20_09-55-34_UTC.png"
        parts = example["images"][0].split('/')
        if parts and len(parts) >= 2:
            sub_dataset = parts[0]
        else:
            raise ValueError(f'Unknown subset for {example["images"][0]}')
        # Optionally, if your images are stored under a common parent folder,
        # you could prepend that folder here. For example, if your images are under
        # a folder called "merged_images", uncomment the next line:
        # example["image"] = os.path.join("merged_images", example["image"])
        if sub_dataset == 'sg_context':
            sub_dataset = 'sg_wiki'
    else:
        sub_dataset = 'sg_wiki'  # assume wiki because only wiki contains examples without images

    example["sub_dataset"] = sub_dataset
    return example

# -----------------------------------------------------------------------------
# Step 3. (Optional) Define features to let Hugging Face know that the "image"
# field should be handled as an Image. You may also specify other fields.
# -----------------------------------------------------------------------------
features = Features({
    "messages": Sequence({
        "content": Value("string"),
        "role": ClassLabel(num_classes=2, names=['user', 'assistant'])
    }),
    "images": Sequence(Image()),
    "metadata": {
        "id": Value("string"),
        "text": Value("string"),
        "gpt_response": ClassLabel(num_classes=2, names=['Yes', 'No']),
        "human_response": ClassLabel(num_classes=2, names=['Yes', 'No'])
    },
    "sub_dataset": Value("string")
})

import re
# harmful: Yes\n
match_label = re.compile(r"harmful: (Yes|No)\n")

def transform_example(example):
    if example['metadata']['gpt_response'] is None:
        # infer using regex
        gpt_answer = example["messages"][-1]["content"]
        match = match_label.search(gpt_answer)
        if match:
            example['metadata']['gpt_response'] = match.group(1)
        else:
            example['metadata']['gpt_response'] = 'Yes'  # if it is rejected, it is harmful for sure.

    if example['metadata']['gpt_response'].lower() not in ['Yes', 'No']: example['metadata']['gpt_response'] = 'No'  # rare case of GPT rejecting to answer due to not enough info given, so we assume it is not offensive since it is likely not enough context to judge.
    example['metadata']['gpt_response'] = example['metadata']['gpt_response'].capitalize()
    example['metadata']['human_response'] = example['metadata']['human_response'].capitalize() if example['metadata']['human_response'] else None
    # make into parallel array for Arrow
    messages = {
        "content": [msg["content"] for msg in example["messages"]],
        "role": [msg["role"] for msg in example["messages"]]
    }
    example["messages"] = messages
    return {
        "messages": example["messages"],
        "images": example["images"],
        "metadata": {
            "id": example["metadata"]["id"],
            "text": example["metadata"]["text"],
            "gpt_response": example["metadata"]["gpt_response"],
            "human_response": example["metadata"]["human_response"],
        },
        "sub_dataset": example["sub_dataset"]
    }


splits = {}
for split_name, file in data_files.items():  # must do 1 by 1 because if more than 2 splits it give "TypeError: Couldn't cast array of type string to null" error.
    ds_single = load_dataset("json", data_files={split_name: file})
    ds_single = ds_single.map(process_example)
    ds_single = ds_single.map(transform_example, features=features)  # Optionally add features later
    splits[split_name] = ds_single[split_name]

ds = DatasetDict(splits)

# -----------------------------------------------------------------------------
# Step 4. Push the DatasetDict to Hugging Face Hub as a private dataset.
# -----------------------------------------------------------------------------

ds.push_to_hub("aliencaocao/multimodal_meme_classification_singapore", private=True)
print("Dataset successfully pushed to the Hugging Face Hub as a private dataset.")