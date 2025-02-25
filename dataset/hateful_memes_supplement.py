import ast
import pandas as pd

from asyncio import run
from gpt import get_responses, DatasetImage
from pathlib import Path

root = Path("hateful_memes_supplement")

if __name__ == "__main__":
    processed = pd.read_csv(root / "metadata.csv")
    processed["image_id"] = processed["image_id"].astype(str)
    processed = [DatasetImage(
            id=x["image_id"],
            path=x["img"],
            lang="en",
            label=x["label"],
            victim_group=None if pd.isna(x["victim_group"]) else ast.literal_eval(x["victim_group"]),
            method_of_attack=None if pd.isna(x["method_of_attack"]) else ast.literal_eval(x["method_of_attack"]),
            text=x["text"],
            harm_reasoning=None if pd.isna(x["harm_reasoning"]) else ast.literal_eval(x["harm_reasoning"]),
    ) for _, x in processed.iterrows()]
    run(get_responses("hateful_memes_supplement", processed, output_path=root/"labels.csv", is_async=True))
