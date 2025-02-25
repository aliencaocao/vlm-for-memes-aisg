import os
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.contrib.concurrent import process_map

from gpt import get_responses, DatasetImage
from asyncio import run
from image_utils import process_single_image

root = Path("mind")

def process_image(): 
    all_images = list(root.rglob("*.png"))
    processed = process_map(process_single_image, all_images, chunksize=1)
    df = pd.DataFrame.from_records(processed, columns=["image_id", "img"])
    df["memeID"] = df["image_id"].apply(lambda x: x[4:])

    dataset_table = pd.read_csv(root / "table.csv", sep=";")
    merged_df = dataset_table.merge(df, on="memeID")
    merged_df["label"] = merged_df.apply(lambda x: x["misogynisticDE"] | x["aggressiveDE"] | x["ironicDE"], axis=1)
    
    merged_df["victim_group"] = merged_df["label"].apply(lambda x: "women" if x else None)

    def fill_method_of_attack(row):
        m = [] 
        if row["misogynisticDE"]:
            m.append("misogyny")
        if row["aggressiveDE"]:
            m.append("aggression")
        if row["ironicDE"]:
            m.append("irony")
        if len(m) == 0:
            return None
        else:
            return tuple(m)
    
    merged_df["method_of_attack"] = merged_df.apply(fill_method_of_attack, axis=1)
    final_df = merged_df[["image_id", "img", "label", "victim_group", "method_of_attack", "text"]]
    final_df.to_csv(root / "metadata.csv", index=False)

if __name__ == "__main__":
    processed = pd.read_csv(root / "metadata.csv")
    processed["image_id"] = processed["image_id"].astype(str)
    processed = [DatasetImage(
            id=x["image_id"],
            path=x["img"],
            lang="en",
            label=x["label"],
            victim_group=x["victim_group"],
            method_of_attack=x["method_of_attack"],
            text=x["text"],
    ) for _, x in processed.iterrows()]
    run(get_responses("mind", processed, output_path=root / "labels.csv", is_async=True))
        