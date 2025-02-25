from PIL import Image
from pathlib import Path
import pandas as pd
import ast
from PIL import Image
from tqdm.contrib.concurrent import process_map

from gpt import get_responses, DatasetImage
from asyncio import run
from image_utils import extract_frame_from_gif

root = Path("facebook_hateful_memes")
img_root = root / "img"

output_imgs_root = root / "processed_images"
output_imgs_root.mkdir(parents=True, exist_ok=True)

def prepare_metadata():
    # Merge the original dataset into 1 big dataframe
    print(len(list(img_root.iterdir()))) # should be 12140
    train_df = pd.read_json(root / "train.jsonl", lines=True)
    train_df["original_split"] = "train"
    dev_seen_df = pd.read_json(root / "dev_seen.jsonl", lines=True)
    dev_seen_df["original_split"] = "dev_seen"
    dev_unseen_df = pd.read_json(root / "dev_unseen.jsonl", lines=True)
    dev_unseen_df["original_split"] = "dev_unseen"
    test_seen_df = pd.read_json(root / "test_seen.jsonl", lines=True)
    test_seen_df["original_split"] = "test_seen"
    test_unseen_df = pd.read_json(root / "test_unseen.jsonl", lines=True)
    test_unseen_df["original_split"] = "test_unseen"
    df = pd.concat([train_df, dev_seen_df, dev_unseen_df, test_seen_df, test_unseen_df])
    print(df.shape) # should be (12540, 4)

    # remove duplicates in df. Duplicates exist between the dev_seen and dev_unseen datasets
    df = df.drop_duplicates(subset="id")
    print(df.shape) # should be (12140, 4)

    # now read the fine grained version
    fg_train_df = pd.read_json(root / "finegrained_ver/annotations" / "train.json", lines=True)
    fg_train_df["fg_split"] = "train"
    fg_dev_seen_df = pd.read_json(root / "finegrained_ver/annotations" / "dev_seen.json", lines=True)
    fg_dev_seen_df["fg_split"] = "dev_seen"
    fg_dev_unseen_df = pd.read_json(root / "finegrained_ver/annotations" / "dev_unseen.json", lines=True)
    fg_dev_unseen_df["fg_split"] = "dev_unseen"
    fg_test_df = pd.read_json(root / "finegrained_ver/annotations" / "test.jsonl", lines=True)
    fg_test_df["fg_split"] = "test_seen"
    fg_df = pd.concat([fg_train_df, fg_dev_seen_df, fg_dev_unseen_df, fg_test_df])
    print(fg_df.shape) # this is (10540, 9)

    # remove duplicates in fg_df. Duplicates exist between the dev_seen and dev_unseen datasets
    fg_df = fg_df.drop_duplicates(subset="id")
    print(fg_df.shape) # should be (10140, 9)
    # There are 2000 less images in the finegrained version because the test_unseen split is not included
    # note that there are no finegrained labels for test_seen

    # merge the original dataset and finegrained version
    merged_df = pd.merge(how="left", left=df, right=fg_df[["id", "text", "gold_hate", "gold_pc", "gold_attack", "fg_split"]], on="id")
    print(merged_df.shape) # should be (12140, 8)

    # check if the fg_label is the same as the original label
    merged_df["fg_label"] = merged_df["gold_hate"].apply(lambda x: 1 if x == ["hateful"] else 0)
    disagreements = merged_df[merged_df["label"] != merged_df["fg_label"]]

    # after inspecting the disagreements, it seems that the original labels are better
    # just need to change image 85761 to not harmful
    merged_df = merged_df.drop(columns=["gold_hate", "fg_label"])
    merged_df.loc[merged_df["id"] == 85761, "label"] = 0

    # after testing, there are no disagreements with the finegrained text so we can drop it and retain original text
    merged_df = merged_df.drop(columns=["text_y", "fg_split"])
    merged_df = merged_df.rename(columns={"text_x": "text"})

    # get all unique values in the gold_pc and gold_attack column
    unique_pcs = set() # {'nationality', 'disability', 'sex', 'religion', 'race', 'pc_empty'}
    unique_attack_methods = set() # {'exclusion', 'slurs', 'mocking', 'inferiority', 'dehumanizing', 'attack_empty', 'contempt', 'inciting_violence'}
    def get_unique_values(row):
        if isinstance(row["gold_pc"], list):
            for pc in row["gold_pc"]:
                unique_pcs.add(pc)
        if isinstance(row["gold_attack"], list):
            for attack in row["gold_attack"]:
                unique_attack_methods.add(attack)
    merged_df.apply(get_unique_values, axis=1)

    # decide to do away with their Protected Category victim group as their victim group is not specific enough
    merged_df = merged_df.drop(columns=["gold_pc"])

    # clean up the attack method column so the English fits with our prompt template
    attack_method_dict = {
        "exclusion": "exclusion",
        "slurs": "slurs",
        "mocking": "mockery",
        "inferiority": "inferiority",
        "dehumanizing": "dehumanization",
        "attack_empty": None,
        "contempt": "contempt",
        "inciting_violence": "calls to incite violence"
    }

    def substitute_attack_method(x):
        if isinstance(x, list):
            if "attack_empty" in x:
                return None
            return [attack_method_dict[attack] for attack in x]
        else:
            return None
    merged_df["gold_attack"] = merged_df["gold_attack"].apply(substitute_attack_method)

    # load the harmfulness reasonings from the captioned datasets
    c_train = pd.read_json(root / "captioned_ver/annotations" / "train.jsonl", lines=True)
    c_train["c_split"] = "train"
    c_test = pd.read_json(root / "captioned_ver/annotations" / "test.jsonl", lines=True)
    c_test["c_split"] = "test"
    c_df = pd.concat([c_train, c_test])
    print(c_df.shape) 

    # merge the captioned dataset with the merged_df
    merged_df = pd.merge(how="left", left=merged_df, right=c_df[["id", "target", "reasonings"]], on="id")

    # rename columns
    merged_df = merged_df.rename(columns={
        "id": "image_id",
        "gold_attack": "method_of_attack",
        "target": "victim_group", 
        "reasonings": "harm_reasoning", 
    })
    merged_df = merged_df.drop(columns=["original_split"])

    # reformat the filepath
    merged_df["img"] = merged_df["img"].apply(lambda x: f"facebook_hateful_memes/processed_images/{x[4:]}")    
    merged_df.to_csv(root / "metadata.csv", index=False)

def _process_single_image(img_path: Path) -> tuple[str, str]:
    try:
        pil_img = Image.open(img_path)
    except:
        print(f"Error opening {img_path}, removing...")
        return None, None
    
    if pil_img.format == "GIF":
        pil_img = extract_frame_from_gif(pil_img, frame_ratio=0.3)

    try:
        pil_img = pil_img.convert('RGB')
    except OSError:
        print(f'Error converting rgb for {img_path}, removing...')
        pil_img.close()
        return None, None
    else:
        new_fp = output_imgs_root / f"{img_path.stem}.png"
        new_fp.parent.mkdir(parents=True, exist_ok=True)
        if pil_img.size[0] > 500 or pil_img.size[1] > 500:
            if pil_img.size[0] > 500 or pil_img.size[1] > 500:
                pil_img.thumbnail((500, 500))
        pil_img.save(new_fp, format="png", optimize=True)
        pil_img.close()
        return img_path.stem, new_fp

def process_images():
    all_images = list(p for p in img_root.rglob("*") if p.is_file())
    process_map(_process_single_image, all_images, chunksize=1, max_workers=20)

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
    run(get_responses(processed, "facebook_hateful_memes", output_path=root/"labels.csv", is_async=True))