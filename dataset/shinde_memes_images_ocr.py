import os
from pathlib import Path
import ast
import pandas as pd
from PIL import Image
from tqdm.contrib.concurrent import process_map

from gpt import get_responses, DatasetImage
from asyncio import run
from image_utils import extract_frame_from_gif

root = Path("shinde_memes_images_ocr_data")
img_root = root / "SampleImagesData"

output_imgs_root = root / "processed_images"
output_imgs_root.mkdir(parents=True, exist_ok=True)

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
    processed = process_map(_process_single_image, all_images, chunksize=1, max_workers=10)
    df = pd.DataFrame.from_records(processed, columns=["image_id", "img"])
    df = df.dropna()
    print(f"Number of images in metadata.csv: {len(df)}")
    print(f"Number of images in processed_images: {len(list(output_imgs_root.rglob('*.png')))}")

    train = pd.read_csv(root / "text_with_ocr.csv")
    val = pd.read_csv(root / "validation_data.csv")
    combined = pd.concat([train, val]).reset_index(drop=True)
    combined["image_id"] = combined["image"].apply(lambda x: x[:-4])
    selected_rows = combined[combined["image_id"].isin(df["image_id"])]
    df = df.merge(selected_rows, on="image_id")
    df = df.rename(columns={"OCR": "text"})

    df.to_csv(root / "metadata.csv", index=False)

if __name__ == "__main__":
    processed = pd.read_csv(root / "metadata.csv")
    processed["image_id"] = processed["image_id"].astype(str)
    processed = [DatasetImage(
            id=x["image_id"],
            path=x["img"],
            text=x["text"],
            lang=x["lang"],
            label=x["label"],
            victim_group=None if pd.isna(x["victim_group"]) else ast.literal_eval(x["victim_group"]),
            method_of_attack=None if pd.isna(x["method_of_attack"]) else ast.literal_eval(x["method_of_attack"]),
    ) for _, x in processed.iterrows()]
    run(get_responses("shinde_memes_images_ocr_data", processed, output_path=root / "labels.csv", is_async=True))
