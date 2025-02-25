import os
from pathlib import Path
import uuid
import pandas as pd
from PIL import Image
from tqdm.contrib.concurrent import process_map

from gpt import get_responses, DatasetImage
from asyncio import run
from image_utils import extract_frame_from_gif

root = Path("bawankar_reddit_memes_and_comments")
img_root = root / "meme-dataset" / "memes"

# Code to print statistics about the dataset
# s = ""
# total_gif = 0
# total_jpg = 0
# total_png = 0
# for child_folder in img_root.iterdir():
#     s += f"* r/{child_folder.name}"
#     gif_count = 0
#     jpg_count = 0
#     png_count = 0
#     imgs = list(child_folder.iterdir())
#     for img_path in imgs:
#         img = Image.open(img_path)
#         if img.format == "GIF":
#             gif_count += 1
#         elif img.format == "JPEG":
#             jpg_count += 1
#         elif img.format == "PNG":
#             png_count += 1
#         img.close()
#         del img
#     total_gif += gif_count
#     total_jpg += jpg_count
#     total_png += png_count
#     s += f": {len(imgs)} ({gif_count} GIF, {jpg_count} JPG, {png_count} PNG)\n"
# s += f"* Total: {total_gif + total_jpg + total_png} ({total_gif} GIF, {total_jpg} JPG, {total_png} PNG)"
# print(s)

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
        subreddit_name = img_path.parent.name
        new_id = str(uuid.uuid4())
        new_fp = output_imgs_root / subreddit_name / f"{new_id}.png"
        new_fp.parent.mkdir(parents=True, exist_ok=True)
        if pil_img.size[0] > 500 or pil_img.size[1] > 500:
            if pil_img.size[0] > 500 or pil_img.size[1] > 500:
                pil_img.thumbnail((500, 500))
        pil_img.save(new_fp, format="png", optimize=True)
        pil_img.close()
        return new_id, new_fp

def process_images():
    all_images = list(img_root.rglob("*.jpg"))
    processed = process_map(_process_single_image, all_images, chunksize=1)
    df = pd.DataFrame.from_records(processed, columns=["image_id", "img"])
    df.to_csv(root / "metadata.csv", index=False)
    print(f"Number of images in metadata.csv: {len(df)}")
    print(f"Number of images in processed_images: {len(list(output_imgs_root.rglob('*.png')))}")

if __name__ == "__main__":
    processed = pd.read_csv(root / "metadata.csv")
    processed["image_id"] = processed["image_id"].astype(str)
    processed = [DatasetImage(
            id=x["image_id"],
            path=x["img"],
            lang=x["lang"],
    ) for _, x in processed.iterrows()]
    run(get_responses("bawankar_reddit_memes_and_comments", processed, output_path=root / "labels.csv", is_async=True))