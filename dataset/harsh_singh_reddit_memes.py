from PIL import Image
from pathlib import Path
import uuid
import pandas as pd
from PIL import Image
from tqdm.contrib.concurrent import process_map

from gpt import get_responses, DatasetImage
from asyncio import run
from image_utils import extract_frame_from_gif

root = Path("harsh_singh_reddit_memes")
img_root = root / "pruned"

# Code to remove duplicate images
# def get_image_hash(image_path):
#     with open(image_path, 'rb') as f:
#         image_bytes = f.read()
#         image_hash = hashlib.md5(image_bytes).hexdigest()
#     return image_hash
# image_hashes = set()
# for raw_img in raw_img_root.iterdir():
#     image_hash = get_image_hash(raw_img)

#     if image_hash in image_hashes:
#         print(f"Duplicate image: hash of {raw_img} has been seen before, skipping...")
#     else:
#         image_hashes.add(image_hash)
#         # copy the image to the pruned folder
#         pruned_img = pruned_img_root / raw_img.name
#         os.system(f"cp {raw_img} {pruned_img}")

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
        new_id = str(uuid.uuid4())
        new_fp = output_imgs_root / f"{new_id}.png"
        new_fp.parent.mkdir(parents=True, exist_ok=True)
        if pil_img.size[0] > 500 or pil_img.size[1] > 500:
            if pil_img.size[0] > 500 or pil_img.size[1] > 500:
                pil_img.thumbnail((500, 500))
        pil_img.save(new_fp, format="png", optimize=True)
        pil_img.close()
        return new_id, new_fp

def process_images():
    all_images = list(p for p in img_root.rglob("*") if p.is_file())
    processed = process_map(_process_single_image, all_images, chunksize=1, max_workers=10)
    df = pd.DataFrame.from_records(processed, columns=["image_id", "img"])
    df = df.dropna()
    print(f"Number of images in metadata.csv: {len(df)}")
    print(f"Number of images in processed_images: {len(list(output_imgs_root.rglob('*.png')))}")
    df.to_csv(root / "metadata.csv", index=False)
    
if __name__ == "__main__":
    processed = pd.read_csv(root / "metadata.csv")
    processed["image_id"] = processed["image_id"].astype(str)
    processed = [DatasetImage(
            id=x["image_id"],
            path=x["img"],
            lang="en",
    ) for _, x in processed.iterrows()]
    run(get_responses("harsh_singh_reddit_memes", processed, output_path=root / "labels.csv", is_async=True))