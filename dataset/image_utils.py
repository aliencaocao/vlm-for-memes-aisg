import base64
import io
import math
import os
from typing import Union

from PIL import Image
from PIL.Image import Image as ImageType


def PNGSaveWithTargetSize(im: ImageType, target: int) -> io.BytesIO:
    """Save the image as png with the given name at best quality that makes less than "target" bytes"""
    # Min and Max quality
    Qmin, Qmax = 1, 95
    # Highest acceptable quality found
    Qacc = -1
    while Qmin <= Qmax:
        m = math.floor((Qmin + Qmax) / 2)
        # Encode into memory and get size
        buffer = io.BytesIO()
        im.save(buffer, format="png", quality=m, optimize=True)
        s = buffer.getbuffer().nbytes
        if s <= target:
            Qacc = m
            Qmin = m + 1
        elif s > target:
            Qmax = m - 1
    # Write to disk at the defined quality
    if Qacc > -1:
        buffer = io.BytesIO()
        im.save(buffer, format="png", quality=Qacc, optimize=True)
        return buffer
    else:
        raise Exception('Could not compress image to target size!')


def resize_with_compression(image_base64: str, target_size: int = 10 * 1024 * 1024, min_resolution: int = -1, max_resolution: int = math.inf) -> str:
    im = Image.open(io.BytesIO(base64.b64decode(image_base64)))
    # make sure no sides of image is shorter than 15 pixels or longer than 8192 pixels, else resize while keeping aspect ratio
    if min(im.size) < min_resolution or max(im.size) > max_resolution:
        if min(im.size) < min_resolution:
            im.thumbnail((min_resolution, min_resolution))
        elif max(im.size) > max_resolution:
            im.thumbnail((max_resolution, max_resolution))
        buffer = io.BytesIO()
        im.save(buffer, format="png", quality=99, optimize=True)
        image_base64 = base64.b64encode(buffer.getbuffer()).decode()

    if len(image_base64) > target_size:  # limit is 10MB
        im = Image.open(io.BytesIO(base64.b64decode(image_base64)))
        image_base64 = base64.b64encode(PNGSaveWithTargetSize(im, target_size).getbuffer()).decode()
    return image_base64


def ensure_format(image_base64: str, acceptable_formats: list[str], return_bytes: bool = False) -> Union[ImageType, bytes]:
    image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
    f = image.format.upper()
    if f not in acceptable_formats:
        image = image.convert('RGB')
        temp_bytes = io.BytesIO()
        image.save(temp_bytes, format="png", quality=99, optimize=True)
    if return_bytes:
        if f in acceptable_formats:
            return base64.b64decode(image_base64)
        else:
            return temp_bytes.getvalue()
    else:
        if f in acceptable_formats:
            return image
        else:
            return Image.open(temp_bytes)


def process_single_image(img: str) -> tuple[str, str]:
    pil_img = Image.open(img)
    try:
        pil_img = pil_img.convert('RGB')
    except OSError:
        print(f'Error converting rgb for {img}, removing...')
        pil_img.close()
        os.remove(img)
        return img.stem, None
    else:
        new_fp = ''.join(str(img).split('.')[:-1]) + '.png'
        if pil_img.size[0] > 500 or pil_img.size[1] > 500 or img.suffix != '.png':
            if pil_img.size[0] > 500 or pil_img.size[1] > 500:
                pil_img.thumbnail((500, 500))
            pil_img.save(new_fp, format='png', optimize=True)
            pil_img.close()
            if img.suffix != '.png':
                os.remove(img)
        else:
            pil_img.close()
        return img.stem, new_fp

def extract_frame_from_gif(img: ImageType, frame_ratio: float=0.3) -> ImageType:
    total_frames = img.n_frames
    if total_frames == 1:
        img.seek(0)
        return img.copy()
    target_frame = max(1, int(total_frames * frame_ratio)) # ensure at least 1 frame is extracted
    img.seek(target_frame)
    return img.copy()