import numpy as np
from PIL import Image


def antialias(image):
    """
    Performs antialiasing on image by downsampling using Lanczos filter.

    Args:
        image (PIL.Image or np.ndarray): Image to be antialiased.

    Returns:
        Image or ndarray: Antialiased image.
    """
    is_image = isinstance(image, Image.Image)
    if not is_image:
        image = Image.fromarray((image.clip(0, 1) * 255).astype(np.uint8))
    shape = np.array(image.size[:2]) // 2
    image = image.resize(shape, Image.LANCZOS)
    if not is_image:
        image = np.array(image) / 255.0
    return image


def crop_image(image, bbox):
    """
    Crops PIL image using bounding box.

    Args:
        image (PIL.Image): Image to be cropped.
        bbox (tuple): Integer bounding box (xyxy).
    """
    bbox = np.array(bbox)
    if image.mode == "RGB":
        default = (255, 255, 255)
    elif image.mode == "RGBA":
        default = (255, 255, 255, 255)
    else:
        default = 0
    bg = Image.new(image.mode, (bbox[2] - bbox[0], bbox[3] - bbox[1]), default)
    bg.paste(image, (-bbox[0], -bbox[1]))
    return bg
