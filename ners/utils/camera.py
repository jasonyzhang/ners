import numpy as np


def compute_crop_parameters(image_size, bbox, image_center=None):
    """
    Computes the principal point and scaling factor for focal length given a square
    bounding box crop of an image. Note that the image size should be (width, height).

    These intrinsic parameters are used to preserve the original principal point even
    after cropping the image.

    Args:
        image_size (int or tuple): Size of image, either length of longer dimension or
            (W, H).
        bbox: Square bounding box in xyxy (4,).
        image_center: Center of projection/principal point (2,).

    Returns:
        principal_point: Coordinates in NDC using Pytorch3D convention with (1, 1)
            as upper-left (2,).
        crop_scale (float): Scaling factor for focal length.
    """
    bbox = np.array(bbox)
    b = max(bbox[2:] - bbox[:2])
    if isinstance(image_size, int):
        w = h = image_size
    else:
        w, h = image_size
        image_size = max(image_size)
    if image_center is None:
        image_center = np.array([w / 2, h / 2])
    bbox_center = (bbox[:2] + bbox[2:]) / 2
    crop_scale = b / image_size
    principal_point = 2 * (bbox_center - image_center) / b
    return principal_point, crop_scale
