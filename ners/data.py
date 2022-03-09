import json
import os.path as osp
from glob import glob

import numpy as np
import pytorch3d
from PIL import Image

import ners.utils.image as image_util
from ners.utils import (
    compute_crop_parameters,
    compute_distance_transform,
    rle_to_binary_mask,
)


def get_bbox(img):
    a = np.where(img != 0)
    bbox = np.min(a[1]), np.min(a[0]), np.max(a[1]) + 1, np.max(a[0]) + 1
    return np.array(bbox)


def load_data_from_dir(instance_dir, image_size=256, pad_size=0.1, skip_indices=()):
    """
    Loads NeRS data from a directory. Assumes that a folder containing images and a
    folder container masks. Mask names should be the same as the images.
    """
    image_dir = osp.join(instance_dir, "images")
    mask_dir = osp.join(instance_dir, "masks")
    data_dict = {
        "images_og": [],
        "images": [],
        "masks": [],
        "masks_dt": [],
        "bbox": [],
        "image_centers": [],
        "crop_scales": [],
        "azimuths": [],
        "elevations": [],
    }
    for i, image_path in enumerate(sorted(glob(osp.join(image_dir, "*.png")))):
        if i in skip_indices:
            continue
        image_name = osp.basename(image_path)
        name_parts = image_name.split(".")[0].split("_")
        elev = int(name_parts[2].split("elev")[1])
        azimuth = int(name_parts[3].split("azim")[1])
        nn_exists = False
        
        for azi in data_dict["azimuths"]:
          if (azimuth > azi - 15) and (azimuth < azi + 15):
            nn_exists = True
        
        if nn_exists:
          continue
        
        mask_path = osp.join(mask_dir, image_name.replace("jpg", "png"))
        image_og = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        image_og_flip = image_og.transpose(Image.FLIP_LEFT_RIGHT)
        mask_flip = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        bbox = get_bbox(np.array(mask) / 255.0 > 0.5)
        center = (bbox[:2] + bbox[2:]) / 2.0
        s = max(bbox[2:] - bbox[:2]) / 2.0 * (1 + pad_size)
        square_bbox = np.concatenate([center - s, center + s]).astype(int)
        # Crop image and mask.
        image = image_util.crop_image(image_og, square_bbox)
        image = np.array(image.resize((image_size, image_size), Image.LANCZOS)) / 255.0
        mask = image_util.crop_image(mask, square_bbox)
        mask = np.array(mask.resize((image_size, image_size), Image.BILINEAR))
        mask = mask / 255.0 > 0.5
        image_center, crop_scale = compute_crop_parameters(image_og.size, square_bbox)
        data_dict["bbox"].append(square_bbox)
        data_dict["crop_scales"].append(crop_scale)
        data_dict["image_centers"].append(image_center)
        data_dict["images"].append(image)
        data_dict["images_og"].append(image_og)
        data_dict["masks"].append(mask)
        data_dict["masks_dt"].append(compute_distance_transform(mask))
        data_dict["azimuths"].append(azimuth)
        data_dict["elevations"].append(elev)

        bbox = get_bbox(np.array(mask_flip) / 255.0 > 0.5)
        center = (bbox[:2] + bbox[2:]) / 2.0
        s = max(bbox[2:] - bbox[:2]) / 2.0 * (1 + pad_size)
        square_bbox = np.concatenate([center - s, center + s]).astype(int)
        # Crop image and mask_flip.
        image = image_util.crop_image(image_og_flip, square_bbox)
        image = np.array(image.resize((image_size, image_size), Image.LANCZOS)) / 255.0
        mask_flip = image_util.crop_image(mask_flip, square_bbox)
        mask_flip = np.array(mask_flip.resize((image_size, image_size), Image.BILINEAR))
        mask_flip = mask_flip / 255.0 > 0.5
        image_center, crop_scale = compute_crop_parameters(image_og_flip.size, square_bbox)
        data_dict["bbox"].append(square_bbox)
        data_dict["crop_scales"].append(crop_scale)
        data_dict["image_centers"].append(image_center)
        data_dict["images"].append(image)
        data_dict["images_og"].append(image_og_flip)
        data_dict["masks"].append(mask_flip)
        data_dict["masks_dt"].append(compute_distance_transform(mask_flip))
        data_dict["azimuths"].append(360 - azimuth)
        data_dict["elevations"].append(elev)

    for k, v in data_dict.items():
        if k != "images_og":  # Original images can have any resolution.
            data_dict[k] = np.stack(v)

    if osp.exists(osp.join(instance_dir, "metadata.json")):
        metadata = json.load(open(osp.join(instance_dir, "metadata.json")))
        data_dict["extents"] = metadata["extents"]
        azimuths = metadata["azimuths"]
        elevations = metadata["elevations"]
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=2,
            elev=elevations,
            azim=azimuths,
        )
        data_dict["initial_poses"] = R.tolist()
    return data_dict


def load_car_data(
    instance_dir, use_optimized_cameras=True, image_size=256, pad_size=0.1
):
    """
    Processes instance of car dataset for NeRS optimization.

    Args:
        instance_dir (str): Path to car instance.
        use_optimized_cameras (bool, optional): If true, uses optimized pose from NeRS.
            Otherwise, uses filtered poses from PoseFromShape.
        image_size (int, optional): Size of image crop.
        pad_size (float, optional): Amount to pad the bounding box before cropping.

    Returns:
        dict: Dictionary containing the following keys:
            "bbox": List of bounding boxes (xyxy).
            "crop_scales": List of crop scales.
            "image_centers": List of image centers.
            "images": List of cropped images.
            "images_og": List of original, uncropped images.
            "initial_poses": List of rotation matrices to initialize pose.
            "masks": List of binary masks.
    """
    annotations_json = osp.join(instance_dir, "annotations.json")
    with open(annotations_json) as f:
        annotations = json.load(f)
    data_dict = {
        "bbox": [],  # (N, 4).
        "crop_scales": [],  # (N,).
        "image_centers": [],  # (N, 2).
        "images": [],  # (N, 256, 256, 3).
        "images_og": [],  # (N, H, W, 3).
        "initial_poses": [],  # (N, 3, 3).
        "initial_trans": [],
        "masks": [],  # (N, 256, 256).
        "masks_dt": [],  # (N, 256, 256).
    }
    for annotation in annotations["annotations"]:
        filename = osp.join(instance_dir, "images", annotation["filename"])

        # # Make a square bbox.
        # bbox = np.array(annotation["bbox"])
        # center = ((bbox[:2] + bbox[2:]) / 2.0).astype(int)
        # s = (max(bbox[2:] - bbox[:2]) / 2.0 * (1 + pad_size)).astype(int)
        # square_bbox = np.concatenate([center - s, center + s])

        # Load image and mask.
        image_og = Image.open(filename).convert("RGB")
        mask = Image.fromarray(rle_to_binary_mask(annotation["mask"]), mode='L')
        
        # Make a square bbox.
        bbox = get_bbox(np.array(mask) > 0.5)
        center = (bbox[:2] + bbox[2:]) / 2.0
        s = max(bbox[2:] - bbox[:2]) / 2.0 * (1 + pad_size)
        square_bbox = np.concatenate([center - s, center + s]).astype(int)

        # Crop image and mask.
        image = image_util.crop_image(image_og, square_bbox)
        image = np.array(image.resize((image_size, image_size), Image.LANCZOS)) / 255.0
        mask = image_util.crop_image(mask, square_bbox)
        mask = np.array(mask.resize((image_size, image_size), Image.BILINEAR)) > 0.5
        image_center, crop_scale = compute_crop_parameters(image_og.size, square_bbox)
        if use_optimized_cameras:
            initial_pose = annotation["camera_optimized"]["R"]
            initial_translation = annotation["camera_optimized"]["T"]
        else:
            initial_pose = annotation["camera_initial"]["R"]
            initial_translation = annotation["camera_initial"]["T"]
        data_dict["bbox"].append(square_bbox)
        data_dict["crop_scales"].append(crop_scale)
        data_dict["image_centers"].append(image_center)
        data_dict["images"].append(image)
        data_dict["images_og"].append(image_og)
        data_dict["initial_poses"].append(initial_pose)
        data_dict["initial_trans"].append(initial_translation)
        data_dict["masks"].append(mask)
        data_dict["masks_dt"].append(compute_distance_transform(mask))
    for k, v in data_dict.items():
        if k != "images_og":  # Original images can have any resolution.
            data_dict[k] = np.stack(v)
    return data_dict
