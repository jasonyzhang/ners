import os.path as osp
import shutil
import tempfile

import lpips
import numpy as np
import skimage
import torch
from cleanfid import fid

LPIPS_NET = lpips.LPIPS(net="alex", verbose=False, pretrained=True)


def compute_perceptual(image_gt, image_pred):
    """
    Compute the perceptual loss between two images.

    Args:
        image_gt (np.ndarray): The ground truth image (H, W, C).
        image_pred (np.ndarray): The predicted image (H, W, C).

    Returns:
        float: Perceptual error.
    """
    image_gt = torch.tensor(image_gt.transpose(2, 0, 1)).float() * 2 - 1
    image_pred = torch.tensor(image_pred.transpose(2, 0, 1)).float() * 2 - 1
    dist = LPIPS_NET(image_gt.unsqueeze(0), image_pred.unsqueeze(0))
    return dist.item()


def compute_psnr(image_gt, image_pred):
    """
    Compute the PSNR between two images.

    Args:
        image_gt (np.ndarray): The ground truth image (H, W, C).
        image_pred (np.ndarray): The predicted image (H, W, C).

    Returns:
        float: PSNR.
    """
    return skimage.metrics.peak_signal_noise_ratio(
        image_true=image_gt,
        image_test=image_pred,
        data_range=1,
    )


def compute_ssim(image_gt, image_pred):
    """
    Compute the SSIM between two images.

    Args:
        image_gt (np.ndarray): The ground truth image (H, W, C).
        image_pred (np.ndarray): The predicted image (H, W, C).

    Returns:
        float: SSIM.
    """
    return skimage.metrics.structural_similarity(
        im1=image_gt,
        im2=image_pred,
        data_range=1,
        channel_axis=2,
    )


def compute_mse(image_gt, image_pred):
    """
    Compute the MSE between two images.

    Args:
        image_gt (np.ndarray): The ground truth image (H, W, C).
        image_pred (np.ndarray): The predicted image (H, W, C).

    Returns:
        float: MSE.
    """
    return np.mean((image_gt - image_pred) ** 2)


def compute_fid(image_paths_gt, image_paths_pred):
    """
    Compute the FID between two directories.

    Args:
        image_paths_gt (list): List of image paths corresponding to ground truth images.
        image_paths_pred (list): List of image paths corresponding to predicted images.

    Returns:
        float: FID.
    """
    dir_gt = tempfile.TemporaryDirectory()
    dir_pred = tempfile.TemporaryDirectory()
    for i, image_path in enumerate(image_paths_gt):
        filename = f"{i:06d}_{osp.basename(image_path)}"
        shutil.copy(image_path, osp.join(dir_gt.name, filename))
    for i, image_path in enumerate(image_paths_pred):
        filename = f"{i:06d}_{osp.basename(image_path)}"
        shutil.copy(image_path, osp.join(dir_pred.name, filename))
    fid_score = fid.compute_fid(dir_gt.name, dir_pred.name, verbose=False)
    dir_gt.cleanup()
    dir_pred.cleanup()
    return fid_score
