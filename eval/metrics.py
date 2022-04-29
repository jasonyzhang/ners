import cleanfid
import lpips
import numpy as np
import skimage
import torch

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
        # multichannel=True,
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


def compute_fid(directory_gt, directory_pred):
    """
    Compute the FID between two directories.

    Args:
        directory_gt (str): The directory containing the ground truth images.
        directory_pred (str): The directory containing the predicted images.

    Returns:
        float: FID.
    """
    return cleanfid.compute_fid(directory_gt, directory_pred)
