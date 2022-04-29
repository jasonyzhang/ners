"""
Evaluation script for Novel View Synthesis.

For a given instance with N images, we are evaluating the model trained with N-1 images
to generate the Nth image. The convention is that frame_i corresponds to the held out
target view.

The director structure should be:

eval_dir
|_ instance_id
   |_ images_masked
   |  |_ image_00.png
   |  |_ image_01.png
   |  |_ ...
   |_ pred_name
      |_ image_00.png
      |_ image_01.png
      |_ ...

Computed metrics are: MSE, PSNR, SSIM, LPIPS, FID.

Example:
    python -m eval.eval --eval-dir data/evaluation \
        --gt-name images_masked --pred-name ners_fixed --print-per-instance
"""
import argparse
import os
import os.path as osp
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from eval.metrics import (
    compute_fid,
    compute_mse,
    compute_perceptual,
    compute_psnr,
    compute_ssim,
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir", default="data/evaluation", help="Path to evaluation set."
    )
    parser.add_argument(
        "--gt-name",
        default="images_masked",
        help="Name of folder containing ground truth images.",
    )
    parser.add_argument(
        "--pred-name",
        default="ners_fixed",
        help="Name of folder containing predicted images.",
    )
    parser.add_argument(
        "--print-per-instance",
        action="store_true",
        help="Print metrics per instance.",
    )
    return parser


def evaluate(eval_dir, gt_name, pred_name, print_per_instance=False):
    instance_ids = os.listdir(eval_dir)

    print(f"{'Name':12s} {'MSE':>6s} {'PSNR':>6s} {'SSIM':>6s} {'LPIPS':>6s}")

    all_metrics = {"mse": [], "psnr": [], "ssim": [], "lpips": []}
    for instance_id in instance_ids:
        gt_dir = osp.join(eval_dir, instance_id, gt_name)
        pred_dir = osp.join(eval_dir, instance_id, pred_name)
        if not osp.isdir(gt_dir) or not osp.isdir(pred_dir):
            continue
        gt_images = sorted(glob(osp.join(gt_dir, "*.png")))
        pred_images = sorted(glob(osp.join(pred_dir, "*.png")))
        assert len(gt_images) == len(pred_images)
        metrics = {
            "mse": [],
            "psnr": [],
            "ssim": [],
            "lpips": [],
        }
        for gt_image_path, pred_image_path in zip(gt_images, pred_images):
            gt_image = plt.imread(gt_image_path)[..., :3]
            pred_image = plt.imread(pred_image_path)[..., :3]

            metrics["mse"].append(compute_mse(gt_image, pred_image))
            metrics["psnr"].append(compute_psnr(gt_image, pred_image))
            metrics["ssim"].append(compute_ssim(gt_image, pred_image))
            metrics["lpips"].append(compute_perceptual(gt_image, pred_image))

        for metric, values in metrics.items():
            mean_metric = np.mean(values)
            metrics[metric] = mean_metric
            all_metrics[metric].append(mean_metric)

        if print_per_instance:
            metrics_str = " ".join(["{0:>#6.3g}".format(v) for v in metrics.values()])
            print(f"{instance_id:12s} {metrics_str}")
    metrics_str = " ".join(
        ["{0:>#6.3g}".format(np.mean(v)) for v in all_metrics.values()]
    )
    print(f"{pred_name:12s} {metrics_str}")


if __name__ == "__main__":
    args = get_parser().parse_args()
    evaluate(
        eval_dir=args.eval_dir,
        gt_name=args.gt_name,
        pred_name=args.pred_name,
        print_per_instance=args.print_per_instance,
    )
