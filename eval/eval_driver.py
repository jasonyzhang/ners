"""
Script to train all evaluation NeRS models.

Example:
    python -m eval.eval_driver --data-dir data/evaluation --evaluation-mode fixed

    python -m eval.eval_driver --data-dir data/evaluation --evaluation-mode in-the-wild
"""
import argparse
import os
import subprocess


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default="data/evaluation", help="Path to evaluation set."
    )
    parser.add_argument(
        "--evaluation-mode", default="fixed", choices=["fixed", "in-the-wild"]
    )
    parser.add_argument(
        "--force", action="store_true", help="If set, overwrites existing predictions."
    )
    return parser


def main(data_dir, evaluation_mode, force=False):
    base_cmd = ["python", "-m", "eval.train_evaluation_model"]
    if evaluation_mode == "fixed":
        base_cmd += ["--fix-cameras", "--camera-type", "camera_optimized"]
    elif evaluation_mode == "in-the-wild":
        base_cmd += ["--camera-type", "camera_pretrained"]
    if force:
        base_cmd += ["--force"]
    instance_ids = os.listdir(data_dir)
    for instance_id in instance_ids:
        num_images = len(os.listdir(os.path.join(data_dir, instance_id, "images")))
        for i in range(num_images):
            cmd = base_cmd + ["--instance-id", instance_id, "--camera-index", str(i)]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))