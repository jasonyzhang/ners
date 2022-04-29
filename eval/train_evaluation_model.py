"""
Script to train NeRS models for novel view synthesis.

Make sure to first download the evaluation dataset, and place it in data/evaluation.

Train a model for instance id 7251103879 with camera index 0 held out using the
"Fixed" camera evaluation:

python -m eval.train_evaluation_model --data_dir data/evaluation --fix-cameras \
    --instance-id 7251103879 --camera-index 0 --camera-type camera_optimized

Train a model for instance id 7251103879 with camera index 0 held out using the
"In-the-wild" camera evaluation:

python -m eval.train_evaluation_model --data_dir data/evaluation \
    --instance-id 7251103879 --camera-index 0 --camera-type camera_pretrained

"""
import argparse
import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.renderer import RasterizationSettings

from ners import Ners
from ners.data import load_car_data
from ners.models import load_car_model
from ners.pytorch3d import PerspectiveCameras
from ners.utils.image import antialias


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="data/evaluation", help="Path to evaluation set."
    )
    parser.add_argument("--instance-id", type=str, required=True)
    parser.add_argument("--camera-index", type=int, required=True)
    parser.add_argument("--fix-cameras", action="store_true")
    parser.add_argument(
        "--camera-type",
        type=str,
        default="camera_optimized",
        choices=["camera_optimized", "camera_pretrained"],
    )
    parser.add_argument("--output-dir", type=str, default="output/eval")
    parser.add_argument(
        "--force", action="store_true", help="If set, overwrites existing predictions."
    )

    # Hyperparameters
    parser.add_argument("--num-iterations-camera", default=500, type=int)
    parser.add_argument("--num-iterations-shape", default=500, type=int)
    parser.add_argument("--num-iterations-texture", default=3000, type=int)
    parser.add_argument("--num-iterations-radiance", default=500, type=int)
    parser.add_argument("--num-layers-tex", default=12, type=int)
    return parser


def render_target_view(ners, target_camera, image_size, use_antialiasing=True):
    torch.cuda.empty_cache()
    if use_antialiasing:
        image_size *= 2
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
        perspective_correct=False,
    )
    with torch.no_grad():
        target_camera = target_camera.to(ners.device)
        rend = ners.renderer_textured(
            ners.meshes_current, cameras=target_camera, raster_settings=raster_settings
        )
        image = np.clip(rend.detach().cpu().numpy()[0, ..., :3], 0, 1)
        if use_antialiasing:
            image = antialias(image)
    return image


def skip_index(data, skip_index):
    if isinstance(data, list):
        return data[:skip_index] + data[skip_index + 1 :]
    else:
        return np.concatenate((data[:skip_index], data[skip_index + 1 :]), axis=0)


def load_cameras(instance_dir, camera_type):
    annotations_json = osp.join(instance_dir, "annotations.json")
    with open(annotations_json) as f:
        annotations = json.load(f)

    all_cameras = {
        "R": [],
        "T": [],
        "fov": [],
    }
    for annotation in annotations["annotations"]:
        camera = annotation[camera_type]
        all_cameras["R"].append(camera["R"])
        all_cameras["T"].append(camera["T"])
        all_cameras["fov"].append(camera["fov"])
    return all_cameras


def main(
    data_dir,
    instance_id,
    camera_index,
    camera_type,
    output_dir,
    fix_cameras,
    num_iterations_camera,
    num_iterations_shape,
    num_iterations_texture,
    num_iterations_radiance,
    num_layers_tex,
    force=False,
):
    instance_dir = osp.join(data_dir, instance_id)
    name = f"{instance_id}_{camera_type}"
    if fix_cameras:
        name += "_fix"
    output_dir = osp.join(output_dir, name)
    os.makedirs(output_dir, exist_ok=True)
    weights_path = osp.join(output_dir, f"weights_{camera_index:02d}.pth")
    render_path = osp.join(output_dir, f"render_{camera_index:02d}.png")
    if osp.exists(weights_path) and not force:
        print(f"Found weights at {weights_path}")
        return

    data = load_car_data(instance_dir, image_size=256)
    f_template = load_car_model()
    images = skip_index(data["images"], camera_index)
    masks = skip_index(data["masks"], camera_index)
    masks_dt = skip_index(data["masks_dt"], camera_index)
    image_center = skip_index(data["image_centers"], camera_index)
    crop_scale = skip_index(data["crop_scales"], camera_index)

    all_cameras = load_cameras(instance_dir, camera_type)
    cameras_training = PerspectiveCameras(
        R=skip_index(all_cameras["R"], camera_index),
        T=skip_index(all_cameras["T"], camera_index),
        fov=skip_index(all_cameras["fov"], camera_index),
        image_center=image_center,
        crop_scale=crop_scale,
    )
    cameras_target = PerspectiveCameras(
        R=[all_cameras["R"][camera_index]],
        T=[all_cameras["T"][camera_index]],
        fov=[all_cameras["fov"][camera_index]],
        image_center=[data["image_centers"][camera_index]],
        crop_scale=[data["crop_scales"][camera_index]],
    )
    ners = Ners(
        images=images,
        masks=masks,
        masks_dt=masks_dt,
        initial_poses=cameras_training.R.tolist(),
        crop_scale=crop_scale,
        image_center=image_center,
        symmetrize=True,
        num_layers_shape=4,
        num_layers_tex=num_layers_tex,
        num_layers_env=4,
        f_template=f_template,
    )
    ners.cameras_current = cameras_training.to(ners.device)
    ners.visualize_input_views(
        filename=osp.join(output_dir, f"1_initial_cameras.jpg"),
    )
    if fix_cameras:
        ners.finetune_camera = False
    else:
        ners.optimize_camera(num_iterations_camera)
        ners.visualize_input_views(
            filename=osp.join(output_dir, f"2_optimize_cameras.jpg"),
        )

    ners.optimize_shape(num_iterations_shape)
    ners.visualize_input_views(
        filename=osp.join(output_dir, f"3_shape.jpg"),
    )
    ners.optimize_texture(num_iterations_texture)
    ners.visualize_input_views(
        filename=osp.join(output_dir, f"4_texture.jpg"),
    )
    ners.optimize_radiance(num_iterations_radiance)
    ners.visualize_input_views(
        filename=osp.join(output_dir, f"5_rad.jpg"),
    )
    target_image = render_target_view(ners, cameras_target, image_size=256)
    plt.imsave(render_path, target_image)
    ners.save_parameters(weights_path)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))
