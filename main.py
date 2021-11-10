"""
Driver script for NeRS.

Directory should contain images and masks. Masks can either be in a subdirectory named
'masks' or saved in a json file (annotations.json) in RLE format. See MVMC dataset for
an example. If running on your own images, the json file should contain initial poses
and 3D cuboid extents for the initialization.

instance_dir
|_ images
|  |_ img1.jpg
|  |_ ...
|_ masks
|  |_ img1.png  (Same filename as corresponding image)
|  |_ ...
|_ [  annotations.json (if using MVMC)  ]
|_ [  metadata.json (if not using MVMC)  ]

Usage:
    python main.py \
        --instance-dir <path to instance directory> \
        [--output-dir <path to output directory>] \
        [--predict-illumination] \
        [--export-mesh] \
        [--symmetrize]

Example:
    python main.py \
        --instance-dir data/espresso --symmetrize --export-mesh --predict-illumination
"""
import argparse
import os
import os.path as osp

import torch

from ners import Ners
from ners.data import load_car_data, load_data_from_dir
from ners.models import TemplateUV, load_car_model, pretrain_template_uv


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance-dir", type=str, required=True, help="Path to instance directory."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to output directory (Defaults to output/<instance directory>).",
    )
    parser.add_argument(
        "--mvmc", action="store_true", help="If set, uses MVMC dataset loader."
    )
    parser.add_argument(
        "--export-mesh",
        action="store_true",
        help="If set, exports textured mesh to an obj file.",
    )
    parser.add_argument(
        "--force", action="store_true", help="If set, overwrites existing predictions."
    )
    parser.add_argument(
        "--predict-illumination",
        action="store_true",
        dest="predict_illumination",
        help="If True, predicts an environment map to model illumination.",
    )
    parser.add_argument(
        "--no-predict-illumination",
        action="store_false",
        dest="predict_illumination",
    )
    parser.add_argument(
        "--symmetrize",
        action="store_true",
        help="If set, makes object symmetric about the y-z plane axis",
    )
    parser.add_argument(
        "--num-frames",
        default=360,
        type=int,
        help="Number of frames for video visualization.",
    )
    # Hyperparameters
    parser.add_argument(
        "--num-iterations-camera",
        default=500,
        type=int,
        help="Number of iterations to optimize camera pose.",
    )
    parser.add_argument(
        "--num-iterations-shape",
        default=500,
        type=int,
        help="Number of iterations to optimize object shape.",
    )
    parser.add_argument(
        "--num-iterations-texture",
        default=3000,
        type=int,
        help="Number of iterations to learn texture network.",
    )
    parser.add_argument(
        "--num-iterations-radiance",
        default=500,
        type=int,
        help="Number of iterations to learn illumination.",
    )
    parser.add_argument(
        "--fov-init", default=60.0, type=float, help="Initial field of view."
    )
    parser.add_argument(
        "--L", type=int, default=10, help="Number of bases for positiional encoding."
    )
    parser.add_argument(
        "--num-layers-shape", type=int, default=4, help="Number of layers in f_shape."
    )
    parser.add_argument(
        "--num-layers-tex", type=int, default=12, help="Number of layers in f_tex."
    )
    parser.add_argument(
        "--num-layers-env", type=int, default=4, help="Number of layers in f_env."
    )
    parser.set_defaults(predict_illumination=True)
    return parser


def main(args):
    print(args)

    instance_dir = args.instance_dir
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = osp.join("output", osp.basename(instance_dir))
    os.makedirs(output_dir, exist_ok=True)
    weights_path = osp.join(output_dir, "weights.pth")
    if not args.force and osp.exists(weights_path):
        print(
            "Weights already exist at {}. Use --force to override.".format(weights_path)
        )
        return
    print("Saving weights to {}".format(weights_path))

    num_gpus = torch.cuda.device_count()
    gpu_ids = list(range(num_gpus - 1))
    gpu_id_illumination = num_gpus - 1

    if args.mvmc:
        data = load_car_data(instance_dir, use_optimized_cameras=True, image_size=256)
        f_template = load_car_model()
    else:
        data = load_data_from_dir(instance_dir, image_size=256, pad_size=0.1)
        if "extents" not in data:
            raise ValueError(
                "For your own objects, please specify the cuboid extents in "
                "metadata.json."
            )
        f_template = TemplateUV(L=10)
        f_template = pretrain_template_uv(f_template, extents=data["extents"])
    ners = Ners(
        images=data["images"],
        masks=data["masks"],
        masks_dt=data["masks_dt"],
        initial_poses=data["initial_poses"],
        image_center=data["image_centers"],
        crop_scale=data["crop_scales"],
        f_template=f_template,
        fov=args.fov_init,
        jitter_uv=True,
        gpu_ids=gpu_ids,
        gpu_id_illumination=gpu_id_illumination,
        L=args.L,
        symmetrize=args.symmetrize,
        num_layers_shape=args.num_layers_shape,
        num_layers_tex=args.num_layers_tex,
        num_layers_env=args.num_layers_env,
    )
    name = osp.basename(instance_dir)
    ners.visualize_input_views(
        filename=osp.join(output_dir, f"{name}_1_initial_cameras.jpg"),
        title=f"{name} Initial Cameras",
    )
    ners.optimize_camera(num_iterations=args.num_iterations_camera)
    ners.visualize_input_views(
        filename=osp.join(output_dir, f"{name}_2_optimized_cameras.jpg"),
        title=f"{name} Optimized Cameras",
    )
    ners.optimize_shape(num_iterations=args.num_iterations_shape)
    ners.visualize_input_views(
        filename=osp.join(output_dir, f"{name}_3_optimized_shape.jpg"),
        title=f"{name} Optimized Shape",
    )
    ners.optimize_texture(num_iterations=args.num_iterations_texture)
    ners.visualize_input_views(
        filename=osp.join(output_dir, f"{name}_4_optimized_texture.jpg"),
        title=f"{name} Optimized Texture",
    )
    if args.export_mesh:
        mesh_name = osp.join(output_dir, f"{name}_mesh.obj")
        ners.save_obj(mesh_name)
    ners.make_video(
        osp.join(output_dir, f"{name}_video_texture_only"),
        use_antialiasing=True,
        visuals=("nn", "albedo"),
        num_frames=args.num_frames,
    )
    if args.predict_illumination:
        torch.cuda.empty_cache()
        ners.optimize_radiance(num_iterations=args.num_iterations_radiance)

        ners.visualize_input_views(
            filename=osp.join(output_dir, f"{name}_5_optimized_radiance.jpg"),
            title=f"{name} Optimized Radiance",
        )
        ners.make_video(
            osp.join(output_dir, f"{name}_video"),
            use_antialiasing=True,
            visuals=("nn", "full", "albedo", "lighting"),
            num_frames=args.num_frames,
        )
    ners.save_parameters(weights_path)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
