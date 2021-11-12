"""
Script to pretrain shape template. Requires an object mesh and a sphere mesh with
corresponding vertices.

It is recommended that the object mesh when rendered with an identity camera is facing
forward.

Usage:
    python -m scripts.pretrain_shape_template \
        --object-mesh <path.obj> \
        --sphere-mesh <path.obj> \
        --output-path <path.pth> \
        [--visualize-path <path.obj>]

Example:
    python -m scripts.pretrain_shape_template \
        --object-mesh models/meshes/car.obj \
        --sphere-mesh models/meshes/car_sphere.obj \
        --output-path models/templates/car.pth \
        --visualize-path models/templates/car.obj
"""
import argparse
import os.path as osp

import torch
import numpy as np
from tqdm.auto import tqdm
import trimesh

import pytorch3d
import pytorch3d.io
from pytorch3d.loss import (
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
    chamfer_distance,
)
from ners.models import TemplateUV
from ners.utils import create_sphere, random_rotation, sample_consistent_points


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object-mesh",
        type=str,
        required=True,
        default="models/meshes/car.obj",
        help="Path to object mesh.",
    )
    parser.add_argument(
        "--sphere-mesh",
        type=str,
        required=True,
        default="models/meshes/car_sphere.obj",
        help="Path to sphere mesh.",
    )
    parser.add_argument(
        "--visualize-path",
        type=str,
        default="",
        help="If an obj path is specified, will visualize the implicit template model.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        default="models/templates/car.pth",
        help="Path to output template weights.",
    )
    parser.add_argument("--num-iterations", type=int, default=5000)
    parser.add_argument("--num-samples", type=int, default=15000)
    return parser


def main(args):
    print(args)
    device = torch.device("cuda:0")
    verts_obj, faces_obj, *_ = pytorch3d.io.load_obj(args.object_mesh)
    verts_sphere, faces_sphere, *_ = pytorch3d.io.load_obj(args.sphere_mesh)
    # Center and rescale
    verts_obj -= (verts_obj.max(dim=0).values + verts_obj.min(dim=0).values) / 2
    verts_obj /= verts_obj.max()
    verts_sphere = verts_sphere / (verts_sphere.norm(dim=1, keepdim=True) + 1e-6)

    # Need to transform meshes such that identity camera corresponds to back.
    rot = pytorch3d.transforms.euler_angles_to_matrix(
        torch.as_tensor([np.pi / 2, np.pi / 2, 0]), "YZX"
    )
    verts_sphere = verts_sphere @ rot
    verts_obj = verts_obj @ rot

    template_uv = TemplateUV().to(device)
    optim = torch.optim.Adam(template_uv.parameters(), lr=1e-4)

    num_iterations = args.num_iterations
    num_samples = args.num_samples

    sphere_vs, sphere_fs = create_sphere(5)
    sphere_vs = sphere_vs.to(device)
    sphere_fs = sphere_fs.to(device)

    for _ in tqdm(range(num_iterations)):
        optim.zero_grad()
        targets, uvs = sample_consistent_points(
            verts_obj, faces_obj.verts_idx, [verts_obj, verts_sphere], num_samples
        )
        pred_vs = template_uv(uvs.to(device), normalize=True)
        sv = sphere_vs @ random_rotation(sphere_vs.device)
        sv = sv.unsqueeze(0)  # (1, V, 3)
        meshes = pytorch3d.structures.Meshes(
            template_uv(sv, normalize=True), sphere_fs.unsqueeze(0)
        )
        loss_reconstruction = torch.mean((pred_vs - targets.to(device)) ** 2)
        loss_laplacian = mesh_laplacian_smoothing(meshes)
        loss_normal = mesh_normal_consistency(meshes)
        loss_chamfer = chamfer_distance(
            template_uv(sv), targets.unsqueeze(0).to(device)
        )[0]
        loss = (
            loss_chamfer
            + loss_reconstruction
            + 0.005 * loss_laplacian
            + 0.005 * loss_normal
        )
        loss.backward()
        optim.step()
    template_uv = template_uv.cpu()
    torch.save(template_uv.state_dict(), args.output_path)

    if args.visualize_path:
        tmesh = trimesh.Trimesh(
            vertices=template_uv(sphere_vs).detach().cpu().numpy(),
            faces=sphere_fs.cpu().numpy(),
            vertex_colors=(sphere_vs.detach().cpu().numpy() + 1) / 2,
        )
        with open(args.visualize_path, "w") as f:
            trimesh.exchange.export.export_mesh(tmesh, f, file_type="obj")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
