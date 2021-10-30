"""
Class for custom Rasterizer for Pytorch3D. The main difference is that we need to keep
around the camera centers in order to compute view direction.
"""
from typing import NamedTuple

import pytorch3d
import torch
import torch.nn
from pytorch3d.renderer.mesh import rasterize_meshes


class Fragments(NamedTuple):
    pix_to_face: torch.Tensor
    zbuf: torch.Tensor
    bary_coords: torch.Tensor
    dists: torch.Tensor
    cameras_centers: torch.Tensor


class MeshRasterizer(pytorch3d.renderer.mesh.rasterizer.MeshRasterizer):
    def forward(self, meshes_world, **kwargs) -> Fragments:
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                          coordinates in world space.
        Returns:
            Fragments: Rasterization outputs as a named tuple.
        """
        meshes_screen = self.transform(meshes_world, **kwargs)
        raster_settings = kwargs.get("raster_settings", self.raster_settings)

        # By default, turn on clip_barycentric_coords if blur_radius > 0.
        # When blur_radius > 0, a face can be matched to a pixel that is outside the
        # face, resulting in negative barycentric coordinates.
        clip_barycentric_coords = raster_settings.clip_barycentric_coords
        if clip_barycentric_coords is None:
            clip_barycentric_coords = raster_settings.blur_radius > 0.0

        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
            clip_barycentric_coords=clip_barycentric_coords,
            cull_backfaces=raster_settings.cull_backfaces,
        )
        cameras = kwargs.get("cameras", self.cameras)
        cameras_centers = cameras.get_camera_center().repeat(  # (K, H, W, N, 3)
            raster_settings.faces_per_pixel,
            raster_settings.image_size,
            raster_settings.image_size,
            1,
            1,
        )

        cameras_centers = cameras_centers.transpose(0, 3)  # (N, H, W, K, 3)
        return Fragments(
            pix_to_face=pix_to_face,
            zbuf=zbuf,
            bary_coords=bary_coords,
            dists=dists,
            cameras_centers=cameras_centers,
        )
