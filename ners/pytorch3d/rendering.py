"""
Helper functions for Pytorch3d Renderers.
"""
import numpy as np
import pytorch3d

from ners.pytorch3d.rasterizer import MeshRasterizer


def ambient_light(device="cpu"):
    lights = pytorch3d.renderer.DirectionalLights(
        device=device,
        ambient_color=((1.0, 1.0, 1.0),),
        diffuse_color=((0.0, 0.0, 0.0),),
        specular_color=((0.0, 0.0, 0.0),),
    )
    return lights


def get_renderers(device=None, cameras=None, img_size=256):
    """
    Returns Pytorch3D MeshRenderers.

    Args:
        device (torch.device): Device to place renderers on.
        cameras: Default pytorch3d cameras for renderers.
        img_size (int): Rendered image size.

    Returns:
        textured_img_renderer
        masked_img_renderer
    """
    blend_params = pytorch3d.renderer.BlendParams(sigma=1e-4, gamma=1e-4)
    # TODO(@jason): As of v0.4.0, there's a bug in Pytorch3d with the default
    # perspective_correct. Revisit this for future versions.
    img_raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=img_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
        perspective_correct=False,
    )
    mask_raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=img_size,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
        faces_per_pixel=50,
        bin_size=0,
        perspective_correct=False,
    )
    img_rasterizer = MeshRasterizer(
        cameras=cameras, raster_settings=img_raster_settings
    )
    mask_rasterizer = MeshRasterizer(
        cameras=cameras, raster_settings=mask_raster_settings
    )
    textured_img_shader = pytorch3d.renderer.SoftPhongShader(
        cameras=cameras,
        device=device,
        blend_params=blend_params,
        lights=ambient_light(device=device),
    )
    mask_shader = pytorch3d.renderer.SoftSilhouetteShader(blend_params=blend_params)
    textured_img_renderer = pytorch3d.renderer.MeshRenderer(
        img_rasterizer, textured_img_shader
    )
    mask_renderer = pytorch3d.renderer.MeshRenderer(mask_rasterizer, mask_shader)
    return textured_img_renderer, mask_renderer
