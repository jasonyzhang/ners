import numpy as np
import torch
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh.textures import TexturesBase

import ners.utils.geometry as geom_util


def dot_product(a, b):
    return torch.sum(a * b, dim=-1, keepdim=True)


def normalize(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def compute_lighting_specular(
    env_map,
    vertex_normals,
    view_direction,
    env_map_rays,
    specularity,
    shininess,
    device=None,
    max_batch_size=None,
):
    """
    env_map (nn.Module): Environment map mapping (dx, dy, dz) to light intensity.
    vertex_normals (N1,3): Normalized normal vector.
    view_direction (N1,3): Normalized vector from vertex/face to camera.
    env_map_rays (N2,3): Discrete sampling of ray directions to environment map.
    specularity ((N1,) or (1,)): Weighting factor of specularity
    shininess ((N1,) or (1,)):
    """
    if device:
        temp_device = vertex_normals.device
        vertex_normals = vertex_normals.to(device)
        view_direction = view_direction.to(device)
        env_map_rays = env_map_rays.to(device)
        specularity = specularity.to(device)
        shininess = shininess.to(device)
    num_rays = env_map_rays.shape[0]
    # shininess = shininess.reshape(-1, 1, 1)  # (N1, 1, 3)
    # specularity = specularity.reshape(-1, 1, 1)  # (N1, 1, 3)
    shininess = shininess.unsqueeze(1)
    specularity = specularity.unsqueeze(1)
    N = vertex_normals.unsqueeze(1)  # (N1, 1, 3)
    V = view_direction.unsqueeze(1)  # (N1, 1, 3)

    L = env_map_rays.unsqueeze(0)  # (1, N2, 3)
    E = env_map(env_map_rays).unsqueeze(0).to(L.device)  # (1, N2, 3)

    lights = []

    if max_batch_size is None:
        max_batch_size = N.shape[0]

    for i in range(0, N.shape[0], max_batch_size):
        ind = slice(i, i + max_batch_size)
        N_ = N[ind]
        V_ = V[ind]
        shininess_ = shininess if shininess.shape[0] == 1 else shininess[ind]
        specularity_ = specularity if specularity.shape[0] == 1 else specularity[ind]
        R = 2 * dot_product(L, N_) * N_ - L  # (B, N2, 3)
        similarity = torch.relu(dot_product(R, V_))  # (B, N2, 1)
        del R
        weight = specularity_ * (shininess_ + 1) / (2 * np.pi)  # (B, 1, 1)
        light = weight * similarity ** shininess_  # (B, 1, 1)
        del similarity
        light = (light * E).sum(dim=1) / num_rays
        lights.append(light)
    light = torch.cat(lights, dim=0)

    if device:
        light = light.to(temp_device)
    if light.shape[-1] == 1:
        # Repeat last channel 3x.
        light = light.repeat((1,) * (light.ndim - 1) + (3,))
    return light


def compute_lighting_diffuse(env_map, vertex_normals, env_map_rays, device=None):
    """
    vertex_normals (N1,3): Normalized normal vector.
    env_map_rays (N2,3): Discrete sampling of ray directions to environment map.
    """
    num_rays = env_map_rays.shape[0]
    if device:
        temp_device = vertex_normals.device
        vertex_normals = vertex_normals.to(device)
        env_map_rays = env_map_rays.to(device)
    N = vertex_normals.unsqueeze(1)  # (N1, 1, 3)
    L = env_map_rays.unsqueeze(0)  # (1, N2, 3)
    E = env_map(env_map_rays).unsqueeze(0).to(L.device)
    light_diffuse = E * torch.relu(dot_product(N, L))  # (N1, N2, 3)
    light_diffuse = light_diffuse.sum(dim=1) / num_rays

    if light_diffuse.shape[-1] == 1:
        # Repeat last channel 3x.
        light_diffuse = light_diffuse.repeat((1,) * (light_diffuse.ndim - 1) + (3,))
    if device:
        light_diffuse = light_diffuse.to(temp_device)
    return light_diffuse


class TexturesImplicit(TexturesBase):
    def __init__(
        self,
        texture_predictor,
        faces,
        verts_sphere_coords,
        verts_deformed_coords=None,
        verts_normals=None,
        predict_radiance=False,
        env_map=None,
        specularity=None,
        shininess=None,
        jitter_env_map_rays=False,
    ):
        """
        Textures are represented implicitly using the weights of a neural network.
        NOTE: only ONE texture is supported across all batches. Thus, each batch should
        correspond to the same mesh.

        Args:
            texture_predictor (ImplicitTextureNet): Network that maps from uv coordinate
                to RGB color.
            faces (F, 3): Faces indices.
            verts_sphere_coords (V, 3): Coordinates from sphere (UV).
            verts_deformed_coords (V, 3): Coordinates from deformed mesh. Used to
                compute illumination.
            verts_normals (V, 3): Normals at each vertex. Used to compute illumination.
            predict_radiance (bool): If True, predicts a radiance (illumination).
            env_map (nn.Module): Environment map mapping (dx, dy, dz) to light
                intensity.
            specularity (float, Tensor): Weighting factor of specularity.
            shininess (float, Tensor): Shininess coefficient.
            jitter_env_map_rays (bool): If True, jitters rays to environment map.

        Returns:
            texels (N,H,W,K,3): RGB texels.
        """
        assert len(faces.shape) == 2, "Faces should be (F,3)."
        assert len(verts_sphere_coords.shape) == 2, "Verts should be (V,3)."
        if verts_deformed_coords is not None:
            assert len(verts_deformed_coords.shape) == 2, "Verts should be (V,3)."
        self.texture_predictor = texture_predictor
        self.device = faces.device
        self.faces = faces
        self.verts_sphere_coords = verts_sphere_coords
        self.verts_deformed_coords = verts_deformed_coords
        self.verts_normals = verts_normals
        self.predict_radiance = predict_radiance
        self.env_map = env_map
        self.specularity = specularity
        self.shininess = shininess
        self.jitter_env_map_rays = jitter_env_map_rays

        # If set, will use this color instead of querying the texture predictor.
        self.texture_color = None
        # Set for when illumination computation needs a new GPU.
        self.custom_device = None
        sphere_vs, _ = geom_util.create_sphere(3)
        self.env_map_rays = sphere_vs.to(self.device)

    def sample_textures(self, fragments, **kwargs):
        N, H, W, K = fragments.pix_to_face.shape
        faces_uv_coords = self.verts_sphere_coords[self.faces.long()]  # (F,3,3)
        # not in torch 1.7.0 yet
        # faces_verts_coords = torch.tile(faces_verts_coords, (N, 1, 1))
        faces_uv_coords = faces_uv_coords.repeat(N, 1, 1, 1).reshape(
            -1, 3, 3
        )  # (N*F,3,3)
        pixel_uvs = interpolate_face_attributes(  # (N,H,W,K,3)
            fragments.pix_to_face, fragments.bary_coords, faces_uv_coords
        )
        pixel_uvs = pixel_uvs.reshape(N * H * W * K, 3)
        # Normalize because interpolated points may not be on sphere anymore.
        pixel_uvs = pixel_uvs / (pixel_uvs.norm(dim=1, keepdim=True) + 1e-6)
        if self.texture_color is None:
            # sampled_texture = self.texture_predictor.forward_batched(pixel_uvs)
            sampled_texture = self.texture_predictor.forward(pixel_uvs)
        else:
            # Set textures to be the given color.
            sampled_texture = torch.ones(N * H * W * K, 3, device=self.device)
            sampled_texture *= self.texture_color

        if self.predict_radiance:
            if not hasattr(fragments, "cameras_centers"):
                raise AttributeError(
                    "fragments do not have attribute 'cameras_centers'. Make sure you"
                    "are using the custom MeshRasterizer."
                )

            sphere_vs = self.verts_sphere_coords
            specularity = self.specularity.reshape(-1, 1)  # (N1, 1)
            shininess = self.shininess.reshape(-1, 1)  # (N1, 1)

            env_map_rays = sphere_vs if self.env_map_rays is None else self.env_map_rays
            if self.jitter_env_map_rays:
                rot = geom_util.random_rotation(self.device)
                env_map_rays = env_map_rays @ rot

            verts_normals = normalize(self.verts_normals).repeat(N, 1, 1)  # (N,V,3)
            verts = self.verts_deformed_coords.unsqueeze(0)  # (1,V,3)
            # (N,1,3)
            camera_centers = fragments.cameras_centers[:, 0, 0, 0].unsqueeze(1)
            verts_views = normalize(camera_centers - verts)  # (N,V,3)

            lighting_diffuse = compute_lighting_diffuse(  # (N, V, 3)
                env_map=self.env_map,
                vertex_normals=verts_normals.reshape(-1, 3),
                env_map_rays=sphere_vs.reshape(-1, 3),
                device=self.custom_device,
            ).reshape(N, -1, 3)

            lighting_diffuse = lighting_diffuse[:, self.faces.long()].reshape(-1, 3, 3)
            pixel_diffuse = interpolate_face_attributes(
                fragments.pix_to_face,
                fragments.bary_coords,
                lighting_diffuse,
            ).reshape(N * H * W * K, 3)

            lighting_specular = compute_lighting_specular(  # (N, V, 3)
                env_map=self.env_map,
                vertex_normals=verts_normals.reshape(-1, 3),
                view_direction=verts_views.reshape(-1, 3),
                env_map_rays=env_map_rays,
                specularity=specularity,
                shininess=shininess,
                device=self.custom_device,
                max_batch_size=5000,
            ).reshape(N, -1, 3)

            lighting_specular = lighting_specular[:, self.faces.long()].reshape(
                -1, 3, 3
            )
            pixel_specular = interpolate_face_attributes(
                fragments.pix_to_face,
                fragments.bary_coords,
                lighting_specular,
            ).reshape(N * H * W * K, 3)
            sampled_texture = sampled_texture * pixel_diffuse
            sampled_radiance = pixel_specular
        else:
            sampled_radiance = 0
        return (sampled_texture + sampled_radiance).reshape(N, H, W, K, 3)

    def check_shapes(
        self, batch_size: int, max_num_verts: int, max_num_faces: int
    ) -> bool:
        """
        Check if the dimensions of the verts features match that of the mesh verts
        """
        return self.verts_deformed_coords.shape[0] == max_num_verts

    def extend(self, N):
        """
        Repeats batch elements N times. (Not implemented since we only support single
        texture across batch.)

        Args:
            N (int).
        """
        return self

    def __getitem__(self, item):
        return self

    def detach(self):
        """
        Note that the texture predictor is NOT detached.
        """
        tex = self.__class__(
            texture_predictor=self.texture_predictor.clone(),
            faces=self.faces.detach(),
            verts_sphere_coords=self.verts_sphere_coords.detach(),
            verts_deformed_coords=self.verts_deformed_coords.detach(),
            radiance_predictor=self.radiance_predictor.detach(),
            predict_radiance=self.predict_radiance,
        )
        return tex

    def clone(self):
        tex = self.__class__(
            texture_predictor=self.texture_predictor.clone(),
            faces=self.faces.clone(),
            verts_sphere_coords=self.verts_sphere_coords.clone(),
            verts_deformed_coords=self.verts_deformed_coords.clone(),
            radiance_predictor=self.radiance_predictor.clone(),
            predict_radiance=self.predict_radiance,
        )
        return tex

    def set_texture_color(self, rgb=(0.5, 0.5, 0.5)):
        """
        Uses a set color for the texture (rather than using the texture network). Use
        for visualizing the radiances.

        Args:
            rgb (tuple): Color. Defaults to (0.5, 0.5, 0.5).
        """
        if not isinstance(rgb, torch.Tensor):
            rgb = torch.tensor(rgb, device=self.device)
        self.texture_color = rgb

    def clear_texture_color(self):
        self.texture_color = None
