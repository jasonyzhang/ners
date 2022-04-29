"""
Implements class definitions for Neural Reflectance Surfaces (NeRS).
"""
import os.path as osp

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pytorch3d
import torch
import torch.nn as nn
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.renderer import RasterizationSettings, TexturesVertex
from pytorch3d.structures import Meshes
from torch.nn.parallel import DataParallel
from tqdm.auto import tqdm

import ners.models as models
import ners.utils.geometry as geom_util
from ners.pytorch3d import (
    FoVPerspectiveCameras,
    PerspectiveCameras,
    TexturesImplicit,
    get_renderers,
)
from ners.utils import PerceptualLoss, antialias, visualize_masks

DEFAULTS = {
    "images": np.zeros((1, 256, 256, 3), dtype=np.float32),
    "masks": np.zeros((1, 256, 256), dtype=np.float32),
    "masks_dt": np.zeros((1, 256, 256), dtype=np.float32),
    "initial_poses": np.zeros((1, 3, 3), dtype=np.float32),
    "image_center": np.zeros((1, 2), dtype=np.float32),
    "crop_scale": np.zeros((1,)),
}


class Ners(object):
    def __init__(
        self,
        images=DEFAULTS["images"],
        masks=DEFAULTS["masks"],
        masks_dt=DEFAULTS["masks_dt"],
        initial_poses=DEFAULTS["initial_poses"],
        image_center=DEFAULTS["image_center"],
        crop_scale=DEFAULTS["crop_scale"],
        f_template=None,
        fov=60.0,
        jitter_uv=True,
        gpu_ids=None,
        gpu_id_illumination=None,
        use_template_as_shape=True,
        symmetrize=False,
        num_layers_shape=4,
        num_layers_tex=12,
        num_layers_env=4,
        L=6,
    ) -> None:
        """
        Wrapper for training and visualizing a NeRS model.

        Args:
            images (ndarray): Cropped masked images (N, H, W, 3).
            masks (ndarray): Masks (N, H, W).
            masks_dt (ndarray): Distance transforms (N, H, W).
            initial_poses (ndarray): Initial camera poses (N, 3, 3).
            image_center (ndarray): Principal points of images (N, 2).
            crop_scale (ndarray): Crop scale (N,).
            f_template (nn.Module): Template shape network mapping from UV to XYZ.
            fov (float, optional): Initial camera field of view. Defaults to 60.0.
            jitter_uv (bool, optional): If True, jitters the UVs each iteration to
                improve coverage. Defaults to True.
            gpu_ids (list, optional): List of GPU ids to run implicit networks on.
                Defaults to all but one of the available GPUs.
            gpu_id_illumination (int, optional): GPU id to run illumination computation
                on. This is separate from gpu_ids because computing the illumination
                effects is computationally expensive, so it is recommended to do this on
                a separate GPU. Defaults to the last available GPU.
            use_template_as_shape (bool, optional): If True, uses the template network
                as the shape model. Otherwise, uses a separate shape model for
                instance-specific deformation. Defaults to True.
            symmetrize (bool, optional): If True, symmetrizes the mesh about the y-z
                plane. Defaults to False.
            num_layers_shape (int, optional): Number of layers for f_shape. Defaults
                to 4.
            num_layers_tex (int, optional): Number of layers for f_tex. The deeper, the
                higher the resolution. Defaults to 8.
            num_layers_env (int, optional): Number of lyaers for f_env. Defaults to 4.
        """
        num_gpus = torch.cuda.device_count()
        assert num_gpus > 0, "No GPUs available."
        if not gpu_ids:
            gpu_ids = list(range(num_gpus))
            if len(gpu_ids) > 1:
                gpu_ids.pop(-1)
        self.N = len(images)
        device = torch.device(f"cuda:{gpu_ids[0]}")  # Device used for hanging params.
        self.device = device
        if gpu_id_illumination:
            self.device_illumination = torch.device(f"cuda:{gpu_id_illumination}")
        else:
            self.device_illumination = torch.device(f"cuda:{num_gpus - 1}")
        self.use_template_as_shape = use_template_as_shape

        all_images = np.stack(images)
        all_images_tensor = torch.from_numpy(all_images.transpose(0, 3, 1, 2))
        all_images_tensor = all_images_tensor.float().to(device).contiguous()
        self.all_images = all_images
        self.all_images_tensor = all_images_tensor
        self.masks = masks
        self.target_masks = torch.from_numpy(np.stack(masks)).float().to(device)
        self.target_masks_dt = torch.from_numpy(np.stack(masks_dt)).float().to(device)
        images_masked = all_images_tensor.clone()
        images_masked[self.target_masks.unsqueeze(1).repeat(1, 3, 1, 1) == 0] = 1
        self.images_masked = images_masked
        self.jitter_uv = jitter_uv
        self.mean_texture = None
        self.finetune_camera = True

        self.renderer_textured, self.renderer_silhouette = get_renderers(device=device)

        # Prepare implicit shape, texture, and environmental illumination networks.
        if not self.use_template_as_shape:
            # To model implicit shape, we can either use a deformation network that
            # learns offsets from the template network, or directly optimize the
            # template network itself. The latter is less compute intensive but may have
            # worse performance.
            f_shape = models.DeltaUV(num_layers=num_layers_shape, hidden_size=128)
            if symmetrize:
                f_shape = models.Symmetrize(f_shape)
            self.f_shape = DataParallel(f_shape, device_ids=gpu_ids).to(device)
        else:
            if symmetrize:
                f_template = models.Symmetrize(f_template)
            self.f_shape = f_template

        f_tex = models.ImplicitTextureNet(
            num_layers=num_layers_tex, hidden_size=256, L=L
        )
        f_env = models.EnvironmentMap(
            num_layers=num_layers_env, hidden_size=128, L=L, gain=1
        )
        self.f_template = DataParallel(f_template, device_ids=gpu_ids).to(device)
        self.f_tex = DataParallel(f_tex, device_ids=gpu_ids).to(device)
        self.f_env = DataParallel(f_env, device_ids=gpu_ids).to(device)
        self.specularity = torch.tensor(1.0, device=device, requires_grad=True).float()
        self.shininess = torch.tensor(15.0, device=device, requires_grad=True).float()

        # Prepare initial camera poses.
        self.fov = torch.tensor(fov, device=device).float()
        self.cameras_current = PerspectiveCameras(
            fov=self.fov,
            R=np.stack(initial_poses),
            T=[[0, 0, 2]] * self.N,
            image_center=image_center,
            crop_scale=crop_scale,
            device=device,
        )

        # Prepare initial mesh.
        self.sphere_vs, self.sphere_fs = geom_util.create_sphere(4, device=device)
        pred_vs, sv = self.get_pred_verts(self.sphere_vs, predict_deformation=False)
        texture_rainbow = TexturesVertex((sv.unsqueeze(0) + 1) / 2)
        self.meshes_current = Meshes(
            [pred_vs], [self.sphere_fs], textures=texture_rainbow
        )

        # Compute edges for chamfer loss.
        pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        masks_pooled = -pool(-self.target_masks)  # Min pool
        edges = self.target_masks - masks_pooled
        # [(E, 2)]
        self.edge_pixels = [torch.stack(torch.where(e > 0.5), dim=1) for e in edges]
        self.edge_counts = [e.shape[0] for e in self.edge_pixels]

        # Prepare perceptual loss.
        lp = PerceptualLoss(net="vgg")
        self.loss_perceptual = DataParallel(lp, device_ids=gpu_ids).to(device)

        self.loss_weights = {
            "chamfer": 0.02,
            "silhouette": 2.0,
            "distance_transform": 20,
            "normal": 0.1,
            "laplacian": 0.1,
            "offscreen": 1000,
            "texture": 0.5,
            "texture_mean": 0.15,
        }

    def get_pred_verts(self, sphere_vs=None, predict_deformation=True, jitter_uv=None):
        """
        Outputs predicted vertex locations from a UV sphere.

        Args:
            sphere_vs (torch.Tensor): UV coordinates on sphere (V, 3).
            predict_deformation (bool): If True, also predict using the current
                deformation model. Otherwise, just outputs shape based on category
                shape.
            jitter_uv (bool): If True, jitters the uv values using a random rotation.

        Returns:
            torch.Tensor: Predicted vertices corresponding to the sphere UVs (V, 3).
            torch.Tensor: Sphere UVs to predict vertices (V, 3).
        """
        sv = self.sphere_vs if sphere_vs is None else sphere_vs
        if jitter_uv is None:
            jitter_uv = self.jitter_uv
        if jitter_uv:
            sv = sv @ geom_util.random_rotation(sv.device)
        if self.use_template_as_shape:
            pred_vs = self.f_template(sv)
        else:
            with torch.no_grad():
                pred_vs = self.f_template(sv)
            if predict_deformation:
                pred_vs = pred_vs + self.f_shape(sv)
        return pred_vs, sv

    def compute_chamfer_loss(self, meshes, cameras, num_samples=300):
        """
        Computes the 2D chamfer loss between the edges of the mask and edges of the
        rendered silhouette.

        Args:
            meshes: Meshes to compute chamfer loss on.
            cameras: Cameras to render meshes with.
            num_samples (int): Number of samples to use for chamfer loss.

        Returns:
            torch.Tensor: Chamfer loss.
        """
        edge_pixels_sampled = []  # [(num_samples, 2)] * N
        for counts, pixels in zip(self.edge_counts, self.edge_pixels):
            # Sampling with replacement because multinomial doesn't play well with Long.
            sample = pixels[torch.randint(counts, size=(num_samples,))]
            sample = sample.roll(1, dims=1)  # row, column -> x, y
            edge_pixels_sampled.append(sample)
        # (N, num_samples, 1, 2)
        edge_pixels_sampled = torch.stack(edge_pixels_sampled).unsqueeze(2)
        # Projected Points:
        proj = cameras.transform_points_screen(  # (N, V, 2)
            meshes.verts_padded(),
            image_size=[(256, 256)] * self.N,
        )
        proj = proj[..., :2]  # Drop the depth buffer
        proj = proj.unsqueeze(1)  # (N, 1, V, 2)
        dists = (proj - edge_pixels_sampled).norm(dim=-1)
        return torch.mean(dists.min(dim=-1).values)

    def compute_offscreen_loss(self, cameras, verts):
        """
        Computes a loss for how much vertices go offscreen, ie outside of [-1, 1] NDC
        coordinates. This helps avoid the degenerate solution of moving objects
        offscreen to minimize the distance transform/silhouette loss.

        Args:
            cameras: Cameras to render vertices with.
            verts: Mesh vertices to compute loss on (N, V, 3).

        Returns:
            torch.Tensor: Offscreen loss.
        """
        points = cameras.transform_points(verts)[..., :2]
        return torch.sum(torch.relu(points - 1) + torch.relu(-1 - points))

    def visualize_input_views(self, filename=None, title="", meshes=None, cameras=None):
        """
        Generates visualizations for each of the training views.

        Args:
            filename (str): If not None, saves the image to this file.
            title (str): Title for the figure.
            meshes (pytorch3d.structures.Meshes): Meshes to visualize. If None, uses
                currently predicted mesh.
            cameras: Cameras to visualize. If None, uses currently predicted cameras.
        """
        if meshes is None:
            meshes = self.meshes_current.extend(self.N)
        if cameras is None:
            cameras = self.cameras_current
        with torch.no_grad():
            rend = self.renderer_textured(meshes, cameras=cameras)
            rend_images = rend.detach().cpu().numpy()[..., :3].clip(0, 1)
            rend_sil = self.renderer_silhouette(meshes, cameras=cameras)
            rend_sil = rend_sil.detach().cpu().numpy()[..., 3].clip(0, 1)
        n = len(rend_images)
        fig, axs = plt.subplots(n, 4, figsize=(8, n * 2), dpi=100)
        axs = axs.flatten()
        for i in range(n):
            image = self.all_images[i]
            axs[4 * i].imshow(image)
            axs[4 * i].set_title(f"Image {i + 1}")
            rend_image = rend_images[i]
            mask = rend_sil[i] > 0.5
            vis_im = 1 - (1 - image.copy()) * 0.7
            vis_im[mask] = rend_image[mask].clip(0, 1)

            axs[4 * i + 1].imshow(vis_im)
            axs[4 * i + 2].imshow(rend_image.clip(0, 1))
            mask_pred = self.masks[i] > 0.5
            axs[4 * i + 3].imshow(visualize_masks(mask, mask_pred))
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        if title:
            plt.suptitle(title, y=0.93)
        if filename:
            plt.savefig(fname=filename, format="jpg", bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def optimize_camera(self, num_iterations=500, finetune_rotations=True, pbar=True):
        cameras = self.cameras_current
        T = cameras.T.detach()
        R = geom_util.matrix_to_rot6d(cameras.R.detach())
        T.requires_grad = True
        R.requires_grad = False
        meshes = self.meshes_current.extend(self.N).detach()
        optim = torch.optim.Adam([T], lr=0.01)
        loop = tqdm(range(num_iterations)) if pbar else range(num_iterations)
        for i in loop:
            optim.zero_grad()
            cameras.T = T
            cameras.R = geom_util.rot6d_to_matrix(R)
            rend = self.renderer_silhouette(meshes, cameras=cameras)[..., 3]
            losses = {"silhouette": torch.mean((rend - self.target_masks) ** 2)}
            losses["chamfer"] = self.compute_chamfer_loss(
                meshes=meshes, cameras=cameras
            )
            losses["distance_transform"] = torch.mean(rend * self.target_masks_dt)
            losses["offscreen"] = self.compute_offscreen_loss(
                cameras, meshes.verts_padded().detach()
            )
            loss = sum(losses[k] * self.loss_weights[k] for k in losses.keys())
            if pbar:
                loop.set_description(f"Camera: {loss.item():.4f}")
            loss.backward()
            optim.step()
            if i == num_iterations // 2 and finetune_rotations:
                R.requires_grad = True
                optim = torch.optim.Adam([T, R], lr=1e-3)
        R.requires_grad = False
        T.requires_grad = False
        self.cameras_current = cameras.detach()
        return T, R

    def optimize_shape(self, num_iterations=500, pbar=True, lr=1e-4):
        R = geom_util.matrix_to_rot6d(self.cameras_current.R.detach())
        T = self.cameras_current.T.detach()
        fov = self.fov.clone().detach()
        if self.finetune_camera:
            params = [R, T, fov]
            for param in params:
                param.requires_grad = True
            parameters = [{"params": params, "lr": lr * 10}]
        else:
            parameters = []
        parameters.append({"params": self.f_shape.parameters(), "lr": lr})

        optim = torch.optim.Adam(parameters)
        loop = tqdm(range(num_iterations)) if pbar else range(num_iterations)
        for _ in loop:
            optim.zero_grad()
            pred_vs, sv = self.get_pred_verts(predict_deformation=True)
            meshes = Meshes([pred_vs], [self.sphere_fs]).extend(self.N)
            cameras = PerspectiveCameras(
                fov=fov,
                R=geom_util.rot6d_to_matrix(R),
                T=T,
                image_center=self.cameras_current.image_center,
                crop_scale=self.cameras_current.crop_scale,
                device=self.device,
            )
            rend = self.renderer_silhouette(meshes, cameras=cameras)[..., 3]
            loss_dict = {}
            loss_dict["silhouette"] = torch.mean((rend - self.target_masks) ** 2)
            loss_dict["distance_transform"] = torch.mean(rend * self.target_masks_dt)
            loss_dict["offscreen"] = self.compute_offscreen_loss(
                cameras, pred_vs.unsqueeze(0).detach()
            )
            loss_dict["laplacian"] = mesh_laplacian_smoothing(meshes)
            loss_dict["normal"] = mesh_normal_consistency(meshes)
            loss_dict["chamfer"] = self.compute_chamfer_loss(meshes, cameras)
            loss = sum(loss_dict[k] * self.loss_weights[k] for k in loss_dict.keys())
            if pbar:
                loop.set_description(f"Shape: {loss.item():.4f}")
            loss.backward()
            optim.step()
        self.fov = fov.clone().detach()
        self.cameras_current = cameras.detach()
        self.meshes_current = Meshes(
            verts=[pred_vs],
            faces=[self.sphere_fs],
            textures=TexturesVertex((sv.unsqueeze(0) + 1) / 2),
        ).detach()

    def optimize_texture(self, num_iterations=3000, lr=1e-4, pbar=True):
        R = geom_util.matrix_to_rot6d(self.cameras_current.R.detach())
        T = self.cameras_current.T.detach()
        fov = self.fov.clone().detach()
        if self.finetune_camera:
            params = [R, T, fov]
            for param in params:
                param.requires_grad = True
            parameters = [{"params": params, "lr": lr * 10}]
        else:
            parameters = []
        parameters.append({"params": self.f_shape.parameters(), "lr": lr})
        parameters.append({"params": self.f_tex.parameters(), "lr": lr})

        optim = torch.optim.Adam(parameters)
        loop = tqdm(range(num_iterations)) if pbar else range(num_iterations)
        for _ in loop:
            optim.zero_grad()
            cameras = PerspectiveCameras(
                fov=fov,
                R=geom_util.rot6d_to_matrix(R),
                T=T,
                image_center=self.cameras_current.image_center,
                crop_scale=self.cameras_current.crop_scale,
                device=self.device,
            )
            pred_vs, sv = self.get_pred_verts(predict_deformation=True)
            meshes = Meshes([pred_vs], [self.sphere_fs]).extend(self.N)
            pred_textures = TexturesImplicit(
                texture_predictor=self.f_tex,
                faces=self.sphere_fs,
                verts_sphere_coords=sv,
                verts_deformed_coords=pred_vs,
                predict_radiance=False,
            )
            meshes.textures = pred_textures
            rend_sil = self.renderer_silhouette(meshes, cameras=cameras)[..., 3]
            rend_text = self.renderer_textured(meshes, cameras=cameras)[..., :3]
            rend_text = rend_text.clamp(0, 1)
            loss_dict = {}
            loss_dict["silhouette"] = torch.mean((rend_sil - self.target_masks) ** 2)
            loss_dict["distance_transform"] = torch.mean(
                rend_sil * self.target_masks_dt
            )
            loss_dict["offscreen"] = self.compute_offscreen_loss(
                cameras, pred_vs.unsqueeze(0).detach()
            )
            loss_dict["laplacian"] = mesh_laplacian_smoothing(meshes)
            loss_dict["normal"] = mesh_normal_consistency(meshes)
            loss_dict["texture"] = self.loss_perceptual(
                self.images_masked, rend_text.permute(0, 3, 1, 2)
            ).mean()
            loss_dict["chamfer"] = self.compute_chamfer_loss(meshes, cameras)
            loss = sum(loss_dict[k] * self.loss_weights[k] for k in loss_dict)
            if pbar:
                loop.set_description(f"Texture: {loss.item():.4f}")
            loss.backward()
            optim.step()
        self.fov = fov.clone().detach()
        self.cameras_current = cameras.detach()
        self.meshes_current = meshes[0]
        self.meshes_current.textures = pred_textures

    def optimize_radiance(self, num_iterations=500, pbar=True, lr=1e-4):
        R = geom_util.matrix_to_rot6d(self.cameras_current.R.detach())
        T = self.cameras_current.T.detach()
        fov = self.fov.clone().detach()
        params = [self.specularity, self.shininess]
        if self.finetune_camera:
            params.extend([R, T, fov])
        for param in params:
            param.requires_grad = True
        parameters = [{"params": params, "lr": lr * 10}]
        parameters.append({"params": self.f_shape.parameters(), "lr": lr})
        parameters.append({"params": self.f_tex.parameters(), "lr": lr})
        parameters.append({"params": self.f_env.parameters(), "lr": lr})

        loop = tqdm(range(num_iterations)) if pbar else range(num_iterations)
        optim = torch.optim.Adam(parameters)
        for _ in loop:
            optim.zero_grad()
            cameras = PerspectiveCameras(
                fov=fov,
                R=geom_util.rot6d_to_matrix(R),
                T=T,
                image_center=self.cameras_current.image_center,
                crop_scale=self.cameras_current.crop_scale,
                device=self.device,
            )
            pred_vs, sv = self.get_pred_verts(predict_deformation=True)
            meshes = Meshes([pred_vs], [self.sphere_fs]).extend(self.N)
            pred_textures = TexturesImplicit(
                texture_predictor=self.f_tex,
                faces=self.sphere_fs,
                verts_sphere_coords=sv,
                verts_deformed_coords=pred_vs,
                verts_normals=meshes.verts_normals_padded()[0],
                predict_radiance=True,
                env_map=self.f_env,
                specularity=self.specularity,
                shininess=self.shininess,
                jitter_env_map_rays=True,
            )
            pred_textures.custom_device = self.device_illumination
            mean_texture = self.f_tex(sv).mean(dim=0).clone().detach()
            meshes.textures = pred_textures
            rend_sil = self.renderer_silhouette(meshes, cameras=cameras)[..., 3]
            rend_text = self.renderer_textured(meshes, cameras=cameras)[..., :3]
            rend_text = rend_text.clamp(0, 1)
            pred_textures.set_texture_color(mean_texture)
            rend_text_mean = self.renderer_textured(meshes, cameras=cameras)
            rend_text_mean = rend_text_mean[..., :3].clamp(0, 1)
            pred_textures.texture_color = None

            loss_dict = {}
            loss_dict["silhouette"] = torch.mean((rend_sil - self.target_masks) ** 2)
            loss_dict["distance_transform"] = torch.mean(
                rend_sil * self.target_masks_dt
            )
            loss_dict["offscreen"] = self.compute_offscreen_loss(
                cameras, pred_vs.unsqueeze(0).detach()
            )
            loss_dict["laplacian"] = mesh_laplacian_smoothing(meshes)
            loss_dict["normal"] = mesh_normal_consistency(meshes)
            loss_dict["texture"] = self.loss_perceptual(
                self.images_masked, rend_text.permute(0, 3, 1, 2)
            ).mean()
            loss_dict["texture_mean"] = self.loss_perceptual(
                self.images_masked, rend_text_mean.permute(0, 3, 1, 2)
            ).mean()
            loss_dict["chamfer"] = self.compute_chamfer_loss(meshes, cameras)

            loss = sum(loss_dict[k] * self.loss_weights[k] for k in loss_dict)
            if pbar:
                loop.set_description(f"Radiance: {loss.item():.4f}")
            loss.backward()
            optim.step()
        self.fov = fov.clone().detach()
        self.cameras_current = cameras.detach()
        self.meshes_current = meshes[0]
        self.meshes_current.texture = pred_textures
        self.mean_texture = mean_texture.clone().detach()

    def save_parameters(self, filename):
        state_dicts = {
            "cameras": self.cameras_current,
            "fov": self.fov,
            "shininess": self.shininess,
            "specularity": self.specularity,
            "f_shape": self.f_shape.state_dict(),
            "f_tex": self.f_tex.state_dict(),
            "f_env": self.f_env.state_dict(),
            "f_template": self.f_template.state_dict(),
        }
        torch.save(state_dicts, filename)

    def load_parameters(self, filename, device=None):
        state_dicts = torch.load(filename)
        self.cameras_current = state_dicts["cameras"]
        self.fov = state_dicts["fov"]
        self.shininess = state_dicts["shininess"]
        self.specularity = state_dicts["specularity"]

        self.f_shape.load_state_dict(state_dicts["f_shape"])
        self.f_tex.load_state_dict(state_dicts["f_tex"])
        self.f_env.load_state_dict(state_dicts["f_env"])
        self.f_template.load_state_dict(state_dicts["f_template"])
        pred_vs, sv = self.get_pred_verts(predict_deformation=True)
        meshes = Meshes([pred_vs], [self.sphere_fs])
        pred_textures = TexturesImplicit(
            texture_predictor=self.f_tex,
            faces=self.sphere_fs,
            verts_sphere_coords=sv,
            verts_deformed_coords=pred_vs,
            verts_normals=meshes.verts_normals_padded()[0],
            predict_radiance=True,
            env_map=self.f_env,
            specularity=self.specularity,
            shininess=self.shininess,
            jitter_env_map_rays=False,
        )
        meshes.textures = pred_textures
        self.meshes_current = meshes
        if device is not None:
            self.f_shape = self.f_shape.to(device)
            self.f_tex = self.f_tex.to(device)
            self.f_env = self.f_env.to(device)
            self.f_template = self.f_template.to(device)
            pred_textures.custom_device = device

    def make_video(
        self,
        fname,
        num_frames=360,
        fps=30,
        image_size=512,
        use_antialiasing=True,
        extension="mp4",
        visuals=("nn", "full", "lighting"),
        pbar=True,
        elev=15,
        dist=2.2,
    ):
        """
        Generates a video of the results

        Args:
            fname (str): Filename to write video.
            num_frames (int, optional): Number of frames. Defaults to 360.
            fps (int, optional): [description]. Defaults to 30.
            image_size (int, optional): Size of image. Defaults to 512.
            use_antialiasing (bool, optional): If True, performs antialiasing. Defaults
                to True.
            extension (str, optional): Defaults to "mp4".
        """
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
        writer = imageio.get_writer(f"{fname}.{extension}", mode="I", fps=fps)
        with torch.no_grad():
            azim = torch.linspace(0, 360, num_frames)
            R, T = pytorch3d.renderer.cameras.look_at_view_transform(
                dist=dist,
                elev=elev,
                azim=azim,
                device=self.device,
            )
            new_R = R
            loop = tqdm(range(num_frames), "Video") if pbar else range(num_frames)
            for i in loop:
                perspective_camera = FoVPerspectiveCameras(
                    device=self.device, R=new_R[i, None], T=T[i, None], fov=self.fov
                )
                images = []
                for visual_name in visuals:
                    textures_og = self.meshes_current.textures
                    textures = TexturesImplicit(
                        texture_predictor=textures_og.texture_predictor,
                        faces=textures_og.faces,
                        verts_sphere_coords=textures_og.verts_sphere_coords,
                        verts_deformed_coords=textures_og.verts_deformed_coords,
                        verts_normals=textures_og.verts_normals,
                        predict_radiance=True,
                        env_map=textures_og.env_map,
                        specularity=textures_og.specularity,
                        shininess=textures_og.shininess,
                        jitter_env_map_rays=False,
                    )
                    if visual_name == "nn":
                        R_dist = torch.norm(
                            self.cameras_current.R - new_R[i, None], dim=(1, 2)
                        )
                        ind = R_dist.cpu().numpy().argmin()
                        images.append(
                            cv2.resize(self.all_images[ind], (image_size, image_size))
                        )
                        continue
                    elif visual_name == "full":
                        # Default
                        pass
                    elif visual_name == "albedo":
                        textures.predict_radiance = False
                    elif visual_name == "lighting":
                        mean_color = self.f_tex(self.sphere_vs).mean(dim=0)
                        textures.set_texture_color(mean_color)
                    else:
                        raise Exception(
                            f"Visualization format: {visual_name} not recognized."
                        )
                    self.meshes_current.textures = textures
                    rend = self.renderer_textured(
                        self.meshes_current,
                        cameras=perspective_camera,
                        raster_settings=raster_settings,
                    )
                    images.append(
                        np.clip(rend.detach().cpu().numpy()[0, ..., :3], 0, 1)
                    )
                if use_antialiasing:
                    images = [antialias(image) for image in images]
                combined = (np.hstack(images) * 255).astype(np.uint8)
                writer.append_data(combined)
        writer.close()

    def save_obj(
        self,
        fname,
        sphere_level=4,
        margin=0.25,
        height=2048,
        width=2048,
    ):
        """
        Exports NeRS model to a textured mesh.

        Texture is discretized using a UV image.

        Args:
            fname (str): Filename
            sphere_level (int, optional): Level for isosphere. Defaults to 20.
            height (int): Height of uv image. Defaults to 2048.
            width (int): Width of uv image. Defaults to 2048.
        """
        device = self.device
        if ".obj" not in fname:
            fname = fname + ".obj"
        basename = osp.basename(fname.replace(".obj", ""))
        sv, sf = geom_util.create_sphere(sphere_level)
        sv = sv.to(device)
        sf = sf.to(device).long()
        with torch.no_grad():
            pred_verts, sv = self.get_pred_verts(sv, jitter_uv=False)

        theta, phi = geom_util.cartesian_to_spherical(sv[:, 0], sv[:, 1], sv[:, 2])
        # Find faces with vertices on opposite sides of the seam
        seam_mask = torch.logical_and(
            torch.any(phi[sf] > np.pi / 2, 1),  # Face has points on right edge.
            torch.any(phi[sf] < -np.pi / 2, 1),  # Face has points on left edge.
        )
        seam_indices = torch.where(seam_mask)[0].cpu()
        verts = pred_verts.cpu()
        faces_verts = sf.cpu()
        faces_textures = faces_verts.clone()
        phi = phi.cpu()
        theta = theta.cpu()
        texture_vertex_map = {}  # map from vert indices to texture vert indices.
        for ind in seam_indices:
            # Make a new set of texture vertices.
            verts_indices = faces_verts[ind]
            texture_verts_indices = []
            for v_i in verts_indices:
                v_i = v_i.tolist()
                if v_i in texture_vertex_map:
                    # Already processed this vertex.
                    texture_verts_indices.append(texture_vertex_map[v_i])
                    continue
                elif phi[v_i] > 0:
                    # Vertex is already on the right side.
                    texture_vertex_map[v_i] = v_i
                else:
                    # Need to construct a new set of texture verts.
                    texture_vertex_map[v_i] = len(phi)
                    new_phi = phi[v_i] + 2 * np.pi
                    phi = torch.cat((phi, torch.tensor([new_phi])))
                    theta = torch.cat((theta, torch.tensor([theta[v_i]])))
                texture_verts_indices.append(texture_vertex_map[v_i])
            faces_textures[ind] = torch.tensor(texture_verts_indices)
        with torch.no_grad():
            uv_map = self.f_tex.module.unwrap_uv_map(height, width, margin=margin)
        theta = 1 - theta / np.pi
        phi = (phi + np.pi) / ((2 + margin) * np.pi)

        with open(fname, "w") as f:
            f.write(f"mtllib {basename}.mtl\n")
            f.write(f"usemtl {basename}\n")
            for vert in verts:
                f.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
            for p, t in zip(phi, theta):
                f.write(f"vt {p} {t}\n")
            for fv, ft in zip(faces_verts, faces_textures):
                fv = fv + 1
                ft = ft + 1
                f.write(f"f {fv[0]}/{ft[0]} {fv[1]}/{ft[1]} {fv[2]}/{ft[2]}\n")
        with open(fname.replace(".obj", ".mtl"), "w") as f:
            f.write(f"newmtl {basename}\n")
            f.write(f"map_Kd {basename}.png\n")
        plt.imsave(fname.replace(".obj", ".png"), uv_map.cpu().numpy())
