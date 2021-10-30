"""
Class for Perspective cameras with image center and crop scale support.
"""
import ipdb
import numpy as np
import torch
from pytorch3d.renderer import FoVPerspectiveCameras

# Default values for rotation and translation matrices.
_R = torch.eye(3)[None]  # (1, 3, 3)
_T = torch.zeros(1, 3)  # (1, 3)


def compute_crop_parameters(image_size, bbox, image_center=None):
    """
    Computes the principal point and scaling factor for focal length given a square
    bounding box crop of an image.

    These intrinsic parameters are used to preserve the original principal point even
    after cropping the image.

    Args:
        image_size (int or array): Size of image, either length of longer dimension or
            (N, H, C).
        bbox: Square bounding box in xyxy (4,).
        image_center: Center of projection/principal point (2,).

    Returns:
        principal_point: Coordinates in NDC using Pytorch3D convention with (1, 1)
            as upper-left (2,).
        crop_scale (float): Scaling factor for focal length.
    """
    bbox = np.array(bbox)
    b = max(bbox[2:] - bbox[:2])
    if isinstance(image_size, int):
        h = w = image_size
    else:
        h, w, *c = image_size
        image_size = max(image_size)
    if image_center is None:
        image_center = np.array([w / 2, h / 2])
    bbox_center = (bbox[:2] + bbox[2:]) / 2
    crop_scale = b / image_size
    principal_point = 2 * (bbox_center - image_center) / b
    return principal_point, crop_scale


class PerspectiveCameras(FoVPerspectiveCameras):
    def __init__(
        self,
        znear=1.0,
        zfar=100.0,
        aspect_ratio=1.0,
        fov=60.0,
        degrees: bool = True,
        R=_R,
        T=_T,
        K=None,
        image_center=((0, 0),),
        crop_scale=(1,),
        device="cpu",
    ):
        """
        Perspective Cameras with support for cropped images and appropriate scaling of
        the focal length.

        Args:
            znear: near clipping plane of the view frustrum.
            zfar: far clipping plane of the view frustrum.
            aspect_ratio: aspect ratio of the image pixels.
                1.0 indicates square pixels.
            fov: field of view angle of the camera.
            degrees: bool, set to True if fov is specified in degrees.
            R: Rotation matrix of shape (N, 3, 3)
            T: Translation matrix of shape (N, 3)
            K: (optional) A calibration matrix of shape (N, 4, 4)
                If provided, don't need znear, zfar, fov, aspect_ratio, degrees
            image_center: Principal points of camera projection specified in NDC (N, 2).
            crop_scale: Scaling factor for the focal length of a crop.
            device: torch.device or string
        """
        super().__init__(
            device=device,
            znear=znear,
            zfar=zfar,
            aspect_ratio=aspect_ratio,
            fov=fov,
            R=R,
            T=T,
            K=K,
            degrees=degrees,
        )
        if not torch.is_tensor(image_center):
            image_center = torch.tensor(image_center, device=self.device)
        if len(image_center) != self._N:
            image_center = image_center.repeat(self._N, 1)
        self.image_center = image_center

        if not torch.is_tensor(crop_scale):
            crop_scale = torch.tensor(crop_scale, device=self.device)
        if len(crop_scale) != self._N:
            crop_scale = crop_scale.repeat(self._N)
        self.crop_scale = crop_scale

    def compute_projection_matrix(self, znear, zfar, fov, aspect_ratio, degrees):
        """
        Compute the calibration matrix K of shape (N, 4, 4)

        Args:
            znear: near clipping plane of the view frustrum.
            zfar: far clipping plane of the view frustrum.
            fov: field of view angle of the camera.
            aspect_ratio: aspect ratio of the image pixels.
                1.0 indicates square pixels.
            degrees: bool, set to True if fov is specified in degrees.

        Returns:
            torch.floatTensor of the calibration matrix with shape (N, 4, 4)
        """
        K = torch.zeros((self._N, 4, 4), device=self.device, dtype=torch.float32)
        ones = torch.ones((self._N), dtype=torch.float32, device=self.device)
        if degrees:
            fov = (np.pi / 180) * fov

        if not torch.is_tensor(fov):
            fov = torch.tensor(fov, device=self.device)
        tanHalfFov = torch.tan((fov / 2)) * self.crop_scale
        principal_point = self.image_center.clone() * (znear * tanHalfFov).unsqueeze(1)
        max_y = tanHalfFov * znear
        min_y = -max_y
        max_x = max_y * aspect_ratio
        min_x = -max_x
        max_y += principal_point[:, 1]
        min_y += principal_point[:, 1]
        max_x += principal_point[:, 0]
        min_x += principal_point[:, 0]

        # NOTE: In OpenGL the projection matrix changes the handedness of the
        # coordinate frame. i.e the NDC space postive z direction is the
        # camera space negative z direction. This is because the sign of the z
        # in the projection matrix is set to -1.0.
        # In pytorch3d we maintain a right handed coordinate system throughout
        # so the so the z sign is 1.0.
        z_sign = 1.0

        K[:, 0, 0] = 2.0 * znear / (max_x - min_x)
        K[:, 1, 1] = 2.0 * znear / (max_y - min_y)
        K[:, 0, 2] = (max_x + min_x) / (max_x - min_x)
        K[:, 1, 2] = (max_y + min_y) / (max_y - min_y)
        K[:, 3, 2] = z_sign * ones

        # NOTE: This maps the z coordinate from [0, 1] where z = 0 if the point
        # is at the near clipping plane and z = 1 when the point is at the far
        # clipping plane.
        K[:, 2, 2] = z_sign * zfar / (zfar - znear)
        K[:, 2, 3] = -(zfar * znear) / (zfar - znear)

        return K

    def __getitem__(self, index):
        attribute_names = [
            "znear",
            "zfar",
            "aspect_ratio",
            "fov",
            "R",
            "T",
            "K",
            "crop_scale",
            "image_center",
        ]
        attributes = {k: getattr(self, k) for k in attribute_names}
        if isinstance(index, int):
            for k, v in attributes.items():
                if v is None:
                    attributes[k] = v
                else:
                    attributes[k] = v[index : index + 1]
        elif isinstance(index, slice):
            for k, v in attributes.items():
                if v is None:
                    attributes[k] = v
                else:
                    attributes[k] = v[index]
        elif isinstance(index, list):
            for k, v in attributes.items():
                if v is None:
                    attributes[k] = v
                else:
                    attributes[k] = torch.stack([v[i] for i in index], 0)
        elif isinstance(index, torch.Tensor):
            if index.dim() != 1 or index.dtype.is_floating_point:
                raise IndexError(index)
            # NOTE consider converting index to cpu for efficiency
            if index.dtype == torch.bool:
                # advanced indexing on a single dimension
                index = index.nonzero()
                index = index.squeeze(1) if index.numel() > 0 else index
                index = index.tolist()
            for k, v in attributes.items():
                if v is None:
                    attributes[k] = v
                else:
                    attributes[k] = torch.stack([v[i] for i in index], 0)
        else:
            raise IndexError(index)

        return self.__class__(device=self.device, degrees=self.degrees, **attributes)

    def detach(self):
        attribute_names = [
            "znear",
            "zfar",
            "aspect_ratio",
            "fov",
            "R",
            "T",
            "K",
            "crop_scale",
            "image_center",
        ]
        attributes = {}
        for k in attribute_names:
            v = getattr(self, k)
            if v is not None:
                attributes[k] = v.detach()
        return self.__class__(device=self.device, degrees=self.degrees, **attributes)

    def get_focal_length(self, fov=None):
        if fov is None:
            fov = self.fov
        if self.degrees:
            fov = (np.pi / 180) * fov
        # If fov is negative, f should still be positive.
        return torch.abs(1 / torch.tan(fov / 2))

    def extra_repr(self):
        return f"N={self._N}"
