"""
Utilities for various geometric operations.
"""
import numpy as np
import pytorch3d
import torch
import torch.nn.functional as F


def random_rotation(device=None):
    quat = torch.randn(4, device=device)
    quat /= quat.norm()
    return pytorch3d.transforms.quaternion_to_matrix(quat)


def rot6d_to_matrix(rot_6d):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Reference: Zhou et al., "On the Continuity of Rotation Representations in Neural
    Networks", CVPR 2019
    Args:
        rot_6d (B x 6): Batch of 6D Rotation representation.
    Returns:
        Rotation matrices (B x 3 x 3).
    """
    rot_6d = rot_6d.view(-1, 3, 2)
    a1 = rot_6d[:, :, 0]
    a2 = rot_6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rot6d(rotmat):
    """
    Convert rotation matrix to 6D rotation representation.
    Args:
        rotmat (B x 3 x 3): Batch of rotation matrices.
    Returns:
        6D Rotations (B x 3 x 2).
    """
    return rotmat.view(-1, 3, 3)[:, :, :2]


def spherical_to_cartesian(theta, phi, radius=1.0):
    """
    Converts from spherical coordinates to cartesian coordinates. Spherical coordinates
    are defined according to the physics convention (theta elevation, phi azimuth).

    https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates

    Args:
        theta (tensor): elevation.
        phi (tensor): azimuth.
        radius (tensor): radius. Defaults to 1.

    Returns:
        (x, y, z)
    """
    x = radius * torch.sin(theta) * torch.cos(phi)
    y = radius * torch.sin(theta) * torch.sin(phi)
    z = radius * torch.cos(theta)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    """
    Converts spherical coordinates to cartesian coordinates.

    Args:
        x (tensor).
        y (tensor).
        z (tensor).

    Returns:
       (theta, phi)
    """
    theta = torch.arccos(z)
    phi = torch.atan2(y, x)
    return theta, phi


def create_sphere(level=4, device=None):
    """
    Creates a unit ico-sphere.
    """
    mesh = pytorch3d.utils.ico_sphere(level=level, device=device)
    return mesh.verts_padded()[0], mesh.faces_padded()[0]


def unwrap_uv_map(height=256, width=256):
    """
    Samples spherical coordinates to unwrap a UV map.

    Args:
        height (int).
        width (int).

    Returns:
        Spherical coordinates (H,W,3).
    """
    theta_ = torch.linspace(0, np.pi, height)
    phi_ = torch.linspace(-np.pi, np.pi, width)
    theta, phi = torch.meshgrid(theta_, phi_)
    x, y, z = spherical_to_cartesian(theta, phi)
    return torch.dstack((x, y, z))
