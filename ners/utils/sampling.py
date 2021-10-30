"""
This module implements utility functions for sampling points from
batches of meshes.

The function here is borrowed from pytorch3d's implementation with one
minor modification to allow sampling proportional to user-specified power
of area.
"""
from typing import Tuple

import torch
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals


def sample_consistent_points(verts, faces, trg_verts_list, num_samples: int = 10000):
    sample_face_idxs = sample_faces(verts, faces, num_samples)
    barycentric_coords = _rand_barycentric_coords(
        num_samples, 1, verts.dtype, verts.device
    )
    samples = []
    for vs in trg_verts_list:
        samples.append(
            sample_points_from_faces(vs, faces, sample_face_idxs, barycentric_coords)
        )
    return samples


def sample_faces(verts, faces, num_samples: int = 10000):
    with torch.no_grad():
        areas, _ = mesh_face_areas_normals(verts, faces)
        sample_face_idxs = areas.multinomial(num_samples, replacement=True)
        return sample_face_idxs


def sample_points_from_faces(verts, faces, face_idx, barycentric_coords):
    w0, w1, w2 = barycentric_coords
    face_verts = verts[faces.long()]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Use the barycentric coords to get a point on each sampled face.
    a = v0[face_idx]  # (N, num_samples, 3)
    b = v1[face_idx]
    c = v2[face_idx]
    samples = w0 * a + w1 * b + w2 * c
    return samples


def _rand_barycentric_coords(
    size1, size2, dtype, device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper function to generate random barycentric coordinates which are uniformly
    distributed over a triangle.
    Args:
        size1, size2: The number of coordinates generated will be size1*size2.
                      Output tensors will each be of shape (size1, size2).
        dtype: Datatype to generate.
        device: A torch.device object on which the outputs will be allocated.
    Returns:
        w0, w1, w2: Tensors of shape (size1, size2) giving random barycentric
            coordinates
    """
    uv = torch.rand(2, size1, size2, dtype=dtype, device=device)
    u, v = uv[0], uv[1]
    u_sqrt = u.sqrt()
    w0 = 1.0 - u_sqrt
    w1 = u_sqrt * (1.0 - v)
    w2 = u_sqrt * v
    return w0, w1, w2
