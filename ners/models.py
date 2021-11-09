import copy

import numpy as np
import pytorch3d
import torch
import torch.nn as nn
import trimesh
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.structures import Meshes
from tqdm.auto import tqdm

import ners.utils.geometry as geom_util
from ners.utils import sample_consistent_points


def pretrain_template_uv(
    template_uv,
    verts=None,
    faces=None,
    extents=None,
    num_iterations=1000,
    num_samples=1000,
    sphere_level=5,
    device="cuda:0",
    pbar=True,
):
    """
    Pretrains the template UV shape model. Must be initialized either with vertices and
    faces or with 3D cuboid extents.

    Args:
        verts (torch.Tensor): (N_v, 3) tensor of vertices.
        faces (torch.Tensor): (N_f, 3) tensor of faces.
        extents (list): list of 3D cuboid extents (w, h, d).

    Returns:
        template_uv (TemplateUV): pretrained template UV shape model mapping from uv
            coordinates (..., 3) to 3D vertex coordinates (..., 3).
    """
    template_uv = template_uv.to(device)
    if verts is None:
        tmesh = trimesh.creation.box(extents=extents)
        verts = torch.tensor(tmesh.vertices, device=device).float()
        faces = torch.tensor(tmesh.faces, device=device).long()
    else:
        verts = verts.to(device)
        faces = faces.to(device)
    verts_sphere = verts / verts.norm(dim=-1, keepdim=True)
    optim = torch.optim.Adam(template_uv.parameters(), lr=0.001)
    sphere_vs, sphere_fs = geom_util.create_sphere(level=sphere_level, device=device)
    loop = tqdm(range(num_iterations)) if pbar else range(num_iterations)
    for _ in loop:
        optim.zero_grad()
        targets, uvs = sample_consistent_points(
            verts, faces, [verts, verts_sphere], num_samples
        )
        pred_vs = template_uv(uvs.to(device), normalize=True)
        sv = (sphere_vs @ geom_util.random_rotation(device)).unsqueeze(0)
        meshes = Meshes(template_uv(sv, normalize=True), sphere_fs.unsqueeze(0))
        loss_reconstruction = torch.mean((pred_vs - targets.to(device)) ** 2)
        loss_laplacian = mesh_laplacian_smoothing(meshes)
        loss = 20 * loss_reconstruction + loss_laplacian
        loss.backward()
        optim.step()
        loop.set_description(f"Template: {loss.item():.4f}")

    return template_uv


def shape_model_to_mesh(shape_model, sphere_level=4, textures=None):
    device = shape_model.get_device(default_device="cuda:0")
    sphere_vs, sphere_fs = geom_util.create_sphere(level=sphere_level, device=device)
    if textures is None:
        textures = pytorch3d.renderer.TexturesVertex((sphere_vs.unsqueeze(0) + 1) / 2)
    mesh = Meshes(
        [shape_model(sphere_vs)],
        [sphere_fs],
        textures=textures,
    )
    return mesh.to(device)


class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=6, omega0=0.1):
        """
        Positional Embedding implementation (adapted from Pytorch3D).

        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**self.n_harmonic_functions * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**self.n_harmonic_functions * x[..., i])
            ]

        Note that `x` is also premultiplied by `omega0` before
        evaluting the harmonic functions.
        """
        super().__init__()
        self.register_buffer(
            "frequencies",
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class BaseNetwork(nn.Module):
    def __init__(self, n_harmonic_functions=6, omega0=0.1):
        super().__init__()
        self.positional_encoding = HarmonicEmbedding(n_harmonic_functions, omega0)

    def get_device(self, default_device=None):
        """
        Returns which device the module is on. If wrapped in DataParallel, will return
        the default device.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            return default_device


class TemplateUV(BaseNetwork):
    def __init__(self, num_layers=3, input_size=3, output_size=3, hidden_size=256, L=8):
        input_size = L * 2 * input_size
        super().__init__(n_harmonic_functions=L, omega0=np.pi)
        layers = []
        for i in range(num_layers - 1):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        nn.init.xavier_uniform_(layers[-1].weight, gain=0.001)
        nn.init.zeros_(layers[-1].bias)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, normalize=True):
        temp_device = x.device
        x = x.to(self.get_device(temp_device))
        if normalize:
            x = x / (x.norm(dim=-1, keepdim=True))
        h = self.positional_encoding(x)
        h = self.mlp(h)
        return (x + h).to(temp_device)


class DeltaUV(BaseNetwork):
    def __init__(self, num_layers=3, input_size=3, output_size=3, hidden_size=256, L=8):
        input_size = L * 2 * input_size
        super().__init__(n_harmonic_functions=L)
        layers = []
        for i in range(num_layers - 1):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        nn.init.xavier_uniform_(layers[-1].weight, gain=0.001)
        nn.init.zeros_(layers[-1].bias)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        temp_device = x.device
        x = x.to(self.get_device(temp_device))
        x = self.positional_encoding(x)
        x = self.mlp(x)
        return x.to(temp_device)


class ImplicitTextureNet(BaseNetwork):
    def __init__(
        self,
        num_layers=8,
        input_size=3,
        hidden_size=256,
        output_size=3,
        L=6,
        max_batch_size=10000,
        output_activation="sigmoid",
        gain=0.01,
    ):
        """
        Texture prediction network mapping UV to RGB.

        Args:
            num_layers (int, optional): Number of layers. Defaults to 12.
            input_size (int, optional): Dimension of input. Defaults to 3.
            hidden_size (int, optional): Dimension of hidden layers. Defaults to 256.
            output_size (int, optional): Dimension of output. Defaults to 3.
            L (int, optional): Number of frequencies for positional encoding. Defaults
                to 6.
            max_batch_size (int, optional): Maximum batch size. If over, automatically
                computes separate batches when using `forward_batched`. Defaults to
                10000.
            output_activation (str, optional): Output activation function can be
                "sigmoid" if outputting RGB or "tanh" if outputting deltas. Defaults to
                "sigmoid".
            gain (float, optional): Gain for output activation to initialize near 0.5.
        """
        input_size = input_size * L * 2
        super().__init__(n_harmonic_functions=L)
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        norm = nn.LayerNorm(hidden_size)
        layers = [nn.Linear(input_size, hidden_size), norm, nn.LeakyReLU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(norm)
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        nn.init.xavier_uniform_(layers[-1].weight, gain=gain)
        nn.init.zeros_(layers[-1].bias)

        self.mlp = nn.Sequential(*layers)
        if output_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        elif output_activation == "tanh":
            self.final_activation = nn.Tanh()
        else:
            raise Exception(
                f"Final activation must be sigmoid or tanh. Got: {output_activation}."
            )
        self.max_batch_size = max_batch_size

    def forward(self, x, normalize=True):
        """
        Args:
            x: (B,3)

        Returns:
            y: (B,3)
        """
        shape = x.shape
        x = x.reshape(-1, shape[-1])

        # The points outside of the mesh also get passed into TexNet, which is a lot of
        # unnecessary computation. We will skip over those points, which correspond to
        # (0, 0, 0)
        mask = torch.any(x != 0, dim=1)
        x = x[mask]
        temp_device = x.device
        if torch.any(mask):
            x = x.to(self.get_device(temp_device))
            if normalize:
                x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)  # Project to sphere.
            x = self.positional_encoding(x)
            x = self.mlp(x)
            x = self.final_activation(x)
            x = x.to(temp_device)
        y = torch.ones(len(mask), self.output_size, device=temp_device)
        y[mask] = x
        y = y.reshape(shape[:-1] + (-1,))
        return y.float()

    def forward_batched(self, x, batch_size=None, normalize=True):
        """
        Computes forward pass using minibatches to reduce memory usage of forward pass.

        Args:
            x (B,3).

        Returns:
            y (B,3).
        """
        n = x.shape[0]
        b = self.max_batch_size if batch_size is None else batch_size
        y = []
        for i in range(0, n, b):
            pred = self.forward(
                x[i : i + b],
                normalize=normalize,
            )
            y.append(pred)
        return torch.cat(y, dim=0)

    def save_model(self, name):
        path = f"{name}.pth"
        torch.save(self.state_dict(), path)

    def load_model(self, name):
        path = f"{name}.pth"
        self.load_state_dict(torch.load(path))

    def unwrap_uv_map(self, height=256, width=256, margin=0):
        """
        Unwraps the tex_net into a UV Image.

        Args:
            tex_net (ImplicitTextureNet): Texture network mapping from spherical coordinates
                to RGB.
            height (int, optional): Height of UV image. Defaults to 256.
            width (int, optional): Width of UV image. Defaults to 256.
            margin (float, optional): Width of redundancy on right side. Defaults to 0
                (no margin).

        Returns:
            tensor: Unwrapped texture (H, W, 3).
        """
        theta = torch.linspace(0, np.pi, height)
        phi = torch.linspace(-np.pi, np.pi * (1 + margin), width)
        theta, phi = torch.meshgrid(theta, phi)
        x, y, z = geom_util.spherical_to_cartesian(theta, phi)
        coords = torch.dstack((x, y, z)).cuda()
        shape = coords.shape[:2] + (3,)
        pred_texture = self.forward(coords.reshape(-1, 3))
        if pred_texture.shape[1] == 1:
            # For single channel environment maps.
            pred_texture = pred_texture.repeat(1, 3)
        pred_texture = pred_texture.reshape(shape)
        return pred_texture

    def clone(self):
        return copy.deepcopy(self)


class EnvironmentMap(ImplicitTextureNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_single_channel = True

    def forward(self, x, normalize=True, **kwargs):
        temp_device = x.device
        x = x.to(self.get_device(temp_device))

        if normalize:
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)  # Project to sphere.
        x = self.positional_encoding(x)
        # We will let the environment map's lighting be non-negative unbounded,
        # initialized at 3 (~75% brightness).
        x = torch.relu(self.mlp(x) + 3)
        if self.use_single_channel:
            x = x.mean(dim=-1, keepdim=True)
            # x = x.repeat((1,) * (x.ndim - 1) + (3,))  # repeat last dimension 3x
        return x.to(temp_device)


class Symmetrize(BaseNetwork):
    def __init__(self, uv3d, sym_axis=0):
        """
        Args:
            uv3d: module to symmetrize
            sym_axis: axis of symmetry: 0 -> X, 1 -> Y, 2-> Z
        """
        super(Symmetrize, self).__init__()
        self.uv3d = uv3d
        self.sym_axis = sym_axis

    def forward(self, uvs, *args):
        uvs_ref = uvs * 1

        uvs_ref[..., [self.sym_axis]] *= -1
        pred = self.uv3d(uvs, *args)
        pred_ref = self.uv3d(uvs_ref, *args)
        pred_ref[..., [self.sym_axis]] *= -1

        return (pred + pred_ref) * 0.5


def load_car_model(path="models/templates/car.pth"):
    template = TemplateUV(L=8, num_layers=3, hidden_size=256)
    template.load_state_dict(torch.load(path))
    return template
