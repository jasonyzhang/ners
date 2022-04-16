import util
from data import get_split_dataset
from encoder import SpatialEncoder
import numpy as np
import torch


def extra_args(parser):
    parser.add_argument(
        "--subset", "-S", type=int, default=0, help="Subset in data to use"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split of data to use train | val | test",
    )
    parser.add_argument(
        "--source",
        "-P",
        type=str,
        default="64",
        help="Source view(s) in image, in increasing order. -1 to do random",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=40,
        help="Number of video frames (rotated views)",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=-10.0,
        help="Elevation angle (negative is above)",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Video scale relative to input size"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.0,
        help="Distance of camera from origin, default is average of z_far, z_near of dataset (only for non-DTU)",
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS of video")
    return parser


def make_encoder(conf, **kwargs):
    net = SpatialEncoder.from_conf(conf, **kwargs)
    return net


args, conf = util.args.parse_args(extra_args)
args.resume = True

device = util.get_cuda(args.gpu_id[0])

dset = get_split_dataset(
    args.dataset_format, args.datadir, want_split=args.split, training=False
)

print("====dir====", args.datadir)

data = dset[args.subset]
data_path = data["path"]
print("Data instance loaded:", data_path)

images = data["images"]

conf = conf["model"]["encoder"]#ConfigTree([('backbone', 'resnet34'), ('pretrained', True), ('num_layers', 4)])
net = make_encoder(conf).to(device=device)
net.load_weights(args)

latent = net(images)
latent = torch.mean(latent, dim=0).numpy()
np.savez('mean_latent', latent)
