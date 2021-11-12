"""
Script for running over all instances of MVMC.

Since running the script can take a long time, it is possible to parallelize across
different machines using the --index and --skip arguments.

Examples:
    python scripts/mvmc_driver.py --mvmc_path data/mvmc --index 0 --skip 1
"""
import argparse
import os
import subprocess

from tqdm.auto import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvmc_path", type=str, default="data/mvmc")
    parser.add_argument(
        "--index", default=0, type=int, help="Initial index to start at."
    )
    parser.add_argument(
        "--skip", default=1, type=int, help="Number of instances to skip at a time."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-run even if output exists."
    )
    return parser


def main(args):
    instance_ids = sorted(os.listdir(args.mvmc_path))
    base_cmd = [
        "python",
        "main.py",
        "--mvmc",
        "--symmetrize",
        "--export-mesh",
        "--predict-illumination",
    ]
    if args.force:
        base_cmd.append("--force")

    for instance_id in tqdm(instance_ids[args.index :: args.skip]):
        cmd = base_cmd + [
            "--instance-dir",
            os.path.join(args.mvmc_path, instance_id),
        ]
        print("Running:", " ".join(cmd))
        subprocess.call(cmd)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
