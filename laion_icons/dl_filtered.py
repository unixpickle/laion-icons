"""
Filter images using a classifier and dump the resulting metadatas.
"""

import argparse
from functools import partial
import io
import json
from multiprocessing import Pool
import os

import requests
from PIL import Image

from .util import root_data_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pending_dir",
        type=str,
        default=os.path.join(root_data_dir(), "filtered", "pending"),
    )
    parser.add_argument(
        "--failed_dir",
        type=str,
        default=os.path.join(root_data_dir(), "filtered", "failed"),
    )
    parser.add_argument(
        "--success_dir",
        type=str,
        default=os.path.join(root_data_dir(), "filtered", "success"),
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default=os.path.join(root_data_dir(), "filtered", "download"),
    )
    parser.add_argument("--ar_limit", type=float, default=1.1)
    args = parser.parse_args()

    os.makedirs(args.failed_dir, exist_ok=True)
    os.makedirs(args.success_dir, exist_ok=True)
    os.makedirs(args.download_dir, exist_ok=True)

    with Pool(8) as p:
        for _ in p.imap_unordered(
            partial(download_image, args), os.listdir(args.pending_dir)
        ):
            pass


def download_image(args, md):
    filename = md.split(".")[0]

    md = os.path.join(args.pending_dir, md)
    if not os.path.isfile(md):
        return
    with open(md, "r") as f:
        info = json.load(f)

    expected_size = (info["width"], info["height"])
    ar = info["width"] / info["height"]
    if ar < 1:
        ar = 1 / ar
    if ar > args.ar_limit:
        return

    try:
        response = requests.get(info["url"], timeout=5)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        if img.size != expected_size:
            raise ValueError(
                f"size {img.size} mismatches expected size {expected_size}"
            )
        with open(os.path.join(args.download_dir, filename), "wb") as f:
            f.write(response.content)
        os.rename(md, os.path.join(args.success_dir, os.path.basename(md)))
    except Exception as exc:
        print(f"Error processing image {info['url']}: {exc}")
        os.rename(md, os.path.join(args.failed_dir, os.path.basename(md)))


if __name__ == "__main__":
    main()
