"""
Filter images using a classifier and dump the resulting metadatas.
"""

import argparse
from datetime import datetime
import glob
import json
import os
import pickle
import numpy as np
import pyarrow.parquet as pq
import traceback
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .util import root_data_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--emb_dir",
        type=str,
        default=os.path.join(root_data_dir(), "embs"),
    )
    parser.add_argument(
        "--meta_dir",
        type=str,
        default=os.path.join(root_data_dir(), "meta"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(root_data_dir(), "filtered", "pending"),
    )
    parser.add_argument("--ar_limit", type=float, default=1.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    check_dirs = [
        os.path.join(os.path.dirname(args.output_dir), x) for x in ["success", "failed"]
    ] + [args.output_dir]

    print("loading classifier...")
    with open(args.classifier, "rb") as f:
        clf = pickle.load(f)

    for emb_path in glob.glob(os.path.join(args.emb_dir, "*.npy")):
        shard_id = os.path.basename(emb_path).split(".")[0].split("_")[-1]
        meta_path = os.path.join(args.meta_dir, f"metadata_{shard_id}.parquet")
        if not os.path.exists(meta_path):
            print(f"skipping shard {shard_id} because {meta_path} does not exist")
            continue

        print(f"working on shard {emb_path} ...")
        try:
            img_emb = np.load(emb_path)
            df = pq.read_table(meta_path).to_pandas()
        except:
            traceback.print_exc()
            continue
        for i in range(0, len(img_emb), args.batch_size):
            preds = clf.predict_proba(img_emb[i : i + args.batch_size])[:, 1].tolist()
            for j, p in enumerate(preds):
                record = {
                    k: next(iter(v.values()))
                    for k, v in df[i + j : i + j + 1].to_dict().items()
                }
                if any(
                    os.path.exists(os.path.join(x, f'{record["key"]}.json'))
                    for x in check_dirs
                ):
                    continue
                if p >= args.threshold:
                    ar = record["width"] / record["height"]
                    if ar < 1:
                        ar = 1 / ar
                    if ar > args.ar_limit:
                        continue
                    with open(
                        os.path.join(args.output_dir, f'{record["key"]}.json'), "w"
                    ) as f:
                        info = dict(
                            url=record["url"],
                            width=record["width"],
                            height=record["height"],
                            caption=record["caption"],
                        )
                        json.dump(info, f)


if __name__ == "__main__":
    main()
