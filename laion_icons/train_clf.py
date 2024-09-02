"""
Train a small classifier on image features using a small dataset of
classified images as training data.

The small classifier can be used to filter and download a larger number
of positive images without needing to download all images.
"""

import argparse
import os
import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .util import root_data_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pos_dir",
        type=str,
        default=os.path.join(root_data_dir(), "labeled", "pos"),
    )
    parser.add_argument(
        "--neg_dir",
        type=str,
        default=os.path.join(root_data_dir(), "labeled", "neg"),
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=os.path.join(root_data_dir(), "meta", "metadata_0000.parquet"),
    )
    parser.add_argument(
        "--emb_path",
        type=str,
        default=os.path.join(root_data_dir(), "embs", "img_emb_0000.npy"),
    )
    args = parser.parse_args()

    print(f"loading features and labels from {args.pos_dir} and {args.neg_dir}...")

    img_emb = np.load(args.emb_path)
    df = pq.read_table(args.metadata_path).to_pandas()

    pos_ids = set(os.listdir(args.pos_dir))
    neg_ids = set(os.listdir(args.neg_dir))

    features = []
    labels = []

    for index, row in df.iterrows():
        key = row["key"]
        if key in pos_ids:
            labels.append(1)
            features.append(img_emb[index])
        elif key in neg_ids:
            labels.append(0)
            features.append(img_emb[index])

    features = np.array(features)
    labels = np.array(labels)
    print(f"total of {len(labels)} examples loaded")

    print("running baseline...")
    dummy = DummyClassifier()
    dummy.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, dummy.predict(X_test))
    print(f"Baseline Accuracy: {accuracy:.4f}")

    print("training classifier...")

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, classifier.predict(X_test))
    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
