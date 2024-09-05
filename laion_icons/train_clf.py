"""
Train a small classifier on image features using a small dataset of
classified images as training data.

The small classifier can be used to filter and download a larger number
of positive images without needing to download all images.
"""

import argparse
from datetime import datetime
import os
import pickle
import numpy as np
import pyarrow.parquet as pq
from sklearn.svm import SVC
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
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(
            root_data_dir(),
            "classifiers",
            f"clf_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pkl",
        ),
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

    test_accs = []
    test_baseline = []
    train_accs = []
    train_baseline = []
    for seed in [0, 1, 2, 3, 4]:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=seed
        )

        dummy = DummyClassifier()
        dummy.fit(X_train, y_train)
        train_baseline.append(accuracy_score(y_train, dummy.predict(X_train)))
        test_baseline.append(accuracy_score(y_test, dummy.predict(X_test)))

        classifier = SVC(probability=True, C=5.0)
        classifier.fit(X_train, y_train)

        train_accs.append(accuracy_score(y_train, classifier.predict(X_train)))
        test_accs.append(accuracy_score(y_test, classifier.predict(X_test)))
    
    print(f'test_acc: {np.mean(test_accs)} (baseline: {np.mean(test_baseline)})')
    print(f'train_acc: {np.mean(train_accs)} (baseline: {np.mean(train_baseline)})')

    classifier = SVC(probability=True, C=5.0)
    classifier.fit(features, labels)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(classifier, f)


if __name__ == "__main__":
    main()
