"""
Use the ChatGPT API to label images as vector graphics.
"""

import argparse
import time
import os
import openai
import base64

from .util import root_data_dir

ERROR_SLEEP = 20
SUCCESS_SLEEP = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join(root_data_dir(), "raw_downloaded", "success"),
    )
    parser.add_argument(
        "--labeled_dir", type=str, default=os.path.join(root_data_dir(), "labeled")
    )
    args = parser.parse_args()

    print(f"labeling from {args.input_dir}")

    # Directories for images
    pos_dir = os.path.join(args.labeled_dir, "pos")
    neg_dir = os.path.join(args.labeled_dir, "neg")
    unknown_dir = os.path.join(args.labeled_dir, "unknown")
    unlabeled_dir = args.input_dir

    # Create directories if they don't exist
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)
    os.makedirs(unknown_dir, exist_ok=True)

    for image_name in os.listdir(unlabeled_dir):
        image_path = os.path.join(unlabeled_dir, image_name)
        if not os.path.isfile(image_path):
            continue

        already_labeled = False
        for out_dir in [pos_dir, neg_dir, unknown_dir]:
            if os.path.exists(os.path.join(out_dir, image_name)):
                already_labeled = True
        if already_labeled:
            continue

        label = label_image(image_path)
        if label is None:
            time.sleep(ERROR_SLEEP)
            continue
        label = label.lower().rstrip(".")

        if label == "yes":
            out_path = os.path.join(pos_dir, image_name)
        elif label == "no":
            out_path = os.path.join(neg_dir, image_name)
        else:
            out_path = os.path.join(unknown_dir, image_name)
        os.symlink(
            os.path.relpath(image_path, start=os.path.dirname(out_path)), out_path
        )

        print(f"labeled {image_name}: {label}")
        time.sleep(SUCCESS_SLEEP)


def image_to_base64(image_path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def label_image(image_path):
    base64_image = image_to_base64(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {
                    "type": "text",
                    "text": "Is this image a simple vector icon, and/or a simple icon in general? Please respond with 'yes' or 'no' only. If it appears to be a photo and not an icon, reply 'no'. If it is an icon or a vector graphic, reply 'yes'. Respond 'no' for computer screenshots. Respond 'no' for complex diagrams or charts, since these are not simple enough.",
                },
            ],
        }
    ]

    try:
        response = openai.ChatCompletion.create(model="gpt-4o-mini", messages=messages)
        label = response["choices"][0]["message"]["content"].strip().rstrip(".").lower()
        return label
    except Exception as e:
        print(f"Error labeling image {image_path}: {e}")
        if "unsupported image" in str(e):
            return "unsupported"
        return None


if __name__ == "__main__":
    main()
