import argparse
from PIL import Image
import io
import os
import requests
import pyarrow.parquet as pq

from .util import root_data_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(root_data_dir(), "raw_downloaded"),
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=os.path.join(root_data_dir(), "meta", "metadata_0000.parquet"),
    )
    args = parser.parse_args()

    df = pq.read_table(args.metadata_path).to_pandas()

    success_dir = os.path.join(args.output_dir, "success")
    failure_dir = os.path.join(args.output_dir, "failure")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failure_dir, exist_ok=True)

    def download_image(row):
        key = row["key"]
        url = row["url"]
        expected_size = (row["width"], row["height"])

        file_path = os.path.join(success_dir, key)
        fail_path = os.path.join(success_dir, key)
        if os.path.exists(file_path) or os.path.exists(fail_path):
            return

        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()  # Raise an error for bad status codes

            img = Image.open(io.BytesIO(response.content))
            if img.size != expected_size:
                raise ValueError(
                    f"size {img.size} mismatches expected size {expected_size}"
                )

            with open(file_path, "wb") as f:
                f.write(response.content)
        except Exception as exc:
            print(f"Error processing image {url}: {exc}")
            with open(fail_path, "w") as f:
                f.write(str(exc))

    df.apply(download_image, axis=1)


if __name__ == "__main__":
    main()
