# Goal

Create a dataset of simple images and icons to train toy text->image models on. This could be a good test dataset for low-compute experiments on text->image, since the problem should be a lot easier than generating natural images.

# Usage

Set the `LAION_ICONS_DIR` environment variable to the data directory of the download. In my case, I `cd` into the `laion-icons` root directory, and `mkdir data` and set `LAION_ICONS_DIR=data`.

The second, third, and fourth step are specifically for training a classifier that we will use to filter the data before downloading actual large amounts of images. You may skip these steps and use a pre-trained classifier.

## Download embeddings and metadata

```
bash laion_icons/dl_emb_files.sh
```

The outputs will go to `$LAION_ICONS_DIR/embs` and `$LAION_ICONS_DIR/meta`. This command hinges on the LAION mirror not being down, which it often tends to be.

## Download some data for a classifier

```
python -m laion_icons.dl_images
```

This command will non-discriminately download images from the first shard of LAION. The goal is to gather some data to train a feature classifier.

The output will go to `$LAION_ICONS_DIR/raw_downloaded`.

## Label training data for classifier

We will label each image so that we can train a small feature classifier. For this step, you need an API token for the OpenAI API. We only need to label a few thousand images to get a good classifier.

```
python -m laion_icons.label
```

## Train a classifier

```
python -m laion_icons.train_clf
```

The classifier will be saved to `$LAION_ICONS_DIR/classifiers` as a `.pkl` file.

You can download my pre-trained classifier here: [https://data.aqnichol.com/laion_icons/clf_20240906-190112.pkl](https://data.aqnichol.com/laion_icons/clf_20240906-190112.pkl)

## Filter images using classifier

```
python -m laion_icons.filter_images --classifier path/to/classifier.pkl
```

This script uses the pre-computed CLIP embeddings from the LAION dataset to determine which images to download. The metadatas for images that are accepted by the filter are written to `$LAION_ICONS_DIR/filtered/pending`. The next script will move this metadata around depending on whether or not the image successfully downloaded.

## Download filtered images

```
python -m laion_icons.dl_filtered
```

This script is the workhorse here: you will use it to download potentially millions of images.

Output images will be written to `$LAION_ICONS_DIR/filtered/download` and metadata to `$LAION_ICONS_DIR/filtered/success`.

