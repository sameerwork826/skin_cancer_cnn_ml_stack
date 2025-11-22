#!/usr/bin/env bash
# Download HAM10000 dataset from Kaggle (requires kaggle CLI and kaggle.json)
# Dataset: kmader/skin-cancer-mnist-ham10000

set -e
mkdir -p data
echo "Make sure ~/.kaggle/kaggle.json exists (your Kaggle API token)."
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data --unzip
# The Kaggle dataset usually contains HAM10000_metadata.csv and images
echo "Downloaded dataset into data/ (check files)."
