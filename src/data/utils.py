import os

import gdown

from src.constants import (
    DATASET_DOWNLOAD_PATH,
    DATASET_DRIVE_URL,
    DATASET_NAME,
)


def download_dataset(force_download: bool = False) -> None:

    if not os.path.exists(DATASET_DOWNLOAD_PATH):
        os.mkdir(DATASET_DOWNLOAD_PATH)

    dataset_path = os.path.join(DATASET_DOWNLOAD_PATH, DATASET_NAME)

    if not os.path.exists(dataset_path) or force_download:
        gdown.download_folder(DATASET_DRIVE_URL, output=DATASET_DOWNLOAD_PATH)
    else:
        print(f"Dataset found at {dataset_path}")
