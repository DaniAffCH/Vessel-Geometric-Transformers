import os

import gdown

from src.constants import DATASET_DRIVE_URL, DATASET_NAME


def download_dataset(
    download_path: str = "dataset", force_download: bool = False
) -> None:

    if not os.path.exists(download_path):
        os.mkdir(download_path)

    dataset_path = os.path.join(download_path, DATASET_NAME)

    if not os.path.exists(dataset_path) or force_download:
        gdown.download_folder(DATASET_DRIVE_URL, output=download_path)
    else:
        print(f"Dataset found at {dataset_path}")
