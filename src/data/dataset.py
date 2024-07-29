import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import gdown
import h5py
import torch
import tqdm
from torch.utils.data import Subset
from torch_geometric.data import Data, InMemoryDataset

from config import DatasetConfig


class VesselDataset(InMemoryDataset):  # type: ignore[misc]
    """
    Dataset class for handling vessel data.

    This class inherits from InMemoryDataset and is used to
    handle vessel dataset.
    The data is expected to be in HDF5 format.
    """

    def __init__(
        self,
        config: DatasetConfig,
        purpose: str,
        transform: Optional[Callable[[Data], Data]] = None,
        pre_transform: Optional[Callable[[Data], Data]] = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            config (DatasetConfig): Configuration object containing
                                    dataset paths and parameters.
            purpose (str): Purpose of the dataset,
                           e.g., 'train', 'val', 'test'.
            transform (Callable, optional): A function/transform that takes in
                                            a Data object and returns a
                                            transformed version.
            pre_transform (Callable, optional): A function/transform that is
                                                applied before the transform.
        """
        self.config = config
        self.purpose = purpose
        super(VesselDataset, self).__init__(
            config.download_path, transform, pre_transform
        )
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data = None
            self.slices = None

    @property
    def raw_file_names(self) -> List[str]:
        """
        The names of the files in the raw directory that
        must be present to skip downloading.

        Returns:
            List[str]: The names of the files required for the dataset.
        """
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        return [os.path.join(project_root, self.root, self.config.name)]

    @property
    def processed_file_names(self) -> List[str]:
        """
        The names of the files in the processed directory that
        must be present to skip processing.

        Returns:
            List[str]: The names of the processed files.
        """
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        data_root = os.path.join(
            project_root, self.root, "processed", self.purpose
        )
        Path(data_root).mkdir(parents=True, exist_ok=True)
        return [os.path.join(data_root, "data.pt")]

    @property
    def has_download(self) -> bool:
        """
        Indicate whether the dataset has a download method.

        Returns:
            bool: True if the dataset can be downloaded, else False.
        """
        return True

    def download(self) -> None:
        """
        Download the dataset files from the specified URL.
        """
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        gdown.download_folder(
            self.config.drive_url, output=self.config.download_path
        )

    def get_data_from_h5(self, sample: h5py.Group) -> Data:
        """
        Extract data from an HDF5 sample group and create a Data object.

        Args:
            sample (h5py.Group): The HDF5 group containing the sample data.

        Returns:
            Data: The created Data object.
        """
        return Data(
            wss=torch.from_numpy(sample["wss"][()]),
            pos=torch.from_numpy(sample["pos"][()]),
            pressure=torch.from_numpy(sample["pressure"][()]),
            face=torch.from_numpy(sample["face"][()].T).long(),
            inlet_index=torch.from_numpy(sample["inlet_idcs"][()]),
        )

    def process_h5(self, h5_path: str) -> List[Data]:
        """
        Process the HDF5 file and create a list of Data objects.

        Args:
            h5_path (str): The path to the HDF5 file.

        Returns:
            List[Data]: A list of created Data objects.
        """
        data_list = []
        with h5py.File(h5_path, "r") as f:
            for sample_name in tqdm.tqdm(f, desc=f"Loading {h5_path}"):
                assert (
                    len(f[sample_name].keys()) == 5
                ), f"Corrupted sample found, {sample_name}"
                data = self.get_data_from_h5(f[sample_name])
                data_list.append(data)
        return data_list

    def process(self) -> None:
        """
        Process the raw data files and save the processed data.
        """
        data_list = []

        bifurcating_db_path = os.path.join(
            self.raw_paths[0], self.config.bifurcating_path
        )
        single_db_path = os.path.join(
            self.raw_paths[0], self.config.single_path
        )

        data_list.extend(self.process_h5(bifurcating_db_path))
        data_list.extend(self.process_h5(single_db_path))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def random_split(
    dataset: InMemoryDataset, ratios: List[float]
) -> List[Subset]:
    """
    Split the dataset into subsets based on the given ratios.

    Args:
        dataset (InMemoryDataset): The dataset to split.
        ratios (List[float]): The ratios for splitting the dataset
                              (e.g., [0.7, 0.2, 0.1]).

    Returns:
        List[Subset]: A list of subsets created from the dataset.
    """
    assert (
        sum(ratios) == 1.0
    ), "The dataset splits (train + val + test) must sum up to 1"

    n_samples = len(dataset)
    indices = list(range(n_samples))

    split_indices = [int(ratio * n_samples) for ratio in ratios]
    split_indices[-1] = n_samples - sum(split_indices[:-1])

    start_idx = 0
    splitted_datasets = []

    for size in split_indices:
        end_idx = start_idx + size
        subset_indices = indices[start_idx:end_idx]
        splitted_datasets.append(Subset(dataset, subset_indices))
        start_idx = end_idx

    return splitted_datasets


def get_datasets(dataset_config: DatasetConfig) -> Dict[str, Subset]:
    """
    Get the training, validation, and test datasets.

    Args:
        dataset_config (DatasetConfig): The configuration object containing
                                        dataset parameters.

    Returns:
        Dict[str, Subset]: A dictionary containing the training,
                           validation, and test datasets.
    """
    dataset = VesselDataset(dataset_config, "complete")

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [
            dataset_config.train_size,
            dataset_config.val_size,
            dataset_config.test_size,
        ],
    )

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
