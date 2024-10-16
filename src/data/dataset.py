import os
import random
from pathlib import Path
from typing import Callable, List, Optional

import gdown
import h5py
import torch
import tqdm
from torch_geometric.data import Data, InMemoryDataset

from config import DatasetConfig
from src.data.vessel import Vessel
from src.utils.definitions import Category

NUM_FEATURES = 4


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

    def __getitem__(self, idx: int) -> Vessel:
        """
        Get the item at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Data: The item at the specified index.
        """
        elem: Vessel = self.get(idx)
        return elem

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

    def get_data_from_h5(self, sample: h5py.Group, label: Category) -> Data:
        """
        Extract data from an HDF5 sample group and create a Data object.

        Args:
            sample (h5py.Group): The HDF5 group containing the sample data.

        Returns:
            Data: The created Data object.
        """
        # The face transpose is a trick to enable collate.
        return Data(
            wss=torch.from_numpy(sample["wss"][()]),
            pos=torch.from_numpy(sample["pos"][()]),
            pressure=torch.from_numpy(sample["pressure"][()]),
            face=torch.from_numpy(sample["face"][()].T).long(),
            inlet_index=torch.from_numpy(sample["inlet_idcs"][()]),
            label=label,
        )

    def process_h5(self, h5_path: str, label: Category) -> List[Data]:
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
                data_list.append(self.get_data_from_h5(f[sample_name], label))
        return data_list

    def process(self) -> None:
        """
        Process the raw data files, shuffle the data,
        and save the processed data.
        """
        data_list = []

        bifurcating_db_path = os.path.join(
            self.raw_paths[0], self.config.bifurcating_path
        )
        single_db_path = os.path.join(
            self.raw_paths[0], self.config.single_path
        )
        data_list.extend(
            self.process_h5(bifurcating_db_path, Category.Bifurcating)
        )
        data_list.extend(self.process_h5(single_db_path, Category.Single))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Shuffle the data list
        random.seed(42)
        random.shuffle(data_list)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
