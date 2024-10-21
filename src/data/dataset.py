import os
import random
from pathlib import Path
from typing import Callable, List, Optional

import gdown
import h5py
import torch
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
        super(VesselDataset, self).__init__(root=config.download_path)
        self._createElementMapping()

    def _createElementMapping(self) -> None:
        """
        Creates and shuffles the list of samples
        """
        with h5py.File(self.raw_paths[Category.Bifurcating.value]) as f:
            self.mapping = [(e, Category.Bifurcating) for e in f]

        with h5py.File(self.raw_paths[Category.Single.value]) as f:
            self.mapping += [(e, Category.Single) for e in f]

        random.shuffle(self.mapping)

    def __getitem__(self, idx: int) -> Vessel:
        """
        Get the item at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Data: The item at the specified index.
        """
        # Load the item from the processed data saved in a .hdf5 file
        item_name, category = self.mapping[idx]
        with h5py.File(self.raw_paths[category.value], "r") as f:
            item: Vessel = self.get_data_from_h5(f[item_name], category)
        return item

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.mapping)

    @property
    def labels(self) -> torch.Tensor:
        """
        Get the labels of the dataset.

        Returns:
            torch.Tensor: The labels of the dataset.
        """
        labels = torch.zeros(len(self), dtype=torch.long)
        for i in range(len(self)):
            labels[i] = self[i].label.value
        return labels

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
        data_root = os.path.join(project_root, self.root)
        Path(data_root).mkdir(parents=True, exist_ok=True)
        return [
            os.path.join(data_root, Category.Bifurcating.name + ".hdf5"),
            os.path.join(data_root, Category.Single.name + ".hdf5"),
        ]

    @property
    def has_download(self) -> bool:
        """
        Indicate whether the dataset has a download method.

        Returns:
            bool: True if the dataset can be downloaded, else False.
        """
        return True

    @property
    def has_process(self) -> bool:
        """
        Indicate whether the dataset has a process method.

        Returns:
            bool: True if the dataset can be processed, else False.
        """
        return False

    def download(self) -> None:
        """
        Download the dataset files from the specified URL.
        """
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        gdown.download(
            id=self.config.bifurcating_id,
            output=self.raw_paths[Category.Bifurcating.value],
        )
        gdown.download(
            id=self.config.single_id,
            output=self.raw_paths[Category.Single.value],
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
