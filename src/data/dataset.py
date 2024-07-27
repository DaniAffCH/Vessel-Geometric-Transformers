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
    def __init__(
        self,
        config: DatasetConfig,
        purpose: str,
        transform: Optional[Callable[[Data], Data]] = None,
        pre_transform: Optional[Callable[[Data], Data]] = None,
    ) -> None:
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
        The name of the files in the :obj:`self.raw_dir` folder that must be
        present in order to skip downloading. Acts as base property needed for
        `raw_paths`.

        Returns
        -------
        List[str]
            The name of the file(s) that need to be downloaded.
        """
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        return [os.path.join(project_root, self.root, self.config.name)]

    @property
    def processed_file_names(self) -> List[str]:
        """The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing. Acts as base property
        needed for `processed_paths`.

        Returns
        -------
        List[str]
            The name of the file(s) that are the result of running the download
            and process methods.
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
        return True

    def download(self) -> None:
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        gdown.download_folder(
            self.config.drive_url, output=self.config.download_path
        )

    def get_data_from_h5(self, sample: h5py.Group) -> Data:
        # It can be expanded depending on what we need
        # https://github.com/sukjulian/coronary-mesh-convolution/blob/main/datasets.py#L75
        return Data(
            wss=torch.from_numpy(sample["wss"][()]),
            pos=torch.from_numpy(sample["pos"][()]),
            pressure=torch.from_numpy(sample["pressure"][()]),
            face=torch.from_numpy(sample["face"][()].T).long(),
            inlet_index=torch.from_numpy(sample["inlet_idcs"][()]),
        )

    def process_h5(self, h5_path: str) -> List[Data]:
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
    assert (
        dataset_config.train_size
        + dataset_config.val_size
        + dataset_config.test_size
        == 1.0
    ), "The dataset splits (train + val + test) must sum up to 1"

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
