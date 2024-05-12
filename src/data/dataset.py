import os
from typing import Callable, List, Optional

import gdown
import h5py
import torch
from torch_geometric.data import Data, InMemoryDataset

from src.constants import (
    BIFURACTING_DB,
    DATASET_DOWNLOAD_PATH,
    DATASET_DRIVE_URL,
    DATASET_NAME,
    SINGLE_DB,
)


class VesselDataset(InMemoryDataset):  # type: ignore[misc]
    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable[[Data], Data]] = None,
        pre_transform: Optional[Callable[[Data], Data]] = None,
    ) -> None:
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> List[str]:
        root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        return [os.path.join(root, self.root, DATASET_NAME)]

    @property
    def has_download(self) -> bool:
        return True

    @property
    def processed_file_names(self) -> List[str]:
        return ["TODO -- Implementa"]

    def download(self) -> None:
        if not os.path.exists(self.root):
            os.mkdir(self.root)

        gdown.download_folder(DATASET_DRIVE_URL, output=DATASET_DOWNLOAD_PATH)

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
        data_list = list()
        with h5py.File(h5_path, "r") as f:
            for sample_name in f:
                assert (
                    len(f[sample_name].keys()) == 5
                ), f"Corrupted sample found, {sample_name}"
                data = self.get_data_from_h5(f[sample_name])
                data_list.append(data)
        return data_list

    def process(self) -> None:
        data_list = list()

        bifurcating_db_path = os.path.join(self.raw_paths[0], BIFURACTING_DB)
        single_db_path = os.path.join(self.raw_paths[0], SINGLE_DB)

        data_list.extend(self.process_h5(bifurcating_db_path))
        data_list.extend(self.process_h5(single_db_path))

        if self.pre_filter is not None:
            # This is necessary to not allocate additional memory
            i = 0
            while i < len(data_list):
                if not self.pre_filter(data_list[i]):
                    del data_list[i]
                else:
                    i += 1

        if self.pre_transform is not None:
            for i in range(len(data_list)):
                data_list[i] = self.pre_transform(data_list[i])

        self.save(data_list, self.processed_paths[0])
