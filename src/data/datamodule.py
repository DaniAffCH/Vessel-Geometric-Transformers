from typing import List, Tuple

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader

from config.dataclasses import DatasetConfig
from src.data.dataset import VesselDataset
from src.data.vessel import Vessel
from src.lib import (
    PointGeometricAlgebra,
    ScalarGeometricAlgebra,
    TranslationGeometricAlgebra,
)


class InMemorySubset(Subset, InMemoryDataset):  # type: ignore[misc]
    """
    A subset of an in-memory dataset.

    This class represents a subset of an in-memory dataset.
    It inherits from the `Subset` class and the `InMemoryDataset` class.

    Args:
        dataset (InMemoryDataset): The original in-memory dataset.
        indices (List[int]): The indices of the subset.

    Returns:
        Vessel: The subset element at the given index.
    """

    def __init__(self, dataset: InMemoryDataset, indices: List[int]):
        Subset.__init__(self, dataset, indices)

    def __getitem__(self, idx: int) -> Vessel:
        elem: Vessel = Subset.__getitem__(self, idx)
        return elem


class VesselDataModule(L.LightningDataModule):  # type: ignore[misc]
    """
    LightningDataModule for handling vessel data.

    Args:
        config (DatasetConfig): The configuration for the dataset.

    Attributes:
        config (DatasetConfig): The configuration for the dataset.
        data (VesselDataset): The vessel dataset.
        train_set (InMemorySubset): The training subset of the dataset.
        val_set (InMemorySubset): The validation subset of the dataset.
        test_set (InMemorySubset): The test subset of the dataset.
    """

    def __init__(self, config: DatasetConfig):
        super().__init__()
        self.config = config
        self.data = VesselDataset(config, "complete")
        self.train_set, self.val_set, self.test_set = (
            VesselDataModule.random_split(
                self.data,
                [
                    self.config.train_size,
                    self.config.val_size,
                    self.config.test_size,
                ],
            )
        )

    def train_dataloader(self) -> GeometricDataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.config.batch_size,
            collate_fn=collate_vessels,
        )

    def val_dataloader(self) -> GeometricDataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.config.batch_size,
            collate_fn=collate_vessels,
        )

    def test_dataloader(self) -> GeometricDataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.config.batch_size,
            collate_fn=collate_vessels,
        )

    @staticmethod
    def random_split(
        dataset: InMemoryDataset, ratios: List[float]
    ) -> List[InMemorySubset]:
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
            splitted_datasets.append(InMemorySubset(dataset, subset_indices))
            start_idx = end_idx

        return splitted_datasets


def collate_vessels(batch: List[Vessel]) -> Tuple[Tensor, Tensor]:
    """
    Collates a batch of Vessel objects into padded tensors.

    Args:
        batch (List[Vessel]): A list of Vessel objects.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the padded batch tensor
        and the corresponding masks tensor.
    """

    elem: Vessel = batch[0]
    assert isinstance(elem, (Vessel, Data)), "DataLoader found invalid type"

    max_size = max(vessel.pos.shape[0] for vessel in batch)
    padded_batch: List[Tensor] = []
    masks: List[Tensor] = []

    for vessel in batch:
        # normalizing to be within 0 and 1
        wss = normalize(vessel.wss)
        pos = normalize(vessel.pos)
        pressure = normalize(vessel.pressure)

        # extracing the geometric algebra elements
        ga_wss = TranslationGeometricAlgebra.fromElement(wss)
        ga_pos = PointGeometricAlgebra.fromElement(pos)
        ga_pressure = ScalarGeometricAlgebra.fromElement(pressure)
        tensor = torch.stack((ga_pos, ga_wss, ga_pressure))

        # padding to the right with max_size-vessel.pos.shape[0] elements
        pad_value = -1  # hardcodato
        padding = (0, 0, 0, max_size - vessel.pos.shape[0])
        padded_tensor = F.pad(tensor, padding, "constant", pad_value)
        padded_batch.append(padded_tensor)

        # Mask: 1 for real data, 0 for padded data
        mask = (padded_tensor != pad_value).float()
        masks.append(mask)

    return torch.stack(padded_batch), torch.stack(masks)


def normalize(data: Tensor) -> Tensor:
    min = data.min()
    max = data.max()
    data = (data - min) / (
        max - min + 1e-6
    )  # Adding epsilon to avoid division by zero
    return data
