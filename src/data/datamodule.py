import gc
from dataclasses import dataclass, field
from typing import List, Optional

import lightning as L
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from config.dataclasses import DatasetConfig
from src.data.dataset import VesselDataset
from src.data.vessel import Vessel, get_feature
from src.lib import (
    PlaneGeometricAlgebra,
    PointGeometricAlgebra,
    ScalarGeometricAlgebra,
    TranslationGeometricAlgebra,
)
from src.utils.definitions import Feature


class InMemorySubset(Subset, InMemoryDataset):  # type: ignore[misc]
    """
    A subset of an in-memory dataset.
    It is used to identify training, validation, and test sets.

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

    def getLabels(self) -> List[int]:
        return [self.dataset[i].label.value for i in self.indices]


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
        gc.collect()
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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.config.batch_size,
            collate_fn=lambda batch: collate_vessels(
                batch, self.config.features_size_limit
            ),
            shuffle=True,
            num_workers=16,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.config.batch_size,
            collate_fn=lambda batch: collate_vessels(
                batch, self.config.features_size_limit
            ),
            num_workers=16,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.config.batch_size,
            collate_fn=lambda batch: collate_vessels(
                batch, self.config.features_size_limit
            ),
            num_workers=16,
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

        # Making sure there are no leftovers due to fp32 precision
        split_indices[-1] = n_samples - sum(split_indices[:-1])

        start_idx = 0
        splitted_datasets = []

        for size in split_indices:
            end_idx = start_idx + size
            subset_indices = indices[start_idx:end_idx]
            splitted_datasets.append(InMemorySubset(dataset, subset_indices))
            start_idx = end_idx

        return splitted_datasets

    def extract_feature(self, feature: Feature) -> torch.Tensor:
        """
        Extracts a specific feature from the dataset. Every feature of the
        dataset is represented as a matrix of dimensions (n,d) where n is the
        size of the feature (number of vectors) and d the dimension of each
        feature vector. This function flattens the matrix of each data point to
        a one-dimensional vector, returning a vector of dimension (m,n*d) where
        m is the number of arteries in the dataset.

        m -> number of arteries
        b -> batch_size
        f -> number of features
        n -> number of vectors
        d -> vector dimension

        Args:
            feature (Feature): The feature to extract.

        Returns:
            torch.Tensor: The extracted feature.
        """
        dataloader = DataLoader(
            self.data,
            batch_size=1,
            collate_fn=lambda batch: collate_vessels(
                batch, self.config.features_size_limit, feature=feature
            ),
            num_workers=16,
        )

        feature_list = []
        label_list = []
        for batch in tqdm(dataloader, desc=f"Extracting {feature}"):
            assert isinstance(batch, VesselBatch), "Batch is not a VesselBatch"
            feature_list.append(batch.data)
            label_list.append(batch.labels.item())
        feature_tensor = torch.stack(feature_list)
        if feature_tensor.dim() == 4:
            feature_tensor = feature_tensor.unsqueeze(-1)
        label_tensor = torch.Tensor(label_list)

        return (
            rearrange(feature_tensor, "m b f n d -> m (b f n d)"),
            label_tensor,
        )


@dataclass
class VesselBatch:
    """
    Dataclass for a batch of Vessel objects
    """

    data: Tensor = field(default_factory=Tensor)
    mask: Tensor = field(default_factory=Tensor)
    labels: Tensor = field(default_factory=Tensor)


def collate_vessels(
    batch: List[Vessel],
    size_limit: int,
    pad_value: int = -0xDEADBEEF,
    feature: Optional[Feature] = None,
) -> VesselBatch:
    """
    Collates a batch of Vessel objects into padded tensors. Used by the
    dataloaders.

    Args:
        batch (List[Vessel]): A list of Vessel objects. size_limit (int):
        Maximum number of elements a feature can have pad_value (int): The
        value used to indicate padding

    Returns:
        VesselBatch: A VesselBatch object containing the padded batch
        tensor, masks tensor, and labels tensor.
    """

    elem: Vessel = batch[0]
    assert isinstance(elem, (Vessel, Data)), "DataLoader found invalid type"

    padded_data: List[Tensor] = []
    masks: List[Tensor] = []
    labels: List[int] = []

    for vessel in batch:

        if feature:
            tensor = get_feature(vessel, feature)
            if feature == Feature.FACE:
                tensor = tensor.T
            elements = [tensor]
        else:
            # Normalizing the vessel data
            wss = normalize(vessel.wss)
            pos = normalize(vessel.pos)
            pressure = normalize(vessel.pressure)
            face = normalize(vessel.face.T)

            # Extracting the geometric algebra elements
            elements = [
                TranslationGeometricAlgebra.fromElement(wss),
                PointGeometricAlgebra.fromElement(pos),
                ScalarGeometricAlgebra.fromElement(pressure),
                PlaneGeometricAlgebra.fromElement(face),
            ]

        elements = [
            (
                F.pad(
                    tensor,
                    (0, 0, 0, size_limit - tensor.size(0)),
                    "constant",
                    pad_value,
                )
                if tensor.size(0) < size_limit
                else tensor[:size_limit]
            )
            for tensor in elements
        ]

        # Construct the tensor by stacking geometric algebra elements
        tensor = torch.stack(elements)

        # Append the padded tensor to the padded_data list
        padded_data.append(tensor)

        # Create a mask tensor: 1 for real data, 0 for padded data
        mask = (tensor != pad_value).float()
        mask = mask[..., 0]
        masks.append(mask)

        # Store the label of the vessel
        labels.append(vessel.label.value)

    # Stack the padded tensors, masks, and labels into batch tensors
    collated_batch = VesselBatch(
        data=torch.stack(padded_data),
        mask=torch.stack(masks),
        labels=torch.tensor(labels, dtype=torch.float32),
    )

    return collated_batch


def normalize(data: Tensor, epsilon: float = 1e-6) -> Tensor:
    """
    Normalizes the input tensor using min-max normalization.
    """
    min = data.min()
    max = data.max()
    data = (data - min) / (
        max - min + epsilon
    )  # Adding epsilon to avoid division by zero
    return data
