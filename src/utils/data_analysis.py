from enum import Enum
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import display
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.data.datamodule import VesselDataModule
from src.data.vessel import get_feature
from src.utils.definitions import Feature


def data_info(data: VesselDataModule) -> None:
    print(f"Train size: {len(data.train_set)}")
    print(f"Val size: {len(data.val_set)}")
    print(f"Test size: {len(data.test_set)}")
    print(f"One Sample: {data.train_set[2]}")

    dataset = data.data

    wss = []
    pos = []
    face = []
    pressure = []

    for vessel in tqdm(dataset):
        wss.append(get_feature(vessel, Feature.WSS).shape[0])
        pos.append(get_feature(vessel, Feature.POS).shape[0])
        face.append(get_feature(vessel, Feature.FACE).shape[1])
        pressure.append(get_feature(vessel, Feature.PRESSURE).shape[0])

    shapes = {"WSS": wss, "POS": pos, "FACE": face, "PRESSURE": pressure}

    df = pd.DataFrame(shapes)

    shapes_df = pd.DataFrame(
        {
            "Mean": df.mean(),
            "Median": df.median(),
            "Std": df.std(),
            "Min": df.min(),
            "Max": df.max(),
        }
    )

    display(shapes_df)


def plot_data(
    X: torch.Tensor,
    y: torch.Tensor,
    classes: Enum,
    name: Optional[str] = None,
) -> None:
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    plt.axis([x_min, x_max, y_min, y_max])

    for type in classes:  # type: ignore
        y = y.int()
        X_pca_type = X[y == type.value]
        plt.scatter(X_pca_type[:, 0], X_pca_type[:, 1], label=type)

    if name:
        plt.title(name)
    plt.legend()
    plt.show()
