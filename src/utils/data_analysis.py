from enum import Enum
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import display
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.data.datamodule import VesselDataModule
from src.data.vessel import get_feature
from src.utils.definitions import Feature


def check_class_distribution(data: VesselDataModule) -> None:
    train_labels = data.train_set.getLabels()
    val_labels = data.val_set.getLabels()
    test_labels = data.test_set.getLabels()

    combined_labels = np.concatenate([train_labels, val_labels, test_labels])
    subsets = (
        ["train"] * len(train_labels)
        + ["val"] * len(val_labels)
        + ["test"] * len(test_labels)
    )

    df = pd.DataFrame({"label": combined_labels, "subset": subsets})

    df["count"] = df.groupby(["subset", "label"])["label"].transform("count")
    df["total"] = df.groupby("subset")["label"].transform("count")
    df["frequency"] = df["count"] / df["total"]

    df_normalized = df.drop_duplicates(subset=["label", "subset"])

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 6))

    sns.barplot(x="subset", y="frequency", hue="label", data=df_normalized)

    plt.title(
        "Normalized Label Distribution Across Train, Validation, and Test Sets"
    )
    plt.xlabel("Dataset Subset")
    plt.ylabel("Normalized Frequency")
    plt.legend(title="Label", loc="upper right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def data_info(data: VesselDataModule) -> None:
    """
    Function to display information about the dataset.
    """

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
    """
    Function to plot the data in 2D using PCA.
    """

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
