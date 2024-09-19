from enum import Enum
from typing import Optional

import matplotlib.pyplot as plt

# import numpy as np
import torch
from sklearn.decomposition import PCA


def reduce_dim(X: torch.Tensor, n_components: int) -> torch.Tensor:
    """
    Reduces the dimension of the input tensor X to n_components using PCA
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def axis_limits(X: torch.Tensor) -> tuple[float, float, float, float]:
    """
    Returns the axis limits for the input tensor X
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    return x_min, x_max, y_min, y_max


def plot_data(
    X: torch.Tensor,
    y: torch.Tensor,
    classes: Enum,
    clf: torch.nn.Module = None,
    name: Optional[str] = None,
) -> None:
    X_pca = reduce_dim(X, 2)
    x_min, x_max, y_min, y_max = axis_limits(X_pca)
    plt.axis([x_min, x_max, y_min, y_max])

    for type in classes:  # type: ignore
        y = y.int()
        X_pca_type = X_pca[y == type.value]
        plt.scatter(X_pca_type[:, 0], X_pca_type[:, 1], label=type)

    # if clf:
    # generating points in this grid
    # xx, yy = np.meshgrid(
    # np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1)
    # )
    # points = np.c_[xx.ravel(), yy.ravel()]

    # predictions reshaped to match the meshgrid shape
    # Z = clf.predict(pca.inverse_transform(points)).reshape(xx.shape)

    # y_1 = [1] * len(y[y == 1])
    # y_0 = [0] * len(y[y == 0])
    # plt.contourf(xx, yy, Z, alpha=0.15, cmap=my_map)
    # plt.contour(xx,yy, Z, colors='k', linewidths=1)

    if name:
        plt.title(name)
    plt.legend()
    plt.show()
