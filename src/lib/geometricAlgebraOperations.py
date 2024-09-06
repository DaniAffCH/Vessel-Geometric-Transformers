from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from typing_extensions import override


class GeometricOperation(ABC):

    @staticmethod
    @abstractmethod
    def apply(*args: Any) -> torch.Tensor:
        pass


class Blade(GeometricOperation):

    # This function always compute the same matrix.
    # Using a lru cache to avoid repeated computations
    @lru_cache(maxsize=1)
    @staticmethod
    def getEquiLinBasis() -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tuples represent range
        # Grade projection (i.e. diagonal)
        diag_range = [(0, 1), (1, 5), (5, 11), (11, 15), (15, 16)]

        # Tuples represent indices
        # homogeneous basis vector * e0
        # [e0 | 0 e0e1 e0e2 e0e3 | 0 0 0 e012 e013 e023 | 0 0 0 e123 | 0]
        additional_idx = [
            [(1, 0)],  # e0
            [(5, 2), (6, 3), (7, 4)],  # e0e1 e0e2 e0e3
            [(11, 8), (12, 9), (13, 10)],  # e012 e013 e023
            [(15, 14)],  # e123
        ]

        b = []

        for start, end in diag_range:
            m = torch.zeros((16, 16))
            m[start:end, start:end] = torch.eye(end - start)
            b.append(m)

        for el in additional_idx:
            m = torch.zeros((16, 16))
            indices = torch.tensor(el).t()
            m[indices[0], indices[1]] = 1
            b.append(m)

        basis = torch.stack(b)

        return basis.to(device=device, dtype=torch.float32)

    @override
    @staticmethod
    def apply(x: torch.Tensor, grade: int) -> torch.Tensor:
        assert grade < 4, (
            f"grade {grade} out of range for algebra "
            + "G_{3,0,1} (max grade 3)"
        )
        op = Blade.getEquiLinBasis()
        op = op[grade]
        op = torch.sum(op, dim=1)
        return op * x


class GeometricProduct(GeometricOperation):

    # This function always retrieve the same matrix.
    # Using a lru cache to avoid repeated computations
    @lru_cache(maxsize=1)
    @staticmethod
    def getBiLinBasis() -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # This file of precomputed basis was taken from
        # https://github.com/Qualcomm-AI-research/geometric-algebra-transformer/blob/main/gatr/primitives/data/geometric_product.pt
        BILIN_PATH = (
            Path(__file__).resolve().parent
            / "precomputed/geometric_product.pt"
        )
        basis = torch.load(BILIN_PATH)
        basis = basis.to_dense()
        basis = basis.to(device=device, dtype=torch.float32)
        return basis

    @override
    @staticmethod
    def apply(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        basis = GeometricProduct.getBiLinBasis()
        return torch.einsum("ijk, ...j, ...k -> ...i", basis, x, y)


class OuterProduct(GeometricProduct):
    # This function always retrieve the same matrix.
    # Using a lru cache to avoid repeated computations
    @lru_cache(maxsize=1)
    @staticmethod
    def getBiLinBasis() -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # This file of precomputed basis was taken from
        # https://github.com/Qualcomm-AI-research/geometric-algebra-transformer/blob/main/gatr/primitives/data/outer_product.pt
        BILIN_PATH = (
            Path(__file__).resolve().parent / "precomputed/outer_product.pt"
        )
        basis = torch.load(BILIN_PATH)
        basis = basis.to_dense()
        basis = basis.to(device=device, dtype=torch.float32)
        return basis


class InnerProduct(GeometricOperation):
    @staticmethod
    def getBasisIndices() -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # keep only elements which don't contain basis e0
        return torch.tensor(
            [0, 2, 3, 4, 8, 9, 10, 14], dtype=torch.int, device=device
        )

    @override
    @staticmethod
    def apply(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        idxs = InnerProduct.getBasisIndices()
        x = x[..., idxs]
        y = y[..., idxs]

        # dot product
        return torch.sum(x * y, dim=-1)


class Dual(GeometricOperation):
    @override
    @staticmethod
    def apply(x: torch.Tensor) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Needed to fix orintation
        # From table 4 of https://geometricalgebra.org/downloads/PGA4CS.pdf
        factors = torch.tensor(
            [1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1],
            dtype=torch.int,
            device=device,
        )
        return factors * x.flip(-1)


class Join(GeometricOperation):
    @override
    @staticmethod
    def apply(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # (x* ∧ y*)*
        xStar = Dual.apply(x)
        yStar = Dual.apply(y)
        product = OuterProduct.apply(xStar, yStar)
        join = Dual.apply(product)
        return join


class EquivariantJoin(GeometricOperation):
    @override
    @staticmethod
    def apply(
        x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        # z0123 (x* ∧ y*)*
        return z[..., [-1]] * Join.apply(x, y)
