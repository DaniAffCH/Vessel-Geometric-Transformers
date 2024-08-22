from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import torch
from torch._prims_common import DeviceLikeType
from typing_extensions import override

# TODO: Probabilmente non va bene usare una lib esterna perchè
# le operazioni non sono differenziabili.
# TODO: cambiare almeno le operazioni usate all'interno
# dei layers


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
    def getEquiLinBasis(
        device: DeviceLikeType = torch.device("cpu"),
    ) -> torch.Tensor:

        # Tuples represent range
        # Grade projection
        diag_range = [(0, 1), (1, 5), (5, 11), (11, 15), (15, 16)]

        # Tuples represent indices
        # homogeneous basis vector e0
        additional_idx = [
            [(1, 0)],
            [(5, 2), (6, 3), (7, 4)],
            [(11, 8), (12, 9), (13, 10)],
            [(15, 14)],
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
        op = Blade.getEquiLinBasis(x)
        op = op[grade]
        op = torch.sum(op, dim=1)
        return op * x


class GeometricProduct(GeometricOperation):

    # This function always retrieve the same matrix.
    # Using a lru cache to avoid repeated computations
    @lru_cache(maxsize=1)
    @staticmethod
    def getBiLinBasis(
        device: DeviceLikeType = torch.device("cpu"),
    ) -> torch.Tensor:
        # This file of precomputed basis was taken from
        # https://github.com/Qualcomm-AI-research/geometric-algebra-transformer/blob/main/gatr/primitives/data/geometric_product.pt
        basis = torch.load("data/geometric_product.pt")
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
    def getBiLinBasis(
        device: DeviceLikeType = torch.device("cpu"),
    ) -> torch.Tensor:
        # This file of precomputed basis was taken from
        # https://github.com/Qualcomm-AI-research/geometric-algebra-transformer/blob/main/gatr/primitives/data/outer_product.pt
        basis = torch.load("data/outer_product.pt")
        basis = basis.to_dense()
        basis = basis.to(device=device, dtype=torch.float32)
        return basis


"""
class GeometricProduct(GeometricOperation):
    @override
    @staticmethod
    def apply(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_mv = GeometricProduct._generateMultivector(x)
        y_mv = GeometricProduct._generateMultivector(y)

        return torch.Tensor(x_mv * y_mv)


class InnerProduct(GeometricOperation):
    @override
    @staticmethod
    def apply(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_mv = InnerProduct._generateMultivector(x)
        y_mv = InnerProduct._generateMultivector(y)

        return torch.Tensor(x_mv | y_mv)


class OuterProduct(GeometricOperation):
    @override
    @staticmethod
    def apply(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_mv = OuterProduct._generateMultivector(x)
        y_mv = OuterProduct._generateMultivector(y)

        return torch.Tensor(x_mv ^ y_mv)


class Dual(GeometricOperation):
    @override
    @staticmethod
    def apply(x: torch.Tensor) -> torch.Tensor:
        x_mv = Dual._generateMultivector(x)

        return torch.Tensor(x_mv.dual())


class Blade(GeometricOperation):
    @override
    @staticmethod
    def apply(x: torch.Tensor, grade: int) -> torch.Tensor:
        x_mv = Blade._generateMultivector(x)
        return torch.Tensor(x_mv(grade))


class SandwichProduct(GeometricOperation):
    @override
    @staticmethod
    def apply(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError()
"""
