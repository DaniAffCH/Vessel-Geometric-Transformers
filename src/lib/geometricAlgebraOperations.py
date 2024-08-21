from abc import ABC, abstractmethod
from typing import Any

import clifford as cf
import torch
from typing_extensions import override

from src.lib.geometricAlgebraElements import GeometricAlgebraBase

# TODO: Probabilmente non va bene usare una lib esterna perchè
# le operazioni non sono differenziabili.
# TODO: cambiare almeno le operazioni usate all'interno
# dei layers


class GeometricOperation(ABC):

    # G_{3,0,1} algebra
    layout, blades = cf.Cl(3, 0, 1)

    @staticmethod
    def _generateMultivector(x: torch.Tensor) -> cf._multivector.MultiVector:
        assert x.ndim == 1, "Expected monodimensional vector"
        assert (
            x.shape[0] == GeometricAlgebraBase.GA_size
        ), f"Expected vector to have {GeometricAlgebraBase.GA_size} elements, \
            {x.shape[0]} given"

        basis = GeometricOperation.blades.values()

        return sum(c * b for c, b in zip(x, basis))

    @staticmethod
    @abstractmethod
    def apply(*args: Any) -> torch.Tensor:
        pass


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
