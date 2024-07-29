from abc import ABC, abstractmethod

import torch
from typing_extensions import override


class GeometricAlgebraBase(ABC):
    """
    Geometric Algebra operations return a 16-element torch tensor
    representing a multivector.
    The basis combination of the multivector are defined
    in the following order:

    1 e0 e1 e2 e3 e01 e02 e03 e12 e13 e23 e012 e013 e023 e123 e01234
    """

    numBasis: int = 4
    GA_size: int = 2**numBasis

    @staticmethod
    def _getEmptyMultivector(numElements: int) -> torch.Tensor:
        return torch.zeros((numElements, GeometricAlgebraBase.GA_size))

    @classmethod
    def fromElement(cls, element: torch.Tensor) -> torch.Tensor:
        cls._validate_element(element)
        v = cls._getEmptyMultivector(element.shape[0])
        cls._fill_multivector(v, element)
        return v

    @staticmethod
    @abstractmethod
    def _validate_element(element: torch.Tensor) -> None:
        pass

    @staticmethod
    @abstractmethod
    def _fill_multivector(
        v: torch.Tensor, element: torch.Tensor
    ) -> torch.Tensor:
        pass


class PointGeometricAlgebra(GeometricAlgebraBase):
    @override
    @staticmethod
    def _validate_element(element: torch.Tensor) -> None:
        coordExpected = 3
        dimExpected = 2
        assert (
            element.ndim == dimExpected
        ), f"Expected an element with {dimExpected} dimensions,\
             {element.ndim} given"
        assert (
            element.shape[1] == coordExpected
        ), f"Expected a point to have {coordExpected} coordinates,\
             {element.shape[1]} given"

    @override
    @staticmethod
    def _fill_multivector(
        v: torch.Tensor, element: torch.Tensor
    ) -> torch.Tensor:
        v[:, 11:14] = element
        v[:, 14] = 1
        return v


class TranslationGeometricAlgebra(GeometricAlgebraBase):
    @override
    @staticmethod
    def _validate_element(element: torch.Tensor) -> None:
        coordExpected = 3
        dimExpected = 2
        assert (
            element.ndim == dimExpected
        ), f"Expected an element with {dimExpected} dimensions,\
             {element.ndim} given"
        assert (
            element.shape[1] == coordExpected
        ), f"Expected a translation to have {coordExpected} coordinates,\
             {element.shape[1]} given"

    @override
    @staticmethod
    def _fill_multivector(
        v: torch.Tensor, element: torch.Tensor
    ) -> torch.Tensor:
        v[:, 0] = 1
        v[:, 5:8] = element / 2
        return v


class ScalarGeometricAlgebra(GeometricAlgebraBase):
    @override
    @staticmethod
    def _validate_element(element: torch.Tensor) -> None:
        dimExpected = 1
        assert (
            element.ndim == dimExpected
        ), f"Expected an element with {dimExpected} dimension,\
             {element.ndim} given"

    @override
    @staticmethod
    def _fill_multivector(
        v: torch.Tensor, element: torch.Tensor
    ) -> torch.Tensor:
        v[:, 0] = element
        return v


# TODO: Non ho capito perchè l'attributo face ha shape [3, 38430]
# mentre un plane si aspetta 4 parametri
class PlaneGeometricAlgebra(GeometricAlgebraBase):
    @override
    @staticmethod
    def _validate_element(element: torch.Tensor) -> None:
        raise NotImplementedError()

    @override
    @staticmethod
    def _fill_multivector(
        v: torch.Tensor, element: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()
