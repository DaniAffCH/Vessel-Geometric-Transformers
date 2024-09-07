from abc import ABC, abstractmethod

import torch
from typing_extensions import override


class GeometricAlgebraBase(ABC):
    """
    Base class for geometric algebra operations.
    Operations return a 16-element torch tensor representing a multivector.
    The basis combination of the multivector
    are defined in the following order:

    1 e0 e1 e2 e3 e01 e02 e03 e12 e13 e23 e012 e013 e023 e123 e01234
    """

    numBasis: int = 4  # Number of basis vectors
    GA_size: int = 2**numBasis  # Size of the geometric algebra space

    @staticmethod
    def _getEmptyMultivector(numElements: int) -> torch.Tensor:
        """
        Create an empty multivector with all elements initialized to zero.

        Args:
            numElements (int): Number of elements (rows) in the tensor.

        Returns:
            torch.Tensor: A tensor of shape
                          (numElements, GA_size) filled with zeros.
        """
        return torch.zeros((numElements, GeometricAlgebraBase.GA_size))

    @classmethod
    def fromElement(cls, element: torch.Tensor) -> torch.Tensor:
        """
        Template method to create a multivector from a geometric element.

        Args:
            element (torch.Tensor): The input tensor representing
                                    the geometric element.

        Returns:
            torch.Tensor: The resulting multivector.
        """
        cls._validate_element(element)
        v = cls._getEmptyMultivector(element.shape[0])
        cls._fill_multivector(v, element)
        return v

    @staticmethod
    @abstractmethod
    def _validate_element(element: torch.Tensor) -> None:
        """
        Abstract method to validate the input geometric element.

        Args:
            element (torch.Tensor): The input tensor to validate.
        """
        pass

    @staticmethod
    @abstractmethod
    def _fill_multivector(
        v: torch.Tensor, element: torch.Tensor
    ) -> torch.Tensor:
        """
        Abstract method to fill the multivector
        with the geometric element data.

        Args:
            v (torch.Tensor): The empty multivector tensor to fill.
            element (torch.Tensor): The input tensor containing
                                    the geometric element data.

        Returns:
            torch.Tensor: The filled multivector.
        """
        pass


class PointGeometricAlgebra(GeometricAlgebraBase):
    """
    Geometric algebra operations specific to points.
    """

    @override
    @staticmethod
    def _validate_element(element: torch.Tensor) -> None:
        """
        Validate the input tensor for points.

        Args:
            element (torch.Tensor): The input tensor representing the point.
        """
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
        """
        Fill the multivector with point data.

        Args:
            v (torch.Tensor): The empty multivector tensor to fill.
            element (torch.Tensor): The input tensor containing the point data.

        Returns:
            torch.Tensor: The filled multivector.
        """
        v[:, 11:14] = element
        v[:, 14] = 1
        return v


class TranslationGeometricAlgebra(GeometricAlgebraBase):
    """
    Geometric algebra operations specific to translations.
    """

    @override
    @staticmethod
    def _validate_element(element: torch.Tensor) -> None:
        """
        Validate the input tensor for translations.

        Args:
            element (torch.Tensor): The input tensor representing
                                    the translation.
        """
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
        """
        Fill the multivector with translation data.

        Args:
            v (torch.Tensor): The empty multivector tensor to fill.
            element (torch.Tensor): The input tensor containing
                                    the translation data.

        Returns:
            torch.Tensor: The filled multivector.
        """
        v[:, 0] = 1
        v[:, 5:8] = element / 2
        return v


class ScalarGeometricAlgebra(GeometricAlgebraBase):
    """
    Geometric algebra operations specific to scalars.
    """

    @override
    @staticmethod
    def _validate_element(element: torch.Tensor) -> None:
        """
        Validate the input tensor for scalars.

        Args:
            element (torch.Tensor): The input tensor representing the scalar.
        """
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
        """
        Fill the multivector with scalar data.

        Args:
            v (torch.Tensor): The empty multivector tensor to fill.
            element (torch.Tensor): The input tensor containing
                                    the scalar data.

        Returns:
            torch.Tensor: The filled multivector.
        """
        v[:, 0] = element
        return v


class PlaneGeometricAlgebra(GeometricAlgebraBase):
    """
    Geometric algebra operations specific to planes.
    """

    @override
    @staticmethod
    def _validate_element(element: torch.Tensor) -> None:
        """
        Validate the input tensor for planes.

        Args:
            element (torch.Tensor): The input tensor representing the plane.
        """
        coordExpected = 3
        dimExpected = 2
        assert (
            element.ndim == dimExpected
        ), f"Expected an element with {dimExpected} dimensions,\
             {element.ndim} given"
        assert (
            element.shape[1] == coordExpected
        ), f"Expected a plane to have {coordExpected} coordinates,\
             {element.shape[1]} given"

    @override
    @staticmethod
    def _fill_multivector(
        v: torch.Tensor, element: torch.Tensor
    ) -> torch.Tensor:
        """
        Fill the multivector with plane data.
        It assumes there is no origin shift

        Args:
            v (torch.Tensor): The empty multivector tensor to fill.
            element (torch.Tensor): The input tensor containing the plane data.

        Returns:
            torch.Tensor: The filled multivector.
        """
        v[:, 2:5] = element
        return v


class ReflectionGeometricAlgebra(GeometricAlgebraBase):
    """
    Geometric algebra operations specific to planes.
    """

    @override
    @staticmethod
    def _validate_element(element: torch.Tensor) -> None:
        """
        Validate the input tensor for reflection.

        Args:
            element (torch.Tensor): The input tensor representing
                                    the reflection.
        """
        coordExpected = 4
        dimExpected = 2
        assert (
            element.ndim == dimExpected
        ), f"Expected an element with {dimExpected} dimensions,\
             {element.ndim} given"
        assert (
            element.shape[1] == coordExpected
        ), f"Expected a plane to have {coordExpected} coordinates,\
             {element.shape[1]} given"

    @override
    @staticmethod
    def _fill_multivector(
        v: torch.Tensor, element: torch.Tensor
    ) -> torch.Tensor:
        """
        Fill the multivector with reflection data.

        Args:
            v (torch.Tensor): The empty multivector tensor to fill.
            element (torch.Tensor): The input tensor containing
                                    the reflection data.

        Returns:
            torch.Tensor: The filled multivector.
        """
        v[:, 1:5] = element
        return v
