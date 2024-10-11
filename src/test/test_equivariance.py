import unittest

import torch

from src.lib.equivariance import checkEquivariance
from src.models.layers.geometric import (
    EquiLinearLayer,
    EquiNormLayer,
    GatedGELU,
    GeometricBilinearLayer,
)
from src.models.layers.geometric.geometricAttention import (
    GeometricAttentionLayer,
)


class TestEquivariance(unittest.TestCase):
    """
    Using unittests to check whether the Geometric layers are equivariant
    """

    INPUT_DATA: torch.Tensor = torch.tensor([])

    def setUp(self) -> None:
        self.num_checks = 100
        self.tolerance = 1e-5
        self.inputs = self.INPUT_DATA
        self.num_inputs = self.INPUT_DATA.shape[0]
        assert self.num_inputs > 0, "expected at least one sample"

    @classmethod
    def setTestData(cls, inputs: torch.Tensor) -> None:
        cls.inputs = inputs
        cls.num_inputs = inputs.shape[0]

    def print_ok(self, layer: torch.nn.Module) -> None:
        print(
            f"\n✅ {layer.__class__.__name__} passed all equivariance tests \
              with a tolerance of {self.tolerance}.\
              [{self.num_inputs} inputs, {self.num_checks} checks each]"
        )

    def test_equilinear_layer(self) -> None:
        """
        Tests the equivariance of the EquiLinearLayer.
        """
        test_layer = EquiLinearLayer(1, 1)

        for i, mv in enumerate(self.inputs):

            res = checkEquivariance(test_layer, mv, self.num_checks)
            self.assertTrue(
                res,
                f"{test_layer.__class__.__name__} failed the equivariance \
                    test for input {i + 1}",
            )

        self.print_ok(test_layer)

    def test_bilinear_layer(self) -> None:
        """
        Tests the equivariance of the GeometricBilinearLayer.
        """
        test_layer = GeometricBilinearLayer(1, 2)

        for i, mv in enumerate(self.inputs):
            res = checkEquivariance(
                test_layer, mv, self.num_checks, self.tolerance, reference=mv
            )
            self.assertTrue(
                res,
                f"{test_layer.__class__.__name__} failed the equivariance \
                    test for input {i + 1}",
            )

        self.print_ok(test_layer)

    def test_gatedgelu_layer(self) -> None:
        """
        Tests the equivariance of the GatedGELU.
        """
        test_layer = GatedGELU()

        for i, mv in enumerate(self.inputs):
            res = checkEquivariance(test_layer, mv, self.num_checks)
            self.assertTrue(
                res,
                f"{test_layer.__class__.__name__} failed the equivariance \
                    test for input {i + 1}",
            )

        self.print_ok(test_layer)

    def test_equinorm_layer(self) -> None:
        """
        Tests the equivariance of the EquiNormLayer.
        """
        test_layer = EquiNormLayer()

        for i, mv in enumerate(self.inputs):
            res = checkEquivariance(test_layer, mv, self.num_checks)
            self.assertTrue(
                res,
                f"{test_layer.__class__.__name__} failed the equivariance \
                    test for input {i + 1}",
            )

        self.print_ok(test_layer)

    def test_geomattention_layer(self) -> None:
        """
        Tests the equivariance of the GeometricAttentionLayer.
        """
        test_layer = GeometricAttentionLayer(1, 1)

        for i, mv in enumerate(self.inputs):
            res = checkEquivariance(
                test_layer,
                mv,
                self.num_checks,
                self.tolerance,
                key=mv,
                value=mv,
            )
            self.assertTrue(
                res,
                f"{test_layer.__class__.__name__} failed the equivariance \
                    test for input {i + 1}",
            )

        self.print_ok(test_layer)
