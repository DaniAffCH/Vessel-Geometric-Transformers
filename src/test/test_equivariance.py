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

    def setUp(self) -> None:
        self.num_inputs = 5
        self.num_checks = 100
        self.tolerance = 1e-5
        self.inputs = [torch.randn(16) for _ in range(self.num_inputs)]
        self.references = [torch.rand_like(i) for i in self.inputs]

    def print_ok(self, layer: torch.nn.Module) -> None:
        print(
            f"\n✅ {layer.__class__.__name__} passed all equivariance tests.\
                  [{self.num_inputs} inputs, {self.num_checks} checks each]"
        )

    def test_equilinear_layer(self) -> None:
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
        test_layer = GeometricBilinearLayer(1, 2)

        for i, (mv, r) in enumerate(zip(self.inputs, self.references)):
            res = checkEquivariance(
                test_layer, mv, self.num_checks, self.tolerance, reference=r
            )
            self.assertTrue(
                res,
                f"{test_layer.__class__.__name__} failed the equivariance \
                    test for input {i + 1}",
            )

        self.print_ok(test_layer)

    def test_gatedgelu_layer(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
