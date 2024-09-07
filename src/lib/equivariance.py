import random
from typing import Any, Optional

import clifford as cf
import numpy as np
import torch
from clifford import pga

from . import ReflectionGeometricAlgebra


class ReflectionMultivector(cf.MultiVector):  # type: ignore
    def __init__(
        self,
        layout: Any,
        value: Optional[Any] = None,
        string: Optional[str] = None,
        *,
        dtype: np.dtype = np.float64
    ) -> None:
        super().__init__(layout, value, string, dtype=dtype)

        assert layout == pga.layout, "Expected a Projective G(3,0,1) algebra."

    """
        Just a wrapper for computing magnitude in G(3,0,1) algebra.
        Unlike cf.MultiVector.__abs__ this function does not require
        a cf.MultiVector instance
    """

    @staticmethod
    def magnitude(v: torch.Tensor) -> torch.Tensor:
        # M^2 in G(3,0,1)

        m2 = pga.layout.gmt_func(
            pga.layout.adjoint_func(v.numpy()), v.numpy()
        )[0]

        return np.sqrt(abs(m2))

    """
        You can see this static method as an "alternative" constructor
        returning an instance of cf.MultiVector containing
        a reflection operator with unitary magnitude (versor).
    """

    @staticmethod
    def randomReflectionVersor() -> "ReflectionMultivector":
        mv = ReflectionGeometricAlgebra.fromElement(torch.rand(4).unsqueeze(0))
        mv = mv[0]

        # ensure it has unitary magnitude
        mv /= ReflectionMultivector.magnitude(mv)

        return ReflectionMultivector(pga.layout, mv)

    def isPin(self) -> bool:
        return bool(self == self.odd)

    def isSpin(self) -> bool:
        return bool(self == self.even)


def generateRandomReflection(
    mulMax: int = 10, epsCheck: float = 1e-5
) -> "ReflectionMultivector":
    nMul = random.randint(1, mulMax)
    mv = ReflectionMultivector.randomReflectionVersor()

    for _ in range(nMul):
        mv = mv * ReflectionMultivector.randomReflectionVersor()

    # The result must be a versor i.e. magnitude = 1
    assert abs(abs(mv) - 1.0) < epsCheck

    return mv


def sandwich(x: cf.MultiVector, u: "ReflectionMultivector") -> cf.MultiVector:
    if u.isPin():
        x = x.gradeInvol()

    return u * x * u.shirokov_inverse()


def torchSandwich(x: torch.Tensor, u: "ReflectionMultivector") -> torch.Tensor:
    x_mv = cf.MultiVector(pga.layout, x)
    return torch.tensor(
        sandwich(x_mv, u).value, dtype=x.dtype, device=x.device
    )


@torch.no_grad()  # type: ignore
def checkEquivariance(
    layer: torch.nn.Module,
    x: torch.Tensor,
    numChecks: int = 100,
    tolerance: float = 1e-5,
    *args,
    **kwargs
) -> bool:
    def doubleUnsqueezeToDev(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0).unsqueeze(0).to(device)

    def doubleSqueezeToCpu(t: torch.Tensor) -> torch.Tensor:
        return t.squeeze(0).squeeze(0).cpu()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    layer = layer.to(device)
    x = x.to(device)

    args = tuple(
        doubleUnsqueezeToDev(arg) if isinstance(arg, torch.Tensor) else arg
        for arg in args
    )
    kwargs = {
        k: doubleUnsqueezeToDev(v) if isinstance(v, torch.Tensor) else v
        for k, v in kwargs.items()
    }

    isEquiv = True

    for _ in range(numChecks):
        u = generateRandomReflection()

        # f(p_u(x))
        x = doubleSqueezeToCpu(x)
        pu = torchSandwich(x, u)
        pu = doubleUnsqueezeToDev(pu)
        fpu = layer(pu, *args, **kwargs)
        fpu = doubleSqueezeToCpu(fpu)

        # Hack for bilinear layer requiring an even number of features
        if fpu.ndim > 1:
            fpu = fpu[0]

        # p_u(f(x))
        x = doubleUnsqueezeToDev(x)
        f = layer(x, *args, **kwargs)
        f = doubleSqueezeToCpu(f)

        # Hack for bilinear layer requiring an even number of features
        if f.ndim > 1:
            f = f[0]

        puf = torchSandwich(f, u)
        x = doubleSqueezeToCpu(x)

        # f(p_u(x)) = p_u(f(x))
        isEquiv &= torch.allclose(fpu, puf, atol=tolerance)

    return isEquiv
