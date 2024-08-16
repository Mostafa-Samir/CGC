from dataclasses import dataclass

import jax.numpy as jnp

from cgc.kernels.base import BaseKernel, BaseKernelParameters
from cgc.kernels.factory import KernelsFactory
from cgc.utils import KernelParameter


@dataclass
class PolynomialParameters(BaseKernelParameters):
    constant: KernelParameter = KernelParameter(1.0, learnable=True)
    exponent: KernelParameter = KernelParameter(1.0, learnable=True)


@KernelsFactory.register("polynomial", PolynomialParameters)
class PolynomialKernel(BaseKernel):

    def _eval(self, x, y, **params):

        constant = params.get("constant")
        exponent = params.get("exponent")
        return jnp.power(jnp.dot(x, y) + constant * constant, exponent)