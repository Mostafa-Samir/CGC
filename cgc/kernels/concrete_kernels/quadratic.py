from dataclasses import dataclass

import jax.numpy as jnp

from cgc.kernels.base import BaseKernel, BaseKernelParameters
from cgc.kernels.factory import KernelsFactory
from cgc.utils import KernelParameter


@dataclass
class QuadraticParameters(BaseKernelParameters):
    constant: KernelParameter = KernelParameter(1.0, learnable=True)


@KernelsFactory.register("quadratic", QuadraticParameters)
class QuadraticKernel(BaseKernel):

    def _eval(self, x, y, **params):

        constant = params.get("constant")
        return (jnp.dot(x, y) + constant) * (jnp.dot(x, y) + constant)