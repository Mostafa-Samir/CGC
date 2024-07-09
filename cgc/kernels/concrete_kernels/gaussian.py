from dataclasses import dataclass

import jax.numpy as jnp

from cgc.kernels.base import BaseKernel, BaseKernelParameters
from cgc.kernels.factory import KernelsFactory
from cgc.utils import KernelParameter


@dataclass
class GaussianKernelParameters(BaseKernelParameters):
    scale: KernelParameter = KernelParameter(1.0, learnable=True)


@KernelsFactory.register("gaussian", GaussianKernelParameters)
class GaussianKernel(BaseKernel):

    def _eval(self, x, y, **params):
        gamma = params.get("scale")
        diff = (x - y) / gamma
        return jnp.exp(-0.5 * jnp.dot(diff, diff))