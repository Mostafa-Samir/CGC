from dataclasses import dataclass

import jax.numpy as jnp

from cgc.kernels.base import BaseKernel, BaseKernelParameters
from cgc.kernels.factory import KernelsFactory
from cgc.utils import KernelParameter

@dataclass
class PeriodicKernelParameters(BaseKernelParameters):
    period: KernelParameter = KernelParameter(1.0, learnable=True)
    width: KernelParameter = KernelParameter(1.0, learnable=True)

@KernelsFactory.register("periodic", PeriodicKernelParameters)
class PeriodicKErnel(BaseKernel):

    def _eval(self, x, y, **params):
        l = params.get("width")
        p = params.get("period")

        diff = x - y
        sin_factor = jnp.sin(jnp.pi * jnp.abs(diff).sum() / p)
        periodic_factor = jnp.exp(-2 * sin_factor * sin_factor / (l * l))

        return periodic_factor