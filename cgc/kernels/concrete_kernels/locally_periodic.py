from dataclasses import dataclass

import jax.numpy as jnp

from cgc.kernels.base import BaseKernel, BaseKernelParameters
from cgc.kernels.factory import KernelsFactory
from cgc.utils import KernelParameter

@dataclass
class LocallyPeriodicKernelParameters(BaseKernelParameters):
    period: KernelParameter = KernelParameter(1.0, learnable=True)
    period_scale: KernelParameter = KernelParameter(1.0, learnable=True)
    locality_scale: KernelParameter = KernelParameter(1.0, learnable=True)

@KernelsFactory.register("locally-periodic", LocallyPeriodicKernelParameters)
class LocallyPeriodicKErnel(BaseKernel):

    def _eval(self, x, y, **params):
        p = params.get("period")
        ps = params.get("period_scale")
        ls = params.get("locality_scale")

        diff = x - y
        sin_factor = (jnp.sin(jnp.pi * (x - y) / p) / ps) ** 2
        scaled_diff = diff / ls
        periodic_factor = jnp.exp(-2 * jnp.sum(sin_factor))
        locality_factor = jnp.exp(-0.5 * jnp.dot(scaled_diff, scaled_diff))

        return periodic_factor * locality_factor