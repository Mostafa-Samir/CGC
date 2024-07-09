from dataclasses import dataclass

import jax.numpy as jnp

from cgc.kernels.base import BaseKernel, BaseKernelParameters
from cgc.kernels.factory import KernelsFactory
from cgc.utils import KernelParameter


@dataclass
class QuadrtaicGaussian(BaseKernelParameters):
    gaussian_scale: KernelParameter = KernelParameter(1.0)
    gaussian_weight: KernelParameter = KernelParameter(1.0)
    quadratic_constant: KernelParameter = KernelParameter(1.0)
    quadratic_weight: KernelParameter = KernelParameter(1.0)


@KernelsFactory.register("quadratic + gaussian", QuadrtaicGaussian)
class QuadraticGaussianKernel(BaseKernel):

    def _eval(self, x, y, **params):
        gaussian_scale = params.get("gaussian_scale")
        gaussian_weight = params.get("gaussian_weight")
        quadratic_constant = params.get("quadratic_constant")
        quadratic_weight = params.get("quadratic_weight")

        diff = (x - y) / gaussian_scale
        gaussian_factor = (gaussian_weight ** 2) * jnp.exp(-0.5 * jnp.dot(diff, diff))
        quadratic_factor = (quadratic_weight ** 2) * jnp.power(jnp.dot(x, y) + quadratic_constant, 2.0)

        return gaussian_factor + quadratic_factor