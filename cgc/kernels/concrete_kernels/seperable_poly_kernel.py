from dataclasses import dataclass

import jax.numpy as jnp

from cgc.kernels.base import BaseKernel, BaseKernelParameters
from cgc.kernels.factory import KernelsFactory
from cgc.utils import KernelParameter


@dataclass
class SeprablePolynomialParameters(BaseKernelParameters):
    constant: KernelParameter = KernelParameter(1.0, learnable=False)
    exponent: KernelParameter = KernelParameter(3.0, learnable=False)


@KernelsFactory.register("separable-polynomial", SeprablePolynomialParameters)
class PolynoSeprablePolynomialmialKernel(BaseKernel):


    def _poly_eval(self, x, y, constant, exponent):
        return jnp.power(jnp.dot(x, y) + constant * constant, exponent)

    def _eval(self, x, y, **params):
        
        p_start, p_end = 0, 1 if x.size < 4 else 2
        q_start = 1 if x.size < 4 else 2

        constant = params.get("constant")
        exponent = params.get("exponent")
        
        return self._poly_eval(x[p_start: p_end], y[p_start:p_end], constant, exponent) + self._poly_eval(x[q_start:], y[q_start:], constant, exponent)