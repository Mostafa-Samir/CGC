from dataclasses import dataclass

import jax.numpy as jnp

from cgc.kernels.base import BaseKernel, BaseKernelParameters
from cgc.kernels.factory import KernelsFactory
from cgc.utils import KernelParameter


@dataclass
class BigKernelParameters(BaseKernelParameters):
    gaussian_width: KernelParameter = KernelParameter(1.0, learnable=True)
    gaussian_weight: KernelParameter = KernelParameter(1.0, learnable=True, weight=True)
    quadratic_constant: KernelParameter = KernelParameter(1.0, learnable=True)
    quadratic_weight: KernelParameter = KernelParameter(1.0, learnable=True, weight=True)
    periodic_periodicity: KernelParameter = KernelParameter(1.0, learnable=True)
    periodic_scale: KernelParameter = KernelParameter(1.0, learnable=True)
    #poly_constant: KernelParameter = KernelParameter(1.0, learnable=True)
    #poly_exponant: KernelParameter = KernelParameter(1.0, learnable=True)
    #poly_weight: KernelParameter = KernelParameter(1.0, learnable=True, weight=True)
    #cubic_constant: KernelParameter = KernelParameter(1.0, learnable=True)
    #cubic_weight: KernelParameter = KernelParameter(1.0, learnable=True, weight=True)
    #locally_periodic_periodicity: KernelParameter = KernelParameter(1.0, learnable=True)
    #locally_periodic_scale: KernelParameter = KernelParameter(1.0, learnable=True)
    #locally_periodic_locality_scale: KernelParameter = KernelParameter(1.0, learnable=True)
    periodic_weight: KernelParameter = KernelParameter(1.0, learnable=True, weight=True)
    #locally_periodic_weight: KernelParameter = KernelParameter(1.0, learnable=True)


@KernelsFactory.register("big-kernel", BigKernelParameters)
class BigKernel(BaseKernel):

    def _gaussian_eval(self, x, y, gamma):
        diff = (x - y) / gamma
        return jnp.exp(-0.5 * jnp.dot(diff, diff))
    
    def _quadratic_eval(self, x, y, c):
        return jnp.power(jnp.dot(x, y) + c, 2)
    
    def _cubic_eval(self, x, y, c):
        return jnp.power(jnp.dot(x, y), 3)
    
    def _periodic_eval(self, x, y, p, l):
        diff = x - y
        sin_factor = jnp.sin(jnp.pi * jnp.square(diff).sum() / p)
        return jnp.exp(-2 * sin_factor * sin_factor / (l * l))
    
    def _locally_periodic_eval(self, x, y, p, l, ll):
        #diff = x - y
        #sin_factor = jnp.sin(jnp.pi * jnp.abs(diff).sum() / p)
        #return jnp.exp(-2 * sin_factor * sin_factor / (l * l)) * jnp.exp(-0.5 * jnp.dot(diff / ll, diff / ll))
        return jnp.exp(-2 * jnp.sin())
    
    def _poly_eval(self, x, y, c, e):
        return jnp.power(jnp.dot(x, y) + c, e)

    def _eval(self, x, y, **params):
        gaussian_weight = params.get("gaussian_weight")
        quadratic_weight = params.get("quadratic_weight")
        gaussian_width = params.get("gaussian_width")
        quadratic_constant = params.get("quadratic_constant")
        periodic_periodicity = params.get("periodic_periodicity")
        periodic_scale = params.get("periodic_scale")
        periodic_weight = params.get("periodic_weight")
        #locally_periodic_periodicity = params.get("locally_periodic_periodicity")
        #locally_periodic_scale = params.get("locally_periodic_scale")
        #locally_periodic_locality_scale = params.get("locally_periodic_locality_scale")
        #periodic_weight = params.get("periodic_weight")
        #locally_periodic_weight = params.get("locally_periodic_weight")
        #poly_constant = params.get("poly_constant")
        #poly_weight = params.get("poly_weight")
        #poly_exponant = params.get("poly_exponant")


        return (gaussian_weight ** 2) * self._gaussian_eval(x, y, gaussian_width) + (quadratic_weight ** 2) * self._quadratic_eval(x, y, quadratic_constant)

        #return #jnp.power(jnp.dot(x, y) + poly_constant, 2 * poly_exponant) \
                #+(gaussian_weight ** 2) * self._gaussian_eval(x, y, gaussian_width) \
                #+ (periodic_weight ** 2) * self._periodic_eval(x, y, periodic_periodicity, periodic_scale) \
                #+ (locally_periodic_weight ** 2) * self._locally_periodic_eval(x, y, locally_periodic_periodicity, locally_periodic_scale, locally_periodic_locality_scale)
        