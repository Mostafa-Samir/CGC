from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Union

from cgc.kernels import RBFKernel

import jax.numpy as jnp
import jax

class CallableInterface(ABC):

    @abstractmethod
    def __call__(self, X: Any) -> Any:
        raise NotImplementedError()

class Observable(CallableInterface):
    
    def __init__(self, index: int) -> None:
        self.index = index

    def __call__(self, Z):
        return jnp.asarray(Z[:, self.index])

class Function(CallableInterface):

    def __init__(self, parameter: CallableInterface):
        self.parameter = parameter

    @abstractmethod
    def evaluate(self, Z):
        raise NotImplementedError()

    def __call__(self, Z):
        return self.evaluate(self.parameter(Z))


class Aggregator(CallableInterface):

    def __init__(self, parameters: List[CallableInterface]) -> None:
        self.parameters = parameters

    def __call__(self, Z):
        return tuple(parameter(Z) for parameter in self.parameters)


class UnknownFunction(Function):

    def __init__(self, parameter, target_index, kernel="rbf", alpha=1.0, gamma=1.0):
        super().__init__(parameter)
        self.target_index = target_index

        self.kernel = RBFKernel(alpha, gamma)

        self._f = lambda x, X_train, y_train: self.kernel(x, X_train) @ jnp.linalg.solve(self.kernel.matrix(X_train), y_train)
        self._vf =  jax.vmap(self._f, in_axes=(0, None, None), out_axes=0)

    def evaluate(self, x, y):
        return self._vf(x, x, y)

    def __call__(self, Z):
        return self.evaluate(self.parameter(Z), jnp.asarray(Z[:, self.target_index]))

    def rkhs_norm(self, Z):
        return jnp.sum(
            jnp.square(
                jnp.asarray(Z[:, self.target_index].T) @ jnp.linalg.solve(self.kernel.matrix(self.parameter(Z)), jnp.asarray(Z[:, self.target_index]))
            )
        )


class UnknownFunctionDerivative(Function):

    def __init__(self, parameter):
        super().__init__(parameter)

        self._df = jax.grad(self.parameter._f)
        self._vdf = jax.vmap(self._df, in_axes=(0, None, None), out_axes=0)

    def evaluate(self, x, y):
        return self._vdf(x, x, y)

    def __call__(self, Z):
        return self.evaluate(self.parameter.parameter(Z), jnp.asarray(Z[:, self.parameter.target_index]))


class KnownFunction(Function):

    def __init__(self, parameter, fn):
        super().__init__(parameter)
        self._f = fn

    def evaluate(self, x):
        return self._f(x)

