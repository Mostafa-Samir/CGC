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
        evaluated_parameters = [parameter(Z) for parameter in self.parameters]
        parameters_2d = [parameter[:, jnp.newaxis] if parameter.ndim == 1 else parameter for parameter in evaluated_parameters]
        return jnp.concatenate(parameters_2d, axis=1)
    


class UnknownFunction(Function):

    def __init__(self, parameter, kernel="rbf", alpha=1.0, gamma=1.0, linear_functional=None, observation=None):
        super().__init__(parameter)
        self.observation = observation

        self.kernel = RBFKernel(alpha, gamma, linear_functional=linear_functional)
        self._vf =  jax.vmap(self._f, in_axes=(0, None, None), out_axes=0)

    def _f(self, x, X_train, y_train):
        matrix = self.kernel.matrix(X_train)
        sims = self.kernel(x, X_train)

        _, matrix_trainling_dim_size = matrix.shape
        observation_leading_dim_size, *_ = y_train.shape
        *_, sims_trailing_dim_size = sims.shape

        if matrix_trainling_dim_size != observation_leading_dim_size:
            y_train = jnp.reshape(y_train, (matrix_trainling_dim_size, ), order='F')

        if matrix_trainling_dim_size != sims_trailing_dim_size:
            sims = jnp.reshape(sims, (matrix_trainling_dim_size, ), order='F')

        return sims @ jnp.linalg.solve(matrix, y_train)
        
    def evaluate(self, x, y):
        return self._vf(x, x, y)

    def __call__(self, Z):
        return self.evaluate(self.parameter(Z), self.observation(Z))

    def rkhs_norm(self, Z):
        matrix = self.kernel.matrix(self.parameter(Z))
        observations = jnp.asarray(self.observation(Z))

        _, matrix_trainling_dim_size = matrix.shape
        observation_leading_dim_size, *_ = observations.shape

        if matrix_trainling_dim_size != observation_leading_dim_size:
            observations = jnp.reshape(observations, (matrix_trainling_dim_size, ), order='F')
        
        return jnp.sum(
            jnp.square(
               observations.T @ jnp.linalg.solve(matrix, observations)
            )
        )


class UnknownFunctionDerivative(Function):

    def __init__(self, parameter):
        super().__init__(parameter)

        self._df = jax.jacobian(self.parameter._f)
        self._vdf = jax.vmap(self._df, in_axes=(0, None, None), out_axes=0)

    def evaluate(self, x, y):
        return self._vdf(x, x, y)

    def __call__(self, Z):
        return self.evaluate(self.parameter.parameter(Z), jnp.asarray(self.parameter.observation(Z)))


class KnownFunction(Function):

    def __init__(self, parameter, fn):
        super().__init__(parameter)
        self._f = fn

    def evaluate(self, x):
        return self._f(x)

