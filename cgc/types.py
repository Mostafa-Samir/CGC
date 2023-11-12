from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Union

from cgc.kernels import RBFKernel, tensor_to_matrix, get_regulaization_term

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
        self.gamma = gamma
        self.alpha = alpha
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
    
    def kflow_loss(self, gamma, Z, sample_ratio=0.5, n_samples=10):
        self.gamma.update(gamma)

        K_matrix = self.kernel.matrix(self.parameter(Z), convert_tesnor_to_matrix=False)
        K_matrix = K_matrix - get_regulaization_term(K_matrix, self.alpha)
        N, *_ = K_matrix.shape

        observations = jnp.asarray(self.observation(Z))

        w_full = None
        full_norm = None

        loss = 0
        n_observations, *_ = Z.shape
        sample_size = int(sample_ratio * n_observations)
        key = jax.random.PRNGKey(seed=42)

        for _ in range(n_samples):
            sample_indecies = jax.random.choice(key, n_observations, shape=(sample_size, ), replace=False)
            sample_indecies = jnp.sort(sample_indecies)

            K_matrix_sample = tensor_to_matrix(K_matrix[jnp.ix_(sample_indecies, sample_indecies)])
            N_sample, *_ = K_matrix_sample.shape
            
            K_matrix_cross = tensor_to_matrix(K_matrix[jnp.ix_(jnp.arange(N), sample_indecies)])
            
            observations_sample = jnp.reshape(observations[sample_indecies, ...], (N_sample, ), order='F')

            if w_full is None:
                K_matrix_full = tensor_to_matrix(K_matrix)
                N_full, *_ = K_matrix_full.shape

                observations_full = jnp.reshape(observations, (N_full, ), order='F')
                w_full = jnp.linalg.solve(K_matrix_full + get_regulaization_term(K_matrix_full, self.alpha), observations_full)
                full_norm = w_full @ K_matrix_full @ w_full

            w_sample = jnp.linalg.solve(K_matrix_sample + get_regulaization_term(K_matrix_sample, self.alpha), observations_sample)
            sample_norm = w_sample @ K_matrix_sample @ w_sample

            loss += 1 + ((sample_norm - 2 * w_full @ K_matrix_cross @ w_sample) / full_norm)

        return loss / n_samples




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
    

class LearnableParameter(CallableInterface):

    def __init__(self, init_value) -> None:
        self.value_ = init_value

    def update(self, new_value):
        self.value_ = new_value

    def __call__(self, _=None) -> Any:
        return self.value_


class ConstantParameter(CallableInterface):

    def __init__(self, value) -> None:
        self.value_ = value

    def __call__(self, _=None) -> Any:
        return self.value_