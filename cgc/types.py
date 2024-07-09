from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Union

from cgc.kernels_old import RBFKernel, tensor_to_matrix, get_regulaization_term, QuadraticKernel, QuadraticAdditiveKernel, CubicAdditiveKernel, CubicKernel, PolyKernel, PolyAdditiveKernel
from cgc.kernels.factory import KernelsFactory

import jax.numpy as jnp
import jax

class CallableInterface(ABC):

    @abstractmethod
    def __call__(self, X: Any) -> Any:
        raise NotImplementedError()

class Observable(CallableInterface):
    
    def __init__(self, index: int) -> None:
        self.index = index

    def __call__(self, Z, params=None):
        return jnp.asarray(Z[:, self.index])

class Function(CallableInterface):

    def __init__(self, parameter: CallableInterface):
        self.parameter = parameter

    @abstractmethod
    def evaluate(self, Z):
        raise NotImplementedError()

    def __call__(self, Z, params=None):
        return self.evaluate(self.parameter(Z, params))


class Aggregator(CallableInterface):

    def __init__(self, parameters: List[CallableInterface]) -> None:
        self.parameters = parameters

    def __call__(self, Z, params=None):
        evaluated_parameters = [parameter(Z, params) for parameter in self.parameters]
        parameters_2d = [parameter[:, jnp.newaxis] if parameter.ndim == 1 else parameter for parameter in evaluated_parameters]
        return jnp.concatenate(parameters_2d, axis=1)
    


class UnknownFunction(Function):

    #def __init__(self, parameter, kernel="rbf", alpha=1.0, gamma=1.0, linear_functional=None, observation=None):
    def __init__(self, parameter, kernel, kernel_parameters, alpha=1.0, gamma=1.0, linear_functional=None, observation=None):
        super().__init__(parameter)
        self.observation = observation
        #self.gamma = gamma
        self.kernel_parameters = kernel_parameters
        self.alpha = alpha
        #self.parameter_order = None
        self.parameters_range = (None, None)


        #self.kernel = kernels_dict.get(kernel)(alpha, gamma, linear_functional=linear_functional)
        self.kernel = KernelsFactory.create(kernel, kernel_parameters, alpha, linear_functional)
        self._vf =  jax.vmap(self._f, in_axes=(0, None, None, None), out_axes=0)

    def _f(self, x, X_train, y_train, params_array=None):
        matrix = self.kernel.matrix(X_train, params_array)
        sims = self.kernel(x, X_train, params_array)

        _, matrix_trainling_dim_size = matrix.shape
        observation_leading_dim_size, *_ = y_train.shape
        *_, sims_trailing_dim_size = sims.shape

        if matrix_trainling_dim_size != observation_leading_dim_size:
            y_train = jnp.reshape(y_train, (matrix_trainling_dim_size, ), order='F')

        if matrix_trainling_dim_size != sims_trailing_dim_size:
            sims = jnp.reshape(sims, (matrix_trainling_dim_size, ), order='F')

        return sims @ jnp.linalg.solve(matrix, y_train)
        
    def evaluate(self, x, y, params_array=None):
        return self._vf(x, x, y, params_array)

    def __call__(self, Z, params=None):
        params_start, params_end = self.parameters_range
        params_array = None if params is None else params[params_start:params_end]
        return self.evaluate(self.parameter(Z, params), self.observation(Z, params), params_array=params_array)

    def rkhs_norm(self, Z, params=None):
        params_start, params_end = self.parameters_range
        params_array = None if params is None else params[params_start:params_end]
        matrix = self.kernel.matrix(self.parameter(Z), params_array=params_array)
        observations = jnp.asarray(self.observation(Z))

        _, matrix_trainling_dim_size = matrix.shape
        observation_leading_dim_size, *_ = observations.shape

        if matrix_trainling_dim_size != observation_leading_dim_size:
            observations = jnp.reshape(observations, (matrix_trainling_dim_size, ), order='F')
        
        return jnp.square(observations.T @ jnp.linalg.solve(matrix, observations))
    

    def kflow_loss_(self, params, Z, M, original_params, trainable_mask, sample_ratio=0.5, n_samples=20):

        loss = 0

        params_start, params_end = self.parameters_range
        params_array = params[params_start:params_end]
        #trainable_mask = trainable_mask[params_start: params_end]
        #original_params = original_params[params_start:params_end]

        #params_array = trainable_mask * params_array + (1 - trainable_mask) * original_params
        
        key = jax.random.PRNGKey(seed=42)
        n_observations, *_ = Z.shape
        permutation = jax.random.permutation(key, n_observations)

        K = self.kernel.matrix(self.parameter(Z, params), params_array, convert_tesnor_to_matrix=False)
        K = K - get_regulaization_term(K, self.alpha)
        O = jnp.array(self.observation(Z, params))

        K_mat = tensor_to_matrix(K)
        N_mat, *_ = K_mat.shape
        O_vec = jnp.reshape(O, (N_mat, ), order='F')
        rkhs_full = O_vec.T @ jnp.linalg.solve(K_mat + get_regulaization_term(K_mat, self.alpha), O_vec)

        sample_size = int(n_observations * sample_ratio)

        for _ in range(n_samples):
            key, subkey = jax.random.split(key)
            sample_indecies = jax.random.choice(subkey, n_observations, shape=(sample_size, ), replace=False)
            sample_indecies = jnp.sort(sample_indecies)

            K_sample = K[jnp.ix_(sample_indecies, sample_indecies)]
            O_sample = O[sample_indecies]
            K_sample_mat = tensor_to_matrix(K_sample)
            N_sample_mat, *_ = K_sample_mat.shape
            O_sample_vec = jnp.reshape(O_sample, (N_sample_mat, ), order='F')
            rkhs_sample = O_sample_vec.T @ jnp.linalg.solve(K_sample_mat + get_regulaization_term(K_sample_mat, self.alpha), O_sample_vec)

            
            loss += 1 - (rkhs_sample / rkhs_full)

        loss /= n_samples

        return loss

    def kflow_loss(self, params, Z, M, original_params, trainable_mask, sample_ratio=0.5, n_samples=20):

        params_start, params_end = self.parameters_range
        params_array = params[params_start:params_end]


        K_matrix = self.kernel.matrix(self.parameter(Z, params), params_array, convert_tesnor_to_matrix=False)
        K_matrix = K_matrix - get_regulaization_term(K_matrix, self.alpha)
        N, *_ = K_matrix.shape

        observations = jnp.asarray(self.observation(Z, params))

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

    def __call__(self, Z, params=None):
        return self.evaluate(self.parameter.parameter(Z, params), jnp.asarray(self.parameter.observation(Z, params)))


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