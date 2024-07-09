from typing import Any, Callable
from dataclasses import dataclass, fields

import jax.numpy as jnp
import numpy as np
import jax

from cgc.utils import KernelParameter


def identity_functional(f, argnums):
    return f


def tensor_to_matrix(tensor):
    if tensor.ndim == 4:
        w1, w2, h1, h2 = tensor.shape
        tensor = jnp.reshape(jnp.transpose(tensor, (0, 2, 1, 3)), (w1 * h1, w2 * h2), order='F')

    return tensor


def get_regulaization_term(kernel_matrix, alpha):
    N, *_ = kernel_matrix.shape
    
    eye = jnp.eye(N)
    if kernel_matrix.ndim > 2:
        for _ in range(kernel_matrix.ndim - 2):
            eye = eye[..., jnp.newaxis]
    
    return alpha * eye


@dataclass
class BaseKernelParameters:

    def __post_init__(self):
        names = []
        for field in fields(self):
            names.append(field.name)
        self.sorted_names = np.sort(names)

    def length(self):
        return len(fields(self))
    
    def get_sorted_names(self):
        return self.sorted_names

    def gather(self):
        names = []
        values = []
        learnable_mask = []
        sparse_mask = []

        for field in fields(self):
            names.append(field.name)
            param: KernelParameter = getattr(self, field.name)
            values.append(param())
            learnable_mask.append(1 if param.is_learnable() else 0)
            sparse_mask.append(1 if param.is_weight() else 0)

        sorted_indecies = np.argsort(names)
        sorted_values = jnp.array(values)[sorted_indecies]
        sorted_learnable_mask = jnp.array(learnable_mask)[sorted_indecies]
        sorted_sparse_mask = jnp.array(sparse_mask)[sorted_indecies]

        return sorted_values, sorted_learnable_mask, sorted_sparse_mask

    def scatter(self, params_array) -> None:
        names = []
        for field in fields(self):
            names.append(field.name)

        sorted_names = np.sort(names)
        for i, name in enumerate(sorted_names):
            param: KernelParameter = getattr(self, name)
            if param.is_learnable():
                param.update(params_array[i])


class BaseKernel:
    
    def __init__(self, parameters: BaseKernelParameters, alpha: float = 1.0, linear_functional: Callable = identity_functional):
        self.parameters = parameters
        self.alpha = alpha
        self.linear_functional = linear_functional

    def _eval(self, x, y):
        raise NotImplementedError()
    
    def _get_params_dictionary(self, params):
        sorted_names = self.parameters.get_sorted_names()
        return {k: v for k, v in zip(sorted_names, params)}
    
    def eval(self, x, y, params_array=None):
        if params_array is None:
            params_array, *_ = self.parameters.gather()

        params_dict = self._get_params_dictionary(params_array)
        
        output = self._eval(x, y, **params_dict)

        return output
    
    def __call__(self, x, X_train, params_array=None):
        operated_eval = self.linear_functional(self.eval, argnums=1)
        return jax.vmap(operated_eval, in_axes=(None, 0, None))(x, X_train, params_array)

    def matrix(self, X_train, params_array=None, convert_tesnor_to_matrix=True):
        N, *_ = X_train.shape
        
        operated_eval = self.linear_functional(self.linear_functional(self.eval, argnums=0), argnums=1)
        K = jax.vmap(lambda x: jax.vmap(lambda y: operated_eval(x, y, params_array=params_array))(X_train))(X_train)

        if convert_tesnor_to_matrix:
            K = tensor_to_matrix(K)

        return K + get_regulaization_term(K, self.alpha)