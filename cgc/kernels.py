from typing import Any
import jax.numpy as jnp
import numpy as np
import jax

class ConstantParameter:

    def __init__(self, value) -> None:
        self.value_ = value

    def __call__(self, _=None) -> Any:
        return self.value_

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

def identity_functional(f, argnums):
    return f

class RBFKernel:
    
    def __init__(self, alpha=1.0, gamma=1.0, linear_functional=None):
        self.alpha = alpha
        self.gamma = gamma
        self.linear_functional = linear_functional or identity_functional
        self._vf = None

    def _eval(self, x, y, gamma=None):
        effective_gamma = self.gamma() if gamma is None else gamma
        diff = (x - y) / effective_gamma
        return jnp.exp(-0.5 * jnp.dot(diff, diff))

    def __call__(self, x, X_train, M=None, gamma=None):
        operated_eval = self.linear_functional(self._eval, argnums=1)
        return jax.vmap(operated_eval, in_axes=(None, 0, None))(x, X_train, gamma)

    def matrix(self, X_train, gamma=None, convert_tesnor_to_matrix=True):
        N, *_ = X_train.shape
        
        operated_eval = self.linear_functional(self.linear_functional(self._eval, argnums=0), argnums=1)
        K = jax.vmap(lambda x: jax.vmap(lambda y: operated_eval(x, y, gamma=gamma))(X_train))(X_train)

        if convert_tesnor_to_matrix:
            K = tensor_to_matrix(K)

        return K + get_regulaization_term(K, self.alpha)
    

class QuadraticKernel:
    
    def __init__(self, alpha=1.0, gamma=1.0, linear_functional=None):
        self.alpha = alpha
        self.gamma = gamma
        self.linear_functional = linear_functional or identity_functional
        self._vf = None

    def _eval(self, x, y, gamma=None):
        effective_gamma = self.gamma() if gamma is None else gamma
        return jnp.power(jnp.dot(x, y) + effective_gamma, 2)

    def __call__(self, x, X_train, gamma=None):
        operated_eval = self.linear_functional(self._eval, argnums=1)
        return jax.vmap(operated_eval, in_axes=(None, 0, None))(x, X_train, gamma)

    def matrix(self, X_train, gamma=None, convert_tesnor_to_matrix=True):
        N, *_ = X_train.shape
        
        operated_eval = self.linear_functional(self.linear_functional(self._eval, argnums=0), argnums=1)
        K = jax.vmap(lambda x: jax.vmap(lambda y: operated_eval(x, y, gamma=gamma))(X_train))(X_train)

        if convert_tesnor_to_matrix:
            K = tensor_to_matrix(K)

        return K + get_regulaization_term(K, self.alpha)
    
class CubicKernel:
    
    def __init__(self, alpha=1.0, gamma=1.0, linear_functional=None):
        self.alpha = alpha
        self.gamma = gamma
        self.linear_functional = linear_functional or identity_functional
        self._vf = None

    def _eval(self, x, y, gamma=None):
        effective_gamma = self.gamma() if gamma is None else gamma
        return jnp.power(jnp.dot(x, y) + effective_gamma, 3)

    def __call__(self, x, X_train, gamma=None):
        operated_eval = self.linear_functional(self._eval, argnums=1)
        return jax.vmap(operated_eval, in_axes=(None, 0, None))(x, X_train, gamma)

    def matrix(self, X_train, gamma=None, convert_tesnor_to_matrix=True):
        N, *_ = X_train.shape
        
        operated_eval = self.linear_functional(self.linear_functional(self._eval, argnums=0), argnums=1)
        K = jax.vmap(lambda x: jax.vmap(lambda y: operated_eval(x, y, gamma=gamma))(X_train))(X_train)

        if convert_tesnor_to_matrix:
            K = tensor_to_matrix(K)

        return K + get_regulaization_term(K, self.alpha)
    

class QuadraticAdditiveKernel:

    def __init__(self, alpha=1.0, gamma=1.0, linear_functional=None):
        self.gamma = gamma
        self.quad_kernel = QuadraticKernel(alpha, gamma=ConstantParameter(1.0), linear_functional=linear_functional)
        self.gaussian_kernel = RBFKernel(alpha, gamma=gamma, linear_functional=linear_functional)

    def __call__(self, x, X_train, gamma=None):
        return self.quad_kernel(x, X_train) + self.gaussian_kernel(x, X_train, gamma=gamma)
    
    def matrix(self, X_train, gamma=None, convert_tesnor_to_matrix=True):
        K_quad = self.quad_kernel.matrix(X_train, convert_tesnor_to_matrix=convert_tesnor_to_matrix)
        K_rbf = self.gaussian_kernel.matrix(X_train, convert_tesnor_to_matrix=convert_tesnor_to_matrix, gamma=gamma)

        return K_quad + K_rbf
    
class CubicAdditiveKernel:

    def __init__(self, alpha=1.0, gamma=1.0, linear_functional=None):
        self.gamma = gamma
        self.quad_kernel = CubicKernel(alpha, gamma=ConstantParameter(1.0), linear_functional=linear_functional)
        self.gaussian_kernel = RBFKernel(alpha, gamma=gamma, linear_functional=linear_functional)

    def __call__(self, x, X_train, gamma=None):
        return self.quad_kernel(x, X_train) + self.gaussian_kernel(x, X_train, gamma=gamma)
    
    def matrix(self, X_train, gamma=None, convert_tesnor_to_matrix=True):
        K_quad = self.quad_kernel.matrix(X_train, convert_tesnor_to_matrix=convert_tesnor_to_matrix)
        K_rbf = self.gaussian_kernel.matrix(X_train, convert_tesnor_to_matrix=convert_tesnor_to_matrix, gamma=gamma)

        return K_quad + K_rbf
