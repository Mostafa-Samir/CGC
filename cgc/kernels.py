from typing import Any
import jax.numpy as jnp
import numpy as np
import jax

def identity_functional(f, argnums):
    return f

class RBFKernel:
    
    def __init__(self, alpha=1.0, gamma=1.0, linear_functional=None):
        self.alpha = alpha
        self.gamma = gamma
        self.linear_functional = linear_functional or identity_functional
        self._vf = None

    def _eval(self, x, y):
        diff = (x - y) / self.gamma
        return jnp.exp(-0.5 * jnp.dot(diff, diff))

    def __call__(self, x, X_train):
        operated_eval = self.linear_functional(self._eval, argnums=1)
        return jax.vmap(operated_eval, in_axes=(None, 0))(x, X_train)

    def matrix(self, X_train):
        N, *_ = X_train.shape
        
        operated_eval = self.linear_functional(self.linear_functional(self._eval, argnums=0), argnums=1)
        K = jax.vmap(lambda x: jax.vmap(lambda y: operated_eval(x, y))(X_train))(X_train)

        if K.ndim == 4:
            w, _, h, _ = K.shape
            K = jnp.reshape(jnp.transpose(K, (0, 2, 1, 3)), (w * h, w * h), order='F')
            N = w * h

        return K + self.alpha * jnp.eye(N)
