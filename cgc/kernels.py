from typing import Any
import jax.numpy as jnp
import jax

class RBFKernel:
    
    def __init__(self, alpha=1.0, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self._vf = None

        self._metric = lambda x, y: jnp.dot((x - y), (x - y))

    def __call__(self, x, X_train):
        dists = jax.vmap(lambda X: self._metric(X / self.gamma, x / self.gamma))(X_train)
        return jnp.exp(-0.5 * dists)

    def matrix(self, X_train):
        if X_train.ndim > 1:
            N, *_ = X_train.shape
        else:
            N = X_train.size
        
        dists = jax.vmap(lambda scaled_X: jax.vmap(lambda scaled_y: self._metric(scaled_X, scaled_y))(X_train / self.gamma))(X_train / self.gamma)
        K =  jnp.exp(-0.5 * dists)

        return K + self.alpha * jnp.eye(N)
