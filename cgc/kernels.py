import jax.numpy as jnp
import jax

class RBFKernel:
    
    def __init__(self, alpha=1.0, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self._vf = None

    def __call__(self, x, X_train):
        return jnp.exp((-(x - X_train) ** 2) / (2 * (self.gamma ** 2)))

    def matrix(self, X_train):
        if X_train.ndim > 1:
            N, *_ = X_train.shape
        else:
            N = X_train.size
        
        X_train = jnp.reshape(X_train, (-1, 1))
        K =  jnp.exp((-(X_train - X_train.T) ** 2) / (2 * (self.gamma ** 2)))

        return K + self.alpha * jnp.eye(N)
