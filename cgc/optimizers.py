from abc import abstractmethod
from typing import Callable

import jax
import jax.numpy as jnp
from tqdm import trange

from cgc.early_stopper import EarlyStopper

class GDOptimizer:

    def __init__(self, loss_fn: Callable, learning_rate: float = 0.001, iterations_max: int = 500000, min_improvement: float = 0.01, patience: int = 1000):
        self.learning_rate = learning_rate
        self.iterations_max = iterations_max
        self.loss_grad = jax.grad(loss_fn)
        self.jitted_loss = jax.jit(loss_fn)
        self.jitted_update_step = jax.jit(self.update_step)
        self.early_stopper = EarlyStopper(min_improvement=min_improvement, patience=patience, improvement_type="relative")

    @abstractmethod
    def update_step(self, Z, X, M):
        pass

    def run(self, Z, X, M):

        pbar = trange(self.iterations_max)
        for i in pbar:
            loss = self.jitted_loss(Z, X, M)
            pbar.set_description(f"Loss: {loss:.4f}")
            Z = self.jitted_update_step(Z, X, M)
            _, stop = self.early_stopper.check(loss, i)
            if stop:
                print(f"Stopped after {self.early_stopper.patience} steps with no improvment in Loss")
                break

        return Z


class NormalizedGDOptimizer(GDOptimizer):

    def update_step(self, Z, X, M):
        g = self.loss_grad(Z, X, M)
        normed_g = g / jnp.linalg.norm(g)
        new_Z = Z - self.learning_rate * normed_g
        return new_Z


class ProjectedNGDOptimizer(GDOptimizer):

    def update_step(self, Z, X, M):
        g = self.loss_grad(Z, X, M)
        normed_g = g / jnp.linalg.norm(g)
        new_Z = Z - self.learning_rate * normed_g
        new_Z = jnp.where(M, X, new_Z)
        return new_Z

