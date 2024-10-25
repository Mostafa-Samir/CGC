from abc import abstractmethod
from typing import Callable
from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from tqdm import trange
from jaxopt import LBFGS, BFGS, ScipyMinimize
import optax

from cgc.early_stopper import EarlyStopper

class GDOptimizer:

    def __init__(self, loss_fn: Callable, learning_rate: float = 0.01, iterations_max: int = 500000, min_improvement: float = 1, patience: int = 1000):
        self.learning_rate = learning_rate
        self.iterations_max = iterations_max
        self.loss_grad = jax.grad(loss_fn)
        self.jitted_loss = jax.jit(loss_fn)
        self.jitted_update_step = jax.jit(self.update_step)
        self.early_stopper = EarlyStopper(min_improvement=min_improvement, patience=patience)

    @abstractmethod
    def update_step(self, Z, X, M):
        pass

    def run(self, Z, X, M, description_prefix=""):

        pbar = trange(self.iterations_max)
        for i in pbar:
            loss = self.jitted_loss(Z, X, M)
            pbar.set_description(f"Loss: {loss:.4f}")
            Z = self.jitted_update_step(Z, X, M)
            _, stop = self.early_stopper.check(loss, i, Z)
            if stop:
                Z = self.early_stopper.get_best_params()
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
    
class BFGSOptimizer():
    def __init__(self, loss_fn: Callable, learning_rate: float = 0.01, iterations_max: int = 1000000, min_improvement: float = 1, patience: int = 10000,  options={"ftol": 1e-6}):
        self.loss_fn = loss_fn
        self.jitted_loss = jax.jit(loss_fn)
        self._optimizer = ScipyMinimize(method="L-BFGS-B", fun=loss_fn, jit=True, maxiter=iterations_max)
        self.early_stopper = EarlyStopper(min_improvement, patience)
        self.iteration_counter = 0
        #self.jitted_update_step = jax.jit(self._optimizer.update)

    def run(self, Z, X, M, description_prefix=""):
        pbar = trange(self._optimizer.maxiter)
        def progressbar_callback(Z):
            self.iteration_counter += 1
            loss = self.jitted_loss(Z, X, M)
            pbar.set_description(f"{description_prefix} Loss: {loss:.4f}")
            pbar.update(1)

            _, stop = self.early_stopper.check(loss, self.iteration_counter, Z)
            #if stop:
            #    print(f"Stopping after {self.early_stopper.patience} iterations with no improvements.")
            #    raise StopIteration

        self._optimizer.callback = progressbar_callback

        Z, *_ = self._optimizer.run(Z, X, M)

        pbar.close()
        
        return Z
    

class OptaxBFGSOptimizer:
    def __init__(self, loss_fn: Callable, learning_rate: float = 0.01, iterations_max: int = 1000000, min_improvement: float = 1, patience: int = 10000,  options={"ftol": 1e-6}):
        self.loss_fn = loss_fn
        self.jitted_loss = jax.jit(loss_fn)
        self._optimizer = ScipyMinimize(method="L-BFGS-B", fun=loss_fn, jit=True, maxiter=iterations_max)
        self.early_stopper = EarlyStopper(min_improvement, patience)
        self.iteration_counter = 0
        #self.jitted_update_step = jax.jit(self._optimizer.update)
    

class BFGSOptimizerForKF():
    def __init__(self, loss_fn: Callable, learning_rate: float = 0.01, iterations_max: int = 500000, min_improvement: float = 1, patience: int = 1000):
        self.loss_fn = loss_fn
        self.jitted_loss = jax.jit(loss_fn)
        self._optimizer = ScipyMinimize(method="L-BFGS-B", fun=loss_fn, jit=True, maxiter=1000)
        #self.jitted_update_step = jax.jit(self._optimizer.update)

    def run(self, params, Z, M, original_params, trainable_mask):
        pbar = trange(self._optimizer.maxiter)
        def progressbar_callback(params):
            self.iteration_counter += 1
            loss = self.jitted_loss(params, Z, M, original_params, trainable_mask)
            pbar.set_description(f"Loss: {loss:.4f}")
            pbar.update(1)

            _, stop = self.early_stopper.check(loss, self.iteration_counter)
            if stop:
                print(f"Stopping after {self.early_stopper.patience} iterations with no improvements.")
                raise StopIteration

        self._optimizer.callback = progressbar_callback

        params, *_ = self._optimizer.run(params, Z, M, original_params, trainable_mask)

        pbar.close()
        
        return params
    

class TwoStepsBFGSOptimizerForKF():
    def __init__(self, loss_fn: Callable, learning_rate: float = 0.01, iterations_max: int = 500000, min_improvement: float = 1, patience: int = 1000):
        self.loss_fn = loss_fn
        self.jitted_loss = jax.jit(loss_fn)
        self._valgrad = jax.value_and_grad(loss_fn)
        #self._optimizer = ScipyMinimize(method="L-BFGS-B", value_and_grad=self._effective_val_and_grad, jit=True, maxiter=10000)

    def _effective_val_and_grad(self, params, Z, M, original_params, trainable_mask):
        val, grad = self._valgrad(params, Z, M, original_params, trainable_mask)
        eff_grad = grad * trainable_mask
        return val, eff_grad

    def run(self, params, Z, M, original_params, trainable_mask, sparse_mask, special_mask=None):

        if special_mask is None:
            special_mask = np.ones_like(trainable_mask)

        def progressbar_callback(params, pass_name):
            loss = self.jitted_loss(params, Z, M, original_params, trainable_mask)
            pbar.set_description(f"({pass_name}) Loss: {loss:.4f}")
            pbar.update(1)

        # Parameters Pass
        params_lernable_mask = trainable_mask * (1 - sparse_mask) * special_mask
        params_optimizer = ScipyMinimize(method="L-BFGS-B", fun=self.loss_fn, value_and_grad=self._effective_val_and_grad, jit=True, maxiter=10000, options={"ftol": 1e-6})
        pbar = trange(params_optimizer.maxiter)
        params_optimizer.callback = partial(progressbar_callback, pass_name="Parameters Pass")
        kparams, *_ = params_optimizer.run(params, Z, M, original_params, params_lernable_mask)
        pbar.close()

        # Weights Pass
        weights_lernable_mask = trainable_mask * sparse_mask * special_mask
        weights_optimizer = ScipyMinimize(method="L-BFGS-B", fun=self.loss_fn, value_and_grad=self._effective_val_and_grad, jit=True, maxiter=10000, options={"ftol": 1e-6})
        pbar = trange(params_optimizer.maxiter)
        weights_optimizer.callback = partial(progressbar_callback, pass_name="Weights Pass")
        wparams, *_ = weights_optimizer.run(kparams, Z, M, np.array(kparams), weights_lernable_mask)
        pbar.close()

        
        return wparams


class GDOptimizerForKF:

    def __init__(self, loss_fn: Callable, learning_rate: float = 0.001, iterations_max: int = 500000, min_improvement: float = 0.1, patience: int = 1000):
        self.learning_rate = learning_rate
        self.iterations_max = iterations_max
        self.loss_grad = jax.grad(loss_fn)
        self.jitted_loss = jax.jit(loss_fn)
        self.jitted_update_step = jax.jit(self.update_step)
        self.early_stopper = EarlyStopper(min_improvement=min_improvement, patience=patience)

    @abstractmethod
    def update_step(self, params, Z):
        pass

    def run(self, params, Z, M, original_params, trainable_mask, weights_mask, prefix=""):
        
        pbar = trange(self.iterations_max)
        for i in pbar:
            loss = self.jitted_loss(params, Z, M, original_params, trainable_mask, weights_mask)
            pbar.set_description(f"({prefix}) Loss: {loss:.9f}")
            params = self.jitted_update_step(params, Z, M, original_params, trainable_mask, weights_mask)
            _, stop = self.early_stopper.check(loss, i)
            if stop:
                print(f"Stopped after {self.early_stopper.patience} steps with no improvment in Loss")
                break
        
        pbar.close()
        
        return params


class NormalizedGDOptimizerForKF(GDOptimizerForKF):

    def update_step(self, params, Z, M, original_params, trainable_mask, weights_mask):
        g = self.loss_grad(params, Z, M, original_params, trainable_mask, weights_mask)
        non_nan_gradients = ~jnp.isnan(g)
        non_nan_g = jnp.where(non_nan_gradients, g, 0)
        g_norm = jnp.linalg.norm(g)
        adjusted_g_norm = jnp.where(g_norm > 0.0, g_norm, 1.0)
        normed_g = non_nan_g / adjusted_g_norm
        new_params = params - self.learning_rate * (normed_g * trainable_mask)
        return new_params
    

class TwoStepsNGDptimizerForKF(NormalizedGDOptimizerForKF):

    def run(self, params, Z, M, original_params, trainable_mask, sparse_mask, special_mask=None):

        if special_mask is None:
            special_mask = np.ones_like(trainable_mask)

         # Parameters Pass
        params_lernable_mask = trainable_mask * (1 - sparse_mask) * special_mask
        kparams = super().run(params, Z, M, np.array(params), params_lernable_mask, sparse_mask, prefix="Parameters Pass")

        self.early_stopper.reset()

        # Weights Pass
        weights_lernable_mask = trainable_mask * sparse_mask * special_mask
        self.learning_rate *= 10
        wparams = super().run(kparams, Z, M, np.array(kparams), weights_lernable_mask, sparse_mask, prefix="Weights Pass")

        return wparams
    

class BFGSOptimizerForLearningParams():
    def __init__(self, loss_fn: Callable, learning_rate: float = 0.01, iterations_max: int = 500000, min_improvement: float = 1, patience: int = 1000):
        self.loss_fn = loss_fn
        self.jitted_loss = jax.jit(loss_fn)
        self._valgrad = jax.value_and_grad(self.loss_fn)
        #self.jitted_update_step = jax.jit(self._optimizer.update)


    def _effective_val_and_grad(self, params, Z, X, M, trainable_mask):
        val, grad = self._valgrad(params, Z, X, M)
        eff_grad = grad * trainable_mask
        return val, eff_grad

    def run(self, params, Z, X, M, trainable_mask, weights_mask, description_prefix=""):

        def progressbar_callback(params, extras=""):
            loss = self.jitted_loss(params, Z, X, M)
            pbar.set_description(f"{description_prefix} {extras} Loss: {loss:.4f}")
            pbar.update(1)
        

        interanl_params_mask = trainable_mask * (1 - weights_mask)
        _optimizer = ScipyMinimize(
            method="L-BFGS-B", 
            fun=self.loss_fn, 
            value_and_grad=partial(self._effective_val_and_grad, trainable_mask=interanl_params_mask),
            jit=True, 
            maxiter=10000
        )
        pbar = trange(_optimizer.maxiter)
        _optimizer.callback = partial(progressbar_callback, extras="[Internal Params Pass]")
        internal_params, *_ = _optimizer.run(params, Z, X, M)
        pbar.close()

        weights_params_mask = trainable_mask * weights_mask
        _optimizer = ScipyMinimize(
            method="L-BFGS-B", 
            fun=self.loss_fn, 
            value_and_grad=partial(self._effective_val_and_grad, trainable_mask=weights_params_mask),
            jit=True, 
            maxiter=10000
        )
        pbar = trange(_optimizer.maxiter)
        _optimizer.callback = partial(progressbar_callback, extras="[Weights Pass]")
        all_params, *_ = _optimizer.run(np.array(internal_params), Z, X, M)
        pbar.close()
        
        return all_params