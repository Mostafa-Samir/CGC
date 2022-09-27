from typing import List

import jax.numpy as jnp


from cgc.types import Observable, Aggregator, UnknownFunction, KnownFunction, Function, UnknownFunctionDerivative
from cgc.optimizers import NormalizedGDOptimizer, ProjectedNGDOptimizer

def derivative(fn: Function):
    if isinstance(fn, UnknownFunction):
        return UnknownFunctionDerivative(fn)

class ComputationalGraph:

    def __init__(self, observables_order):

        self._observables_order = {symbol: index for (index, symbol) in enumerate(observables_order)}

        self._observables = dict()
        self._unknown_functions = dict()
        self._known_functions = dict()
        self._aggregators = dict()
        self._constraints = dict()

        self.unknown_functions_loss_multipler = 1000
        self.contraints_loss_multiplier = 1000
        self.data_compliance_loss_multipler = 1000

    def _get_callable_parameter(self, parameter_name: str):
        callable_parameter = None
        for container in [self._observables, self._unknown_functions, self._known_functions, self._aggregators]:
            if parameter_name in container:
                callable_parameter = container.get(parameter_name)
                break
        
        if callable_parameter is None:
            raise KeyError(f"Parameter {parameter_name} is not defined")

        return callable_parameter

    def add_observable(self, name: str):
        self._observables[name] = Observable(self._observables_order.get(name))

    def add_unknown_fn(self, parameter: str, target: str, alpha=1, gamma=1):
        callable_parameter = self._get_callable_parameter(parameter)
        self._unknown_functions[target] = UnknownFunction(callable_parameter, target_index=self._observables_order.get(target), alpha=alpha, gamma=gamma)

    def add_known_fn(self, parameter: str, target: str, fn):
        callable_parameter = self._get_callable_parameter(parameter)
        if fn is derivative:
            self._known_functions[target] = derivative(callable_parameter)
        else:
            self._known_functions[target] = KnownFunction(callable_parameter, fn=fn)

    def add_aggregator(self, parameters: List[str], target: str):
        callable_parameters = [self._get_callable_parameter(parameter) for parameter in parameters]
        self._aggregators[target] = Aggregator(callable_parameters)

    def add_constraint(self, parameter: str, target: str, fn):
        callable_parameter = self._get_callable_parameter(parameter)
        self._constraints[target] = KnownFunction(callable_parameter, fn)


    def set_loss_multipliers(
        self,
        unknown_functions_loss_multiplier: float = 1000,
        constraints_loss_multiplier: float = 1000,
        data_compliance_loss_multiplier: float = 1000
    ):
        self.unknown_functions_loss_multipler = unknown_functions_loss_multiplier
        self.contraints_loss_multiplier = constraints_loss_multiplier
        self.data_compliance_loss_multipler = data_compliance_loss_multiplier


    def _loss(self, Z, X, M):
        rkhs_norms = 0
        unknown_funcs_loss = 0
        data_compliance_loss = 0
        constraints_loss = 0

        for (fn_name, fn) in self._unknown_functions.items():
            rkhs_norms += fn.rkhs_norm(Z)
            unknown_funcs_loss += jnp.sum(jnp.square(fn(Z) - Z[:, self._observables_order.get(fn_name)]))

        multipled_unknwon_funcs_loss = self.unknown_functions_loss_multipler * unknown_funcs_loss

        for (_, fn) in self._constraints.items():
            constraints_loss += jnp.sum(jnp.square(fn(Z)))

        multiplied_constraints_loss = self.contraints_loss_multiplier * constraints_loss

        data_compliance_loss += jnp.sum(jnp.square(jnp.where(M, Z, 0) - jnp.where(M, X, 0)))
        multiplied_data_compliance_loss = self.data_compliance_loss_multipler * data_compliance_loss

        total_loss = rkhs_norms + multipled_unknwon_funcs_loss + multiplied_constraints_loss + multiplied_data_compliance_loss

        return total_loss


    def complete(self, X, M, optimizer="normalized-gd"):

        optimizer_class = NormalizedGDOptimizer if optimizer == "normalized-gd" else ProjectedNGDOptimizer
        optimizer_obj = optimizer_class(self._loss)

        Z_initial = X.copy()

        Z = optimizer_obj.run(Z_initial, X, M)

        return Z


