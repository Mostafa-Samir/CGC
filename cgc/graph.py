from typing import List

import jax.numpy as jnp
import jax


from cgc.types import Observable, Aggregator, UnknownFunction, KnownFunction, Function, UnknownFunctionDerivative, ConstantParameter, LearnableParameter, CallableInterface
from cgc.optimizers import NormalizedGDOptimizer, ProjectedNGDOptimizer, BFGSOptimizer, BFGSOptimizerForKF, NormalizedGDOptimizerForKF

def derivative(fn: Function):
    if isinstance(fn, UnknownFunction):
        return UnknownFunctionDerivative(fn)

class ComputationalGraph:

    def __init__(self, observables_order):

        self._observables_order = {symbol: index for (index, symbol) in enumerate(observables_order)}

        self._observables = dict()
        self._unknown_functions = dict()
        self._unknown_functions_with_learnable_parameters = []
        self._unknown_functions_unconditioned = dict()
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

        if callable_parameter is None and parameter_name in self._observables_order:
            callable_parameter =  Observable(self._observables_order.get(parameter_name))
        
        if callable_parameter is None:
            raise KeyError(f"Parameter {parameter_name} is not defined")

        return callable_parameter

    def add_observable(self, name: str):
        self._observables[name] = Observable(self._observables_order.get(name))

    def add_unknown_fn(self, parameter: str, target: str, alpha=1, gamma=1, linear_functional=None, observations=None):

        if not isinstance(gamma, CallableInterface):
            gamma = ConstantParameter(gamma)
        elif isinstance(gamma, LearnableParameter):
            self._unknown_functions_with_learnable_parameters.append(target)
        
        callable_parameter = self._get_callable_parameter(parameter)
        callable_observation = self._get_callable_parameter(target if not linear_functional else observations)
        self._unknown_functions[target] = UnknownFunction(
            callable_parameter, alpha=alpha, gamma=gamma,
            linear_functional=linear_functional, observation=callable_observation
        )
        if linear_functional is not None:
            self._unknown_functions_unconditioned[target] = UnknownFunction(
                callable_parameter, alpha=alpha, gamma=gamma,
                observation=self._get_callable_parameter(target)
            )

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


    def _loss(self, Z, X, M, params=None):
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
    
    def _total_kflow_loss(self, params, Z):
        total_loss = 0
        constraint_loss = 0
        for i, fn_name in enumerate(self._unknown_functions_with_learnable_parameters):
            fn = self._unknown_functions[fn_name]
            total_loss += fn.kflow_loss(params[i], Z)

        return total_loss
    

    def _gather_learnable_parameters_values(self):
        params = []
        for fn_name in self._unknown_functions_with_learnable_parameters:
            fn = self._unknown_functions[fn_name]
            params.append(fn.gamma())

        return jnp.asarray(params)
    
    def _scatter_learnable_parameters_values(self, params_values):
        for i, fn_name in enumerate(self._unknown_functions_with_learnable_parameters):
            fn = self._unknown_functions[fn_name]
            fn.gamma.update(params_values[i])


    def complete(self, X, M, optimizer="normalized-gd", learn_parameters=False, n_rounds=10):

        optimizer_class = NormalizedGDOptimizer if optimizer == "normalized-gd" else BFGSOptimizer
        

        kf_opts = dict()

        Z = X.copy()

        for _ in range(n_rounds if learn_parameters else 1):
            
            optimizer_obj = optimizer_class(self._loss)
            Z = optimizer_obj.run(Z, X, M)

            if learn_parameters:
                #params = self._gather_learnable_parameters_values()
                for fn_name in self._unknown_functions_with_learnable_parameters:
                    if fn_name in self._unknown_functions_unconditioned:
                        fn = self._unknown_functions_unconditioned[fn_name]
                    else:
                        fn = self._unknown_functions[fn_name]

                    if fn_name not in kf_opts:
                        loss_fn = fn.kflow_loss
                        opt = NormalizedGDOptimizerForKF(loss_fn, patience=50, min_improvement=0.0001)
                        kf_opts[fn_name] = opt
                    else:
                        opt = kf_opts[fn_name]
                        if hasattr(opt, 'early_stopper'):
                            opt.early_stopper.reset()

                    param_init = fn.gamma.value_
                    param = opt.run(param_init, Z)
                    print(f"{fn_name}: {param}")
                    fn.gamma.update(param)

        print(self._gather_learnable_parameters_values)

        return Z


