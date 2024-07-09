import importlib
import pkgutil
from typing import Any, Callable, Dict, Type

from dacite import from_dict

from cgc.kernels.base import BaseKernel, BaseKernelParameters, identity_functional
from cgc.utils import KernelParameter

import cgc.kernels.concrete_kernels as kernels_pkg


class KernelsFactory:

    __kernels_registrey = {}
    __parameters_registery = {}

    @classmethod
    def register(cls, tracker_name: str, parameters_class: Type[BaseKernelParameters]) -> Callable:
        """Register an implementation of the BaseKernel class into the factory registery.

        Parameters
        ----------
        tracker_name : str
            the name of the kernel class to register
        parameters_class: Type[BaseKernelParameters]
            the class of kernel parameters associated with the tracker

        Returns
        -------
        Type[BaseKernel]
            the concrete BaseKernel object

        Raises
        ------
        TypeError
            if the class to be registered is not an instance of BaseKernel
            or if the parameters class is not an instance of BaseKernelParameters
        """

        def register_wrapper(decorated_class: Type[BaseKernel]) -> Type[BaseKernel]:
            if not issubclass(decorated_class, BaseKernel):
                raise TypeError("Attempting to register a non-BaseModel class.")


            cls.__kernels_registrey[tracker_name] = decorated_class
            cls.__parameters_registery[tracker_name] = parameters_class

            return decorated_class

        return register_wrapper
    
    @classmethod
    def create(cls, kernel_name: str, kernel_parameters: Dict[str, KernelParameter], alpha: float = 1.0, linear_functional: Callable = identity_functional) -> BaseKernel:
        """Create a concerte kernel from the given name and parameters dictionary.

        Parameters
        ----------
        kernel_name : str
           the name of the concrete kernel to create
        kernel_parameters: Dict[str, Any]
            the dictionary containing kernel parameters

        Returns
        -------
        BaseKernel
            An instance of the concrete kernel

        Raises
        ------
        KeyError
            if the model name provided is not registered
        """
        if kernel_name not in cls.__kernels_registrey:
            raise KeyError(f"{kernel_name} is not registered.")

        kernel_class = cls.__kernels_registrey.get(kernel_name)
        parameter_class = cls.__parameters_registery.get(kernel_name)

        parameters = from_dict(parameter_class, kernel_parameters)

        linear_functional = linear_functional or identity_functional
        kernel = kernel_class(parameters, alpha, linear_functional)

        return kernel
    

def _link_kernels_to_registry() -> None:
    """Link the concrete trackers with the factory registery."""
    for _, name, is_pkg in pkgutil.walk_packages(kernels_pkg.__path__):
        full_name = f"{kernels_pkg.__name__}.{name}"
        if not is_pkg:
            importlib.import_module(full_name)


_link_kernels_to_registry()