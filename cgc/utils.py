class KernelParameter:
    def __init__(self, init_value, learnable=True, weight=False) -> None:
        self._value = init_value
        self._learnable = learnable
        self._weight = weight
    
    def update(self, new_value):
        self._value = new_value

    def is_learnable(self) -> bool:
        return self._learnable
    
    def is_weight(self) -> bool:
        return self._weight

    def __call__(self, *args):
        return self._value