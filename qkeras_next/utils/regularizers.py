from keras import ops
from keras.api.regularizers import Regularizer
from keras.api.saving import register_keras_serializable

from .misc import numbers


@register_keras_serializable(package="qkeras_next")
class MonoL1(Regularizer):
    def __init__(self, l1: numbers):
        self.l1 = float(l1)

    def __call__(self, x):
        return self.l1 * ops.sum(x)  # type: ignore

    def get_config(self):
        return {"l1": self.l1}
