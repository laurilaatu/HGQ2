from keras import ops
from keras.regularizers import Regularizer
from keras.saving import register_keras_serializable

from ..utils.misc import numbers


@register_keras_serializable(package='hgq')
class MonoL1(Regularizer):
    def __init__(self, l1: numbers):
        self.l1 = float(l1)

    def __call__(self, x):
        return self.l1 * ops.sum(x)  # type: ignore

    def get_config(self):
        return {'l1': self.l1}
