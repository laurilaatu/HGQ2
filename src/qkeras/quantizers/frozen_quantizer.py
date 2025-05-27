from keras import backend, initializers, ops
from keras.constraints import Constraint
from keras.regularizers import Regularizer

from hgq.constraints import MinMax
from hgq.quantizer.internal import FixedPointQuantizerKBI, numbers
from hgq.utils.misc import gather_vars_to_kwargs


class FrozenFixedPointQuantizer(FixedPointQuantizerKBI):
    """Abstract base class for all fixed-point quantizers."""

    def __init__(
        self,
        k0: numbers | bool,
        b0: numbers,
        i0: numbers,
        round_mode: str,
        overflow_mode: str,
        bc: Constraint | None = MinMax(0, 12),
        ic: Constraint | None = None,
        br: Regularizer | None = None,
        ir: Regularizer | None = None,
        **kwargs,
    ):
        trainable = kwargs.pop('trainable', False)
        assert not trainable, 'FrozenFixedPointQuantizer does not support trainable=True'

        kwargs = gather_vars_to_kwargs('self')
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.building = True
        super().build(input_shape)
        del self.building

    def add_weight(  # type: ignore
        self,
        shape=None,
        initializer=None,
        dtype=None,
        trainable=True,
        autocast=True,
        regularizer=None,
        constraint=None,
        aggregation='mean',
        name=None,
    ):
        """Override add_weight to make it adds a constant weight."""
        if dtype is not None:
            dtype = backend.standardize_dtype(dtype)
        else:
            dtype = self.variable_dtype
        initializer = initializers.get(initializer)
        assert initializer is not None, f'Could not interpret initializer: {initializer}'
        value = initializer(shape, dtype=dtype)
        if ops.size(value) == 1:
            value = ops.squeeze(value)
        return value

    def get_config(self):
        config = super().get_config()
        config.pop('trainable')
        return config
