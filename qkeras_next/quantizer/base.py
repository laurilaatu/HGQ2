import keras
from keras import ops
from keras.src.layers import Layer


class QuantizerBase(Layer):
    """Abstract base class for all quantizers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def range(self) -> tuple[float, float] | None:
        """Return the range of the quantizer in the form of (min, max). If the quantizer has no defined range, return None."""
        raise NotImplementedError

    @property
    def scale(self):
        raise NotImplementedError

    @property
    def zero_point(self):
        raise NotImplementedError

    @property
    def is_heterogeneous(self) -> bool:
        """Return True if the quantizer is heterogeneous in precision."""
        ...

    def __repr__(self) -> str:
        ...
