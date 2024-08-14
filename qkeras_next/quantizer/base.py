import random
import typing
from collections.abc import Sequence

import keras
import numpy as np
from keras import ops
from keras.src.layers import Layer

numbers = int | float | np.integer | np.floating


@ops.custom_gradient
def round_conv(x):
    qx = ops.round(x)

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        return upstream
    return qx, grad


class BitwidthMapperBase:
    """Abstract base class for mapping bitwidth tensor to input tensors for HG quantizers."""

    def bw_to_x(self, bw, x_shape):
        raise NotImplementedError

    def x_to_bw(self, x):
        raise NotImplementedError

    def inference_weight_shape(self, input_shape) -> tuple[int, ...]:
        raise NotImplementedError


def check_axis(axis: Sequence[int], ndim: int):
    axis = [a if a >= 0 else a + ndim for a in axis]
    assert all(0 <= a < ndim for a in axis), f"Invalid axis {axis} for shape {ndim}."
    return axis


class TrainableQuantizerBase(Layer):
    """Abstract base class for all quantizers."""

    def __init__(self, **kwargs):
        homogeneous_axis = kwargs.pop("homogeneous_axis", None) or kwargs.pop("skip_axis", ())
        heterogeneous_axis = kwargs.pop("heterogeneous_axis", None) or kwargs.pop("quantize_axis", None)
        bw_mapper: BitwidthMapperBase = kwargs.pop("bw_mapper", None) or DefaultBitwidthMapper(heterogeneous_axis, homogeneous_axis)
        self.bw_mapper = bw_mapper
        self._seed = kwargs.pop("seed", int(np.random.randint(0, 2**32)))
        super().__init__(**kwargs)
        self.supports_masking = True

    @property
    def scale(self):
        return 1.

    @property
    def zero_point(self):
        return 0.

    def call(self, inputs, training=None):
        ...

    def __repr__(self) -> str:
        ...

    def quantize(self, mode):
        raise ValueError("Quantize method is built-in for keras v3. To avoid name conflicts, please use stateless_quantizer_call method instead.")

    def stateless_quantizer_call(self, *args, **kwargs):
        return self.stateless_quantizer(*args, **kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    @property
    def bits(self):
        raise NotImplementedError


class DefaultBitwidthMapper(BitwidthMapperBase):
    """Default bitwidth mapper for HG quantizers."""

    def __init__(self, heterogeneous_axis: Sequence[int] | None = None, homogeneous_axis: Sequence[int] | None = None, **kwargs):
        super().__init__(**kwargs)
        assert (heterogeneous_axis is None) ^ (homogeneous_axis is None), "Only one of quantize_dims and skip_dims can be specified."
        self.heterogeneous_axis = heterogeneous_axis
        self.homogeneous_axis = homogeneous_axis

    def inference_weight_shape(self, input_shape):
        N = len(input_shape)
        axis = np.arange(N)
        if self.heterogeneous_axis is not None:
            self.heterogeneous_axis = check_axis(self.heterogeneous_axis, N)  # type: ignore
            self.homogeneous_axis = tuple(np.setdiff1d(axis, self.heterogeneous_axis))
        elif self.homogeneous_axis is not None:
            self.homogeneous_axis = check_axis(self.homogeneous_axis, N)  # type: ignore
            self.heterogeneous_axis = tuple(np.setdiff1d(axis, self.homogeneous_axis))

        weight_shape = [1] * N
        for i in self.heterogeneous_axis:  # type: ignore
            assert input_shape[i] is not None, f"Unable to heterogeneously quantize axis {i} with unknown shape. Input shape: {input_shape}."
            weight_shape[i] = input_shape[i]

        return tuple(weight_shape)

    def bw_to_x(self, bw, x_shape):
        return ops.broadcast_to(bw, x_shape)

    def x_to_bw(self, x):
        return ops.max(ops.abs(x), axis=self.homogeneous_axis, keepdims=True)
