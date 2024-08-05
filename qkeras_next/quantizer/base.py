import typing
from collections.abc import Sequence

import keras
import numpy as np
from keras import ops
from keras.src.layers import Layer

from .fixed_point_ops import get_fixed_quantizer

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
        homogeneous_axis = kwargs.pop("homogeneous_axis", ())
        heterogeneous_axis = kwargs.pop("heterogeneous_axis", None)
        bw_mapper: BitwidthMapperBase = kwargs.pop("bw_mapper", DefaultBitwidthMapper(heterogeneous_axis, homogeneous_axis))
        self.bw_mapper = bw_mapper
        super().__init__(**kwargs)

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


def minimal_i_given_xb(x, b, symmetric=False):
    eps = 3e-8  # 1/2 fp16 minimal positive
    if symmetric:
        return ops.ceil(ops.log2(ops.abs(x) / (2**-b) + eps))
    i_pos = ops.ceil(ops.log2(x / (1 - 2**-b) + eps))
    i_neg = ops.ceil(ops.log2(-x - eps))
    return ops.where(x >= 0, i_pos, i_neg)


def minimal_i_given_xf(x, f, symmetric=False):
    eps = 3e-8  # 1/2 fp16 minimal positive
    if symmetric:
        return ops.ceil(ops.log2(ops.abs(x) + 2**-f))
    i_pos = ops.ceil(ops.log2(x + 2**-f))
    i_neg = ops.ceil(ops.log2(-x - eps))
    return ops.where(x >= 0, i_pos, i_neg)


class FixedPointQuantizerBase(TrainableQuantizerBase):
    @property
    def round_mode(self):
        return self._round_mode

    @property
    def overflow_mode(self):
        return self._overflow_mode

    def build(self, input_shape):
        super().build(input_shape)
        self.stateless_quantizer = get_fixed_quantizer(self.round_mode, self.overflow_mode)
        if self.overflow_mode == 'WRAP':
            init = keras.initializers.Constant(self._i_decay_speed0)
            self._i_decay_speed = self.add_weight(name="i_decay_speed", shape=(), initializer=init, trainable=False)
        self._symmetric = self.overflow_mode.endswith('SYM')

    @property
    def symmetric(self):
        return self._symmetric

    @property
    def i_decay_speed(self):
        return self._i_decay_speed

    @property
    def kif(self):
        raise NotImplementedError

    @property
    def k(self):
        raise NotImplementedError

    @property
    def b(self):
        raise NotImplementedError

    @property
    def i(self):
        raise NotImplementedError

    @property
    def f(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        if self.built:
            k, i, f = self.k, self.i, self.f
            kstd, istd, fstd = np.std(k), np.std(i), np.std(f)  # type: ignore
            kmean, imean, fmean = np.mean(k), np.mean(i), np.mean(f)  # type: ignore
            kstr = f"{kmean:.2f}±{kstd:.2f}"
            istr = f"{imean:.2f}±{istd:.2f}"
            fstr = f"{fmean:.2f}±{fstd:.2f}"
            return f"{self.__class__.__name__}(k={kstr}, i={istr}, f={fstr}, {self.round_mode}, {self.overflow_mode})"
        return f"{self.__class__.__name__}({self.round_mode}, {self.overflow_mode}, UNBUILT)"

    def get_minimum_i(self, inputs):
        raise NotImplementedError

    def call(self, inputs, training=None):
        if training and self.overflow_mode == 'WRAP':
            _new_i = self.get_minimal_i(inputs)
            new_i = ops.stop_gradient(ops.maximum((self._i - self.i_decay_speed), _new_i))
            self._i.assign(new_i)

        k, i, f = self.kif
        k = self.bw_mapper.bw_to_x(k, ops.shape(inputs))
        i = self.bw_mapper.bw_to_x(i, ops.shape(inputs))
        f = self.bw_mapper.bw_to_x(f, ops.shape(inputs))

        return self.stateless_quantizer_call(inputs, k, i, f, training)


class FixedPointQuantizerKBI(FixedPointQuantizerBase):
    """Abstract base class for all fixed-point quantizers."""

    def __init__(self, k0: numbers | bool, b0: numbers, i0: numbers, round_mode: str, overflow_mode: str, i_decay_speed: numbers = np.inf, **kwargs):
        k0 = int(k0)
        assert k0 == 0 or k0 == 1, f"Invalid k0 value {k0}: must be 0 or 1."
        assert b0 >= 0, f"Invalid b0 value {b0}: must be non-negative."
        self._k0 = float(k0)
        self._b0 = float(b0)
        self._i0 = float(i0)
        self._i_decay_speed0 = float(i_decay_speed)
        self._round_mode = round_mode.upper()
        self._overflow_mode = overflow_mode.upper()
        super().__init__(**kwargs)

    def build(self, input_shape):
        bw_shape = self.bw_mapper.inference_weight_shape(input_shape)

        init_k = keras.initializers.Constant(self._k0)
        init_b = keras.initializers.Constant(self._b0)
        init_i = keras.initializers.Constant(self._i0)
        self._k = self.add_weight(name="k", shape=bw_shape, initializer=init_k, trainable=False, dtype='uint8')
        self._b = self.add_weight(name="b", shape=bw_shape, initializer=init_b, trainable=True, constraint=keras.constraints.NonNeg())
        i_trainable = self.overflow_mode != 'WRAP'
        self._i = self.add_weight(name="i", shape=bw_shape, initializer=init_i, trainable=i_trainable)
        super().build(input_shape)

    @property
    def k(self):
        return ops.cast(self._k, self.dtype)

    @property
    def b(self):
        return round_conv(ops.cast(self._b, self.dtype))

    @property
    def i(self):
        return round_conv(ops.cast(self._i, self.dtype))

    @property
    def f(self):
        return self.b - self.i

    @property
    def kif(self):
        k = self.k
        b = self.b
        i = self.i
        return k, i, b - i  # type: ignore

    def get_minimal_i(self, inputs):
        xr = self.bw_mapper.x_to_bw(inputs)
        return minimal_i_given_xb(xr, self.b, self.symmetric)


class FixedPointQuantizerKIF(FixedPointQuantizerBase):
    """Abstract base class for all fixed-point quantizers."""

    def __init__(self, k0: numbers | bool, i0: numbers, f0: numbers, round_mode: str, overflow_mode: str, i_decay_speed: numbers = np.inf, **kwargs):
        k0 = int(k0)
        assert k0 == 0 or k0 == 1, f"Invalid k0 value {k0}: must be 0 or 1."
        assert i0 + f0 >= 0, f"Invalid i0+f0 value {i0 + f0}: must be non-negative."
        self._k0 = float(k0)
        self._i0 = float(i0)
        self._f0 = float(f0)
        self._i_decay_speed0 = float(i_decay_speed)
        self._round_mode = round_mode.upper()
        self._overflow_mode = overflow_mode.upper()
        super().__init__(**kwargs)

    def build(self, input_shape):
        bw_shape = self.bw_mapper.inference_weight_shape(input_shape)

        init_k = keras.initializers.Constant(self._k0)
        init_i = keras.initializers.Constant(self._i0)
        init_f = keras.initializers.Constant(self._f0)

        self._k = self.add_weight(name="k", shape=bw_shape, initializer=init_k, trainable=False, dtype='uint8')
        i_trainable = self.overflow_mode != 'WRAP'
        self._i = self.add_weight(name="i", shape=bw_shape, initializer=init_i, trainable=i_trainable)
        self._f = self.add_weight(name="f", shape=bw_shape, initializer=init_f, trainable=True)

        super().build(input_shape)

    @property
    def k(self):
        return ops.cast(self._k, self.dtype)

    @property
    def b(self):
        return self.i + self.f

    @property
    def i(self):
        return round_conv(ops.cast(self._i, self.dtype))

    @property
    def f(self):
        return round_conv(ops.cast(self._f, self.dtype))

    @property
    def kif(self):
        return self.k, self.i, self.f

    def get_minimal_i(self, inputs):
        xr = self.bw_mapper.x_to_bw(inputs)
        return minimal_i_given_xf(xr, self.f, self.symmetric)
