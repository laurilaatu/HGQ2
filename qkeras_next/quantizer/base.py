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


class QuantizerBase(Layer):
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


class BitwidthMapperBase:
    """Abstract base class for mapping bitwidth tensor to input tensors for HG quantizers."""

    def bw_to_x(self, bw, shape):
        raise NotImplementedError

    def x_to_bw(self, x):
        raise NotImplementedError

    def inference_weight_shape(self, input_shape) -> tuple[int, ...]:
        raise NotImplementedError


def check_axis(axis: Sequence[int], ndim: int):
    axis = [a if a >= 0 else a + ndim for a in axis]
    assert all(0 <= a < ndim for a in axis), f"Invalid axis {axis} for shape {ndim}."
    return axis


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

    def bw_to_x(self, bw, shape):
        return ops.broadcast_to(bw, shape)

    def x_to_bw(self, x):
        return ops.max(ops.abs(x), axis=self.homogeneous_axis, keepdims=True)


DefaultBitwidthMapper(homogeneous_axis=(1,))


def minimal_i_given_xb(x, b):
    eps = 2**-23
    i_pos = ops.ceil(ops.log2(x / (1 - 2**-b) + eps))
    i_neg = ops.ceil(ops.log2(-x))
    return ops.where(x >= 0, i_pos, i_neg)


def minimal_i_given_xf(x, f):
    i_pos = ops.ceil(ops.log2(x + 2**-f))
    i_neg = ops.ceil(ops.log2(-x))
    return ops.where(x >= 0, i_pos, i_neg)


class FixedPointQuantizerKBIBase(QuantizerBase):
    """Abstract base class for all fixed-point quantizers."""

    def __init__(self, k0: numbers | bool, b0: numbers, I0: numbers, i_decay_speed: numbers = np.inf, **kwargs):
        k0 = int(k0)
        assert k0 == 0 or k0 == 1, f"Invalid k0 value {k0}: must be 0 or 1."
        assert b0 >= 0, f"Invalid b0 value {b0}: must be non-negative."
        self._k0 = float(k0)
        self._b0 = float(b0)
        self._I0 = float(I0)
        self._i_decay_speed0 = float(i_decay_speed)
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        bw_shape = self.bw_mapper.inference_weight_shape(input_shape)

        init_k = keras.initializers.Constant(self._k0)
        init_b = keras.initializers.Constant(self._b0)
        init_i = keras.initializers.Constant(self._I0)
        self._k = self.add_weight(name="k", shape=bw_shape, initializer=init_k, trainable=False)
        self._b = self.add_weight(name="b", shape=bw_shape, initializer=init_b, trainable=True, constraint=keras.constraints.NonNeg())
        self._I = self.add_weight(name="I", shape=bw_shape, initializer=init_i, trainable=True)

    @property
    def k(self):
        return ops.cast(self._k, self.dtype)

    @property
    def B(self):
        return round_conv(ops.cast(self._b, self.dtype))

    @property
    def I(self):
        return round_conv(ops.cast(self._I, self.dtype))

    @property
    def i(self):
        return self.I - self.k

    @property
    def f(self):
        return self.B - self.I

    @property
    def b(self):
        return self.B - self.k


class FixedPointQuantizerKBI(FixedPointQuantizerKBIBase):
    """Fixed-point quantizer with k, b, and I parameters."""

    def __init__(self, round_mode: str, overflow_mode: str, k0: numbers | bool, b0: numbers, I0: numbers, i_decay_speed: numbers, **kwargs):
        super().__init__(k0=k0, b0=b0, I0=I0, i_decay_speed=i_decay_speed, **kwargs)
        self._round_mode = round_mode.upper()
        self._overflow_mode = overflow_mode.upper()

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
            self._I.trainable = False
            init = keras.initializers.Constant(self._i_decay_speed0)
            self._i_decay_speed = self.add_weight(name="i_decay_speed", shape=(), initializer=init, trainable=False)

    @property
    def i_decay_speed(self):
        return self._i_decay_speed

    def call(self, inputs, training=None):
        if training and self.overflow_mode == 'WRAP':
            b = self.B
            xr = self.bw_mapper.x_to_bw(inputs)
            _new_I = minimal_i_given_xb(xr, b) + self.k  # type: ignore
            new_I = ops.stop_gradient(ops.maximum((self._I - self.i_decay_speed), _new_I))
            self._I.assign(new_I)

        k = self.bw_mapper.bw_to_x(self.k, ops.shape(inputs))
        i = self.bw_mapper.bw_to_x(self.i, ops.shape(inputs))
        f = self.bw_mapper.bw_to_x(self.f, ops.shape(inputs))

        return self.stateless_quantizer_call(inputs, k, i, f, training)

    def __repr__(self) -> str:

        cls_name = self.__class__.__name__
        round_str = f"round_mode={self.round_mode}"
        overflow_str = f"overflow_mode={self.overflow_mode}"
        if not self.built:
            return f"{cls_name}(name={self.name}, {round_str}, {overflow_str})"
        k, b, I = self.k, self.B, self.I
        kstd, bstd, Istd = np.std(k), np.std(b), np.std(I)  # type: ignore
        kmean, bmean, Imean = np.mean(k), np.mean(b), np.mean(I)  # type: ignore
        name_str = f"name={self.name}"
        k_str = f"k={kmean:.2f}±{kstd:.2f}"
        b_str = f"b={bmean:.2f}±{bstd:.2f}"
        I_str = f"I={Imean:.2f}±{Istd:.2f}"

        return f"{cls_name}({name_str}, {k_str}, {b_str}, {I_str}, {round_str}, {overflow_str})"


class FixedPointQuantizerKIFBase(QuantizerBase):
    """Abstract base class for all fixed-point quantizers."""

    def __init__(self, k0: numbers | bool, i0: numbers, f0: numbers, i_decay_speed0: numbers = np.inf, **kwargs):
        k0 = int(k0)
        assert k0 == 0 or k0 == 1, f"Invalid k0 value {k0}: must be 0 or 1."
        assert i0 + f0 >= 0, f"Invalid b0=i0+f0 value {i0 + f0}: must be non-negative."
        self._k0 = float(k0)
        self._i0 = float(i0)
        self._f0 = float(f0)
        self._i_decay_speed0 = float(i_decay_speed0)
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        bw_shape = self.bw_mapper.inference_weight_shape(input_shape)

        init_k = keras.initializers.Constant(self._k0)
        init_i = keras.initializers.Constant(self._i0)
        init_f = keras.initializers.Constant(self._f0)

        self._k = self.add_weight(name="k", shape=bw_shape, initializer=init_k, trainable=False)
        self._i = self.add_weight(name="i", shape=bw_shape, initializer=init_i, trainable=True)
        self._f = self.add_weight(name="f", shape=bw_shape, initializer=init_f, trainable=True)

    @property
    def k(self):
        return ops.cast(self._k, self.dtype)

    @property
    def i(self):
        _i = ops.cast(self._i, self.dtype)
        _f = ops.cast(self._f, self.dtype)
        return round_conv(ops.maximum(-_f, _i))  # type: ignore # i+f >= 0

    @property
    def f(self):
        return round_conv(ops.cast(self._f, self.dtype))

    @property
    def b(self):
        return self.i + self.f

    @property
    def I(self):
        return self.i + self.k

    @property
    def i_decay_speed(self):
        return ops.cast(self._i_decay_speed, self.dtype)


class FixedPointQuantizerKIF(FixedPointQuantizerKIFBase):
    """Fixed-point quantizer with k, b, and I parameters."""

    def __init__(self, round_mode: str, overflow_mode: str, k0: numbers | bool, i0: numbers, f0: numbers, i_decay_speed: numbers, **kwargs):
        super().__init__(k0=k0, i0=i0, f0=f0, i_decay_speed0=i_decay_speed, **kwargs)
        self._round_mode = round_mode.upper()
        self._overflow_mode = overflow_mode.upper()

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
            self._i.trainable = False
            init = keras.initializers.Constant(self._i_decay_speed0)
            self._i_decay_speed = self.add_weight(name="i_decay_speed", shape=(), initializer=init, trainable=False)

    @property
    def i_decay_speed(self):
        return ops.cast(self._i_decay_speed, self.dtype)

    def call(self, inputs, training=None):
        if training and self.overflow_mode == 'WRAP':
            f = self.f
            xr = self.bw_mapper.x_to_bw(inputs)
            new_i = (xr, f)
            self._i.assign(ops.maximum((self._i._value - self.i_decay_speed), new_i))

        k = self.k
        i = self.i
        f = self.f
        k = self.bw_mapper.bw_to_x(k, ops.shape(inputs))
        i = self.bw_mapper.bw_to_x(i, ops.shape(inputs))
        f = self.bw_mapper.bw_to_x(f, ops.shape(inputs))

        return self.stateless_quantizer_call(inputs, k, i, f, training)

    def __repr__(self) -> str:
        k, i, f = self.k, self.i, self.f
        kstd, istd, fstd = np.std(k), np.std(i), np.std(f)  # type: ignore
        kmean, imean, fmean = np.mean(k), np.mean(i), np.mean(f)  # type: ignore
        return f"{self.__class__.__name__}(k={kmean:.2f}±{kstd:.2f}, i={imean:.2f}±{istd:.2f}, f={fmean:.2f}±{fstd:.2f}, {self.round_mode}, {self.overflow_mode})"
