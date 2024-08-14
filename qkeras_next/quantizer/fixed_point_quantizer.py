import keras
import numpy as np
from keras import ops
from keras.api.constraints import Constraint
from keras.api.initializers import Constant, Initializer
from keras.api.regularizers import Regularizer
from keras.src import backend

from ..utils.constraints import MinMax
from .base import TrainableQuantizerBase, numbers
from .fixed_point_ops import get_fixed_quantizer, round_conv


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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._seed_gen = None

    @property
    def round_mode(self) -> str:
        return self._round_mode

    @property
    def overflow_mode(self) -> str:
        return self._overflow_mode

    @property
    def seed_gen(self):
        return self._seed_gen

    def build(self, input_shape):
        super().build(input_shape)
        if self.round_mode.startswith('S_'):
            self._seed_gen = keras.random.SeedGenerator(self._seed)
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

    @property
    def bits(self):
        return self.b

    def __repr__(self) -> str:
        if not self.built:
            return f"{self.__class__.__name__}({self.round_mode}, {self.overflow_mode}, name={self.name}, built=False)"
        k, i, f = self.k, self.i, self.f
        kstd, istd, fstd = float(ops.std(k)), float(ops.std(i)), float(ops.std(f))  # type: ignore
        kmean, imean, fmean = float(ops.mean(k)), float(ops.mean(i)), float(ops.mean(f))  # type: ignore
        kstr = f"{kmean:.2f}±{kstd:.2f}"
        istr = f"{imean:.2f}±{istd:.2f}"
        fstr = f"{fmean:.2f}±{fstd:.2f}"
        return f"{self.__class__.__name__}(k={kstr}, i={istr}, f={fstr}, {self.round_mode}, {self.overflow_mode}, name={self.name})"

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

        return self.stateless_quantizer_call(inputs, k, i, f, training, self.seed_gen)


class FixedPointQuantizerKBI(FixedPointQuantizerBase):
    """Abstract base class for all fixed-point quantizers."""

    def __init__(
        self,
        k0: numbers | bool | Initializer,
        b0: numbers | Initializer,
        i0: numbers | Initializer,
        round_mode: str,
        overflow_mode: str,
        bc: Constraint | None = MinMax(0, 12),
        ic: Constraint | None = None,
        br: Regularizer | None = None,
        ir: Regularizer | None = None,
        i_decay_speed: numbers = np.inf,
        **kwargs
    ):
        if not isinstance(k0, Initializer) and not isinstance(b0, Initializer) and not isinstance(i0, Initializer):
            k0 = int(k0)
            assert k0 == 0 or k0 == 1, f"Invalid k0 value {k0}: must be 0 or 1."
            assert b0 >= 0, f"Invalid b0 value {b0}: must be non-negative."
        self._k0 = k0 if isinstance(k0, Initializer) else Constant(float(k0))
        self._b0 = b0 if isinstance(b0, Initializer) else Constant(float(b0))
        self._i0 = i0 if isinstance(i0, Initializer) else Constant(float(i0))
        self._i_decay_speed0 = i_decay_speed if isinstance(i_decay_speed, Initializer) else Constant(float(i_decay_speed))
        self._round_mode = round_mode.upper()
        self._overflow_mode = overflow_mode.upper()
        self.b_constraint = bc
        self.i_constraint = ic
        self.b_regularizer = br
        self.i_regularizer = ir

        self.validate_config()
        super().__init__(**kwargs)

    def build(self, input_shape):
        bw_shape = self.bw_mapper.inference_weight_shape(input_shape)

        self._k = self.add_weight(
            name="k",
            shape=bw_shape,
            initializer=self._k0,
            trainable=False,
            dtype='uint8'
        )
        self._b = self.add_weight(
            name="b",
            shape=bw_shape,
            initializer=self._b0,
            trainable=True,
            constraint=self.b_constraint,
            regularizer=self.b_regularizer
        )
        i_trainable = self.overflow_mode != 'WRAP'
        self._i = self.add_weight(
            name="i",
            shape=bw_shape,
            initializer=self._i0,
            trainable=i_trainable,
            constraint=self.i_constraint,
            regularizer=self.i_regularizer
        )
        super().build(input_shape)

    @property
    def k(self):
        return backend.convert_to_tensor(self._k, self.dtype)

    @property
    def b(self):
        return round_conv(backend.convert_to_tensor(self._b))

    @property
    def i(self):
        return round_conv(backend.convert_to_tensor(self._i))

    @property
    def f(self):
        return self.b - self.i  # type: ignore

    @property
    def kif(self):
        k = self.k
        b = self.b
        i = self.i
        return k, i, b - i  # type: ignore

    def get_minimal_i(self, inputs):
        xr = self.bw_mapper.x_to_bw(inputs)
        return minimal_i_given_xb(xr, self.b, self.symmetric)

    def validate_config(self):
        assert self.overflow_mode in ('WRAP', 'SAT', 'SYM', 'SYM_SAT')
        assert self.round_mode in ('RND', 'TRN', 'RND_CONV', 'S_RND', 'S_RND_CONV')
        assert self.b_constraint is None or isinstance(self.b_constraint, Constraint)
        assert self.i_constraint is None or isinstance(self.i_constraint, Constraint)
        assert self.b_regularizer is None or isinstance(self.b_regularizer, Regularizer)
        assert self.i_regularizer is None or isinstance(self.i_regularizer, Regularizer)
        if self.overflow_mode == 'WRAP':
            assert self.i_regularizer is None, "Regularization of i is not supported with overflow_mode='WRAP'."
            assert self.i_constraint is None, "Constraint of i is not supported with overflow_mode='WRAP'."


class FixedPointQuantizerKIF(FixedPointQuantizerBase):
    """Abstract base class for all fixed-point quantizers."""

    def __init__(
        self,
        k0: numbers | bool,
        i0: numbers,
        f0: numbers,
        round_mode: str,
        overflow_mode: str,
        ic: Constraint | None = None,
        fc: Constraint | None = None,
        ir: Regularizer | None = None,
        fr: Regularizer | None = None,
        i_decay_speed: numbers = np.inf,
        **kwargs
    ):
        if not isinstance(k0, Initializer) and not isinstance(i0, Initializer) and not isinstance(f0, Initializer):
            k0 = int(k0)
            assert k0 == 0 or k0 == 1, f"Invalid k0 value {k0}: must be 0 or 1."
            assert i0 + f0 >= 0, f"Invalid i0+f0 value {i0 + f0}: must be non-negative."
        self._k0 = k0 if isinstance(k0, Initializer) else Constant(float(k0))
        self._i0 = i0 if isinstance(i0, Initializer) else Constant(float(i0))
        self._f0 = f0 if isinstance(f0, Initializer) else Constant(float(f0))
        self._i_decay_speed0 = i_decay_speed if isinstance(i_decay_speed, Initializer) else Constant(float(i_decay_speed))
        self._round_mode = round_mode.upper()
        self._overflow_mode = overflow_mode.upper()
        self.i_constraint = ic
        self.f_constraint = fc
        self.i_regularizer = ir
        self.f_regularizer = fr
        self.validate_config()
        super().__init__(**kwargs)

    def build(self, input_shape):
        bw_shape = self.bw_mapper.inference_weight_shape(input_shape)

        self._k = self.add_weight(
            name="k",
            shape=bw_shape,
            initializer=self._k0,
            trainable=False,
            dtype='uint8'
        )
        i_trainable = self.overflow_mode != 'WRAP'
        self._i = self.add_weight(
            name="i",
            shape=bw_shape,
            initializer=self._i0,
            trainable=i_trainable,
            constraint=self.i_constraint,
            regularizer=self.i_regularizer
        )
        self._f = self.add_weight(
            name="f",
            shape=bw_shape,
            initializer=self._f0,
            trainable=True,
            constraint=self.f_constraint,
            regularizer=self.f_regularizer
        )

        super().build(input_shape)

    @property
    def k(self):
        return ops.cast(self._k, self.dtype)

    @property
    def b(self):
        return self.i + self.f  # type: ignore

    @property
    def i(self):
        return round_conv(backend.convert_to_tensor(self._i))

    @property
    def f(self):
        return round_conv(backend.convert_to_tensor(self._f))

    @property
    def kif(self):
        return self.k, self.i, self.f

    def get_minimal_i(self, inputs):
        xr = self.bw_mapper.x_to_bw(inputs)
        return minimal_i_given_xf(xr, self.f, self.symmetric)

    def validate_config(self):
        assert self.overflow_mode in ('WRAP', 'SAT', 'SYM', 'SYM_SAT')
        assert self.round_mode in ('TRN', 'RND', 'RND_CONV', 'S_RND', 'S_RND_CONV')
        assert self.f_constraint is None or isinstance(self.f_constraint, Constraint)
        assert self.i_constraint is None or isinstance(self.i_constraint, Constraint)
        assert self.f_regularizer is None or isinstance(self.f_regularizer, Regularizer)
        assert self.i_regularizer is None or isinstance(self.i_regularizer, Regularizer)
        if self.overflow_mode == 'WRAP':
            assert self.i_regularizer is None, "Regularization of i is not supported with overflow_mode='WRAP'."
            assert self.i_constraint is None, "Constraint of i is not supported with overflow_mode='WRAP'."
