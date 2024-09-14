import keras
import numpy as np
from keras import ops
from keras.api.constraints import Constraint
from keras.api.initializers import Constant, Initializer
from keras.api.regularizers import Regularizer
from keras.src import backend

from ..utils.constraints import Min
from .base import TrainableQuantizerBase, numbers
from .fixed_point_ops import round_conv
from .float_point_ops import float_decompose, float_quantize


class FloatPointQuantizer(TrainableQuantizerBase):
    def __init__(
        self,
        m0: numbers | Initializer,
        e0: numbers | Initializer,
        e00: numbers | Initializer = 0,
        mc: Constraint | None = Min(-1),
        ec: Constraint | None = keras.constraints.NonNeg(),
        e0c: Constraint | None = None,
        mr: Regularizer | None = None,
        er: Regularizer | None = None,
        e0r: Regularizer | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._m0 = m0 if isinstance(m0, Initializer) else Constant(float(m0))
        self._e0 = e0 if isinstance(e0, Initializer) else Constant(float(e0))
        self._e00 = e00 if isinstance(e00, Initializer) else Constant(float(e00))
        self.m_constraint = mc
        self.e_constraint = ec
        self.e0_constraint = e0c
        self.m_regularizer = mr
        self.e_regularizer = er
        self.e0_regularizer = e0r

    def build(self, input_shape):
        super().build(input_shape)
        bw_shape = self.bw_mapper.inference_weight_shape(input_shape)
        self._m = self.add_weight(
            name="m",
            shape=bw_shape,
            initializer=self._m0,
            trainable=True,
            constraint=self.m_constraint,
            regularizer=self.m_regularizer
        )
        self._e = self.add_weight(
            name="e",
            shape=bw_shape,
            initializer=self._e0,
            trainable=True,
            constraint=self.e_constraint,
            regularizer=self.e_regularizer
        )
        self._e0 = self.add_weight(
            name="e0",
            shape=bw_shape,
            initializer=self._e00,
            trainable=True,
            constraint=self.e0_constraint,
            regularizer=self.e0_regularizer
        )
        super().build(input_shape)

    @property
    def m(self):
        return round_conv(ops.cast(self._m, ops.dtype(self._m)))  # type: ignore

    @property
    def e(self):
        return round_conv(ops.cast(self._e, ops.dtype(self._e)))  # type: ignore

    @property
    def e0(self):
        return round_conv(ops.cast(self._e0, ops.dtype(self._e0)))  # type: ignore

    @property
    def bits(self):
        return self.m + self.e + 1.  # type: ignore

    @property
    def min(self):
        return -self.max

    @property
    def max(self):
        return 2.**(2**(self.e - 1.) + self.e0 - 1) * (2 - 2.**-self.m)  # type: ignore

    @property
    def epsilon(self):
        return 2.**(-2**(self.e - 1) + self.e0 + 1) * (2.**-self.m)  # type: ignore

    def call(self, inputs, training=None):
        m = self.bw_mapper.bw_to_x(self.m, ops.shape(inputs))
        e = self.bw_mapper.bw_to_x(self.e, ops.shape(inputs))
        e0 = self.bw_mapper.bw_to_x(self.e0, ops.shape(inputs))
        return float_quantize(inputs, m, e, e0)

    def __repr__(self):
        if not self.built:
            return f"{self.__class__.__name__}(name={self.name}, built=False)"
        mstd, estd, e0std = float(ops.std(self.m)), float(ops.std(self.e)), float(ops.std(self.e0))  # type: ignore
        mmean, emean, e0mean = float(ops.mean(self.m)), float(ops.mean(self.e)), float(ops.mean(self.e0))  # type: ignore
        mstr = f"{mmean:.2f}±{mstd:.2f}"
        estr = f"{emean:.2f}±{estd:.2f}"
        e0str = f"{e0mean:.2f}±{e0std:.2f}"
        return f"{self.__class__.__name__}(m={mstr}, e={estr}, e0={e0str}, name={self.name})"
