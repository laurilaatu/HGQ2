from collections.abc import Sequence

from keras import ops
from keras.src import backend

from ..quantizer import QuantizerConfig
from .activation import QUnaryFunctionLUT
from .core import QLayerBaseSingleInput


class QSoftmax(QLayerBaseSingleInput):
    def __init__(
        self,
        axis: int | Sequence[int] = -1,
        iq_conf: None | QuantizerConfig = None,
        stable=False,
        exp_q_conf: None | QuantizerConfig = None,
        inv_q_conf: None | QuantizerConfig = None,
        allow_heterogeneous_table: bool = False,
        **kwargs
    ):
        self.supports_masking = True
        super().__init__(iq_conf=iq_conf, **kwargs)  # type: ignore
        self.stable = stable
        self.axis = tuple(axis) if isinstance(axis, Sequence) else (axis,)

        def _inv(x):
            return 1.0 / x

        self.inv_table = QUnaryFunctionLUT(
            _inv,
            inv_q_conf,
            enable_out_quantizer=True,
            allow_heterogeneous_table=allow_heterogeneous_table
        )
        self.exp_table = QUnaryFunctionLUT(
            ops.exp,
            exp_q_conf,
            enable_out_quantizer=True,
            allow_heterogeneous_table=allow_heterogeneous_table
        )

    def build(self, input_shape):
        self.exp_table.build(input_shape)

        inv_shape = list(input_shape)
        for i in self.axis:
            inv_shape[i] = 1
        self.inv_table.build(tuple(inv_shape))
        super().build(input_shape)

    def call(self, inputs, training=None, mask=None):  # type: ignore
        if self.stable:
            inputs = self.iq(inputs, training=training)
            inputs = inputs - ops.max(inputs, axis=self.axis, keepdims=True)

        exp_inp = self.exp_table(inputs, training=training)

        if mask is not None:
            exp_inp = backend.cast(mask, ops.dtype(inputs)) * exp_inp

        sums = ops.sum(exp_inp, axis=self.axis, keepdims=True)
        divisor = self.inv_table(sums, training=training)

        if training and self.enable_ebops:
            self._compute_ebops(ops.shape(exp_inp), ops.shape(divisor))

        return exp_inp * divisor

    def _compute_ebops(self, shape1, shape2):
        _shape1 = (1,) + shape1[1:]
        _shape2 = (1,) + shape2[1:]

        if self.stable:
            inp_bits = self.iq.bits_(_shape1)
            substract_ebops = 1.55 * ops.sum(inp_bits)  # type: ignore # TODO: better ebops cost model for add and max
        else:
            substract_ebops = 0

        exp_bits = self.exp_table.oq.bits_(_shape1)
        inv_bits = self.inv_table.oq.bits_(_shape2)

        accum_ebops = ops.sum(exp_bits) - ops.sum(ops.min(exp_bits, axis=self.axis))  # type: ignore
        mult_ebops = ops.sum(accum_ebops * inv_bits)

        ebops = substract_ebops + accum_ebops + mult_ebops
        self.add_loss(self.beta * ebops)
        ebops = ebops + self.inv_table.ebops + self.exp_table.ebops
        self._ebops.assign(ops.cast(ebops, self._ebops.dtype))  # type: ignore
