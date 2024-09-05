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
        exp_iq_conf: None | QuantizerConfig = None,
        exp_oq_conf: None | QuantizerConfig = None,
        inv_iq_conf: None | QuantizerConfig = None,
        inv_oq_conf: None | QuantizerConfig = None,
        allow_heterogeneous_table: bool = False,
        **kwargs
    ):
        self.supports_masking = True
        super().__init__(iq_conf=iq_conf, enable_iq=stable, **kwargs)  # type: ignore
        self.stable = stable
        self.axis = tuple(axis) if isinstance(axis, Sequence) else (axis,)
        self._allow_heterogeneous_table = allow_heterogeneous_table

        def _inv(x):
            return 1.0 / x

        self.inv_table = QUnaryFunctionLUT(
            _inv,
            inv_iq_conf,
            inv_oq_conf,
            enable_oq=True,
            allow_heterogeneous_table=allow_heterogeneous_table,
            name=f"{self.name}_inv_table",
            enable_ebops=self.enable_ebops,
            beta0=self._beta0.clone(),
        )
        self.exp_table = QUnaryFunctionLUT(
            ops.exp,
            exp_iq_conf,
            exp_oq_conf,
            enable_oq=True,
            allow_heterogeneous_table=allow_heterogeneous_table,
            name=f"{self.name}_exp_table",
            enable_ebops=self.enable_ebops,
            beta0=self._beta0.clone(),
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

        return exp_inp * divisor

    def _compute_ebops(self, shape):
        shape2 = tuple(1 if i in self.axis else s for i, s in enumerate(shape))

        if self.stable:
            inp_bits = self.iq.bits_(shape)
            substract_ebops = 1.55 * ops.sum(inp_bits)  # type: ignore # TODO: better ebops cost model for add and max
        else:
            substract_ebops = 0

        exp_bits = self.exp_table.oq.bits_(shape)
        inv_bits = self.inv_table.oq.bits_(shape2)

        accum_ebops = ops.sum(exp_bits) - ops.sum(ops.min(exp_bits, axis=self.axis))  # type: ignore
        mult_ebops = ops.sum(accum_ebops * inv_bits)

        ebops = substract_ebops + accum_ebops + mult_ebops
        return ebops

    def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "stable": self.stable,
            "exp_oq_conf": self.exp_table.oq.config,
            "exp_iq_conf": self.exp_table.iq.config,
            "inv_oq_conf": self.inv_table.oq.config,
            "inv_iq_conf": self.inv_table.iq.config,
            "allow_heterogeneous_table": self._allow_heterogeneous_table
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
