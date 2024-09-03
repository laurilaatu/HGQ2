from collections.abc import Callable, Sequence

from keras import ops
from keras.api.layers import Activation, Softmax
from keras.src import backend

from ..quantizer import Quantizer, QuantizerConfig
from .core.base import QLayerBaseSingleInput
from .ops import QSum


def _large_negative_number(dtype):
    """Return a Large negative number based on dtype."""
    if backend.standardize_dtype(dtype) == "float16":
        return -3e4
    return -1e9


class QUnaryFunctionLUT(Activation, QLayerBaseSingleInput):
    def __init__(
        self,
        activation: Callable | str,
        iq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        enable_out_quantizer=True,
        allow_heterogeneous_table: bool = False,
        override_oq_k0_to_0: bool = False,
        **kwargs
    ):
        act_name = activation.__name__ if isinstance(activation, Callable) else activation
        assert act_name not in ('softmax', 'log_softmax'), f"activation {act_name} is not unary"
        self.enable_out_quantizer = enable_out_quantizer

        super().__init__(activation=activation, iq_conf=iq_conf, **kwargs)

        if enable_out_quantizer:
            oq_conf = oq_conf or QuantizerConfig('default', 'input')
            if override_oq_k0_to_0:
                if 'k0' in oq_conf.config:
                    oq_conf.config['k0'] = False
            if not allow_heterogeneous_table:
                oq_conf.config['homogeneous_axis'] = None
                oq_conf.config['heterogeneous_axis'] = ()
            self.oq = Quantizer(oq_conf)

    def call(self, inputs, training=None):
        qinputs = self.iq(inputs, training=training)
        x = super().call(qinputs)
        if self.enable_out_quantizer:
            x = self.oq(x, training=training)
        if self.enable_ebops and training:
            self._compute_ebops(ops.shape(inputs))
        return x

    def _compute_ebops(self, shape):
        if not self.enable_out_quantizer:
            return
        _shape = (1,) + shape[1:]
        bw_inp = self.iq.bits_(_shape)
        bw_out = self.oq.bits_(_shape)
        # TODO: more realistic cost for lookup tables
        ebops = ops.sum((2.**bw_inp) * bw_out) * 0.01  # type: ignore
        self._ebops.assign(ops.cast(ebops, self._ebops.dtype))  # type: ignore
        self.add_loss(self.beta * ebops)


class QPositiveUnaryFunctionLUT(QUnaryFunctionLUT):
    def __init__(
        self,
        activation: Callable | str,
        iq_conf: QuantizerConfig | None = None,
        oq_conf: QuantizerConfig | None = None,
        allow_heterogeneous_table: bool = False,
        **kwargs
    ):
        assert kwargs.pop('enable_out_quantizer', True), "enable_out_quantizer must be True for QPositiveUnaryFunctionLUT, if set."
        super().__init__(
            activation=activation,
            iq_conf=iq_conf,
            oq_conf=oq_conf,
            enable_out_quantizer=True,
            allow_heterogeneous_table=allow_heterogeneous_table,
            override_oq_k0_to_0=True,
            **kwargs
        )

    def call(self, inputs, training=None):
        x = super().call(inputs, training=training)
        eps = self.oq.epsilon_(ops.shape(inputs))
        return ops.maximum(x, eps)


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
