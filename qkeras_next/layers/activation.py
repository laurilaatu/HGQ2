from collections.abc import Callable

from keras import ops
from keras.api.layers import Activation
from keras.src import backend

from ..quantizer import Quantizer, QuantizerConfig
from .core.base import QLayerBaseSingleInput


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
            oq_conf = oq_conf or QuantizerConfig('default', 'table')
            if override_oq_k0_to_0:
                if 'k0' in oq_conf.config:
                    oq_conf.config['k0'] = False
            if not allow_heterogeneous_table:
                oq_conf.config['homogeneous_axis'] = None
                oq_conf.config['heterogeneous_axis'] = ()
            self.oq = Quantizer(oq_conf, name=f'{self.name}_oq')

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
