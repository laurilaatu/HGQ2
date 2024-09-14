from keras import ops
from keras.api.layers import Dense

from ...config.quantizer import QuantizerConfig
from ...quantizer import Quantizer
from .base import QLayerBaseSingleInput


class QDense(QLayerBaseSingleInput, Dense):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        kq_conf: None | QuantizerConfig = None,
        iq_conf: None | QuantizerConfig = None,
        bq_conf: None | QuantizerConfig = None,
        **kwargs,
    ):
        super().__init__(
            units=units,
            use_bias=use_bias,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            activity_regularizer=activity_regularizer,
            lora_rank=None,
            iq_conf=iq_conf,
            **kwargs,
        )

        kq_conf = kq_conf or QuantizerConfig('default', 'weight')
        self._kq = Quantizer(kq_conf, name=f"{self.name}_kq")
        bq_conf = bq_conf or QuantizerConfig('default', 'bias')
        self._bq = Quantizer(bq_conf, name=f"{self.name}_bq") if use_bias else None

    @property
    def kq(self):
        return self._kq

    @property
    def bq(self):
        return self._bq

    def build(self, input_shape):
        super().build(input_shape)

        self.kq.build(ops.shape(self._kernel))
        if self.bias is not None:
            assert self.bq is not None
            self.bq.build(ops.shape(self.bias))

    def call(self, inputs, training=None):
        qinputs = self.iq(inputs, training=training)
        qkernel = self.kq(self.kernel)
        x = ops.matmul(qinputs, qkernel)
        if self.bias is not None:
            qbias = self.bq(self.bias)  # type: ignore
            x = ops.add(x, qbias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def _compute_ebops(self, shape):
        bw_inp = self.iq.bits_(shape)
        bw_ker = self.kq.bits_(ops.shape(self.kernel))
        ebops = ops.sum(ops.matmul(bw_inp, bw_ker))
        if self.bq is not None:
            bw_bias = self.bq.bits_(ops.shape(self.bias))
            size = ops.cast(ops.prod(shape), self.dtype)
            ebops = ebops + ops.mean(bw_bias) * size  # type: ignore

        return ebops

    def get_config(self):
        config = super().get_config()
        config.update({
            'kq_conf': self.kq.config,
            'bq_conf': self.bq.config if self.bq is not None else None,
        })
        return config

    @property
    def qkernel(self):
        return self.kq(self._kernel)

    @property
    def qbias(self):
        if self.bias is None:
            return None
        assert self.bq is not None
        return self.bq(self.bias)
