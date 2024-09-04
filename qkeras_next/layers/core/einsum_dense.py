from keras import ops
from keras.api.layers import EinsumDense
from keras.src.layers.core.einsum_dense import _analyze_einsum_string

from ...quantizer import Quantizer
from ...utils.config.quantizer import QuantizerConfig
from .base import QLayerBaseSingleInput


class QEinsumDense(QLayerBaseSingleInput, EinsumDense):
    def __init__(
        self,
        equation,
        output_shape,
        activation=None,
        bias_axes=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        kq_conf: None | QuantizerConfig = None,
        iq_conf: None | QuantizerConfig = None,
        bq_conf: None | QuantizerConfig = None,
        **kwargs,
    ):
        super().__init__(
            equation=equation,
            output_shape=output_shape,
            activation=activation,
            bias_axes=bias_axes,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            lora_rank=None,
            iq_conf=iq_conf,
            **kwargs,
        )

        kq_conf = kq_conf or QuantizerConfig('default', 'weight')
        self._kq = Quantizer(kq_conf, name=f"{self.name}_kq")
        bq_conf = bq_conf or QuantizerConfig('default', 'bias')
        self._bq = None if bias_axes is None else Quantizer(bq_conf, name=f"{self.name}_bq")

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
        qkernel = self.kq(self._kernel, training=training)
        qinputs = self.iq(inputs, training=training)
        x = ops.einsum(self.equation, qinputs, qkernel)
        if self.bias is not None:
            assert self.bq is not None
            x += self.bq(self.bias, training=training)
        if self.activation is not None:
            x = self.activation(x)

        return x

    def _compute_ebops(self, shape):
        # shape = shapes[0]
        bw_inp = self.iq.bits_((1,) + shape[1:])
        bw_ker = self.kq.bits_(ops.shape(self.kernel))
        ebops = ops.sum(ops.einsum(self.equation, bw_inp, bw_ker))
        if self.bq is not None:
            bw_bias = self.bq.bits_(ops.shape(self.bias))
            size = ops.cast(ops.prod(shape[1:]), self.dtype)
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
