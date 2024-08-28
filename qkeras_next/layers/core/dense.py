from keras import ops
from keras.api.layers import Dense
from keras.api.saving import deserialize_keras_object, register_keras_serializable, serialize_keras_object

from ...quantizer import Quantizer
from ...utils.config.quantizer import QuantizerConfig
from .base import QLayerBase


@register_keras_serializable(package='qkeras_next')
class QDense(QLayerBase, Dense):
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
        self._kq = Quantizer(kq_conf)
        bq_conf = bq_conf or QuantizerConfig('default', 'bias')
        self._bq = Quantizer(bq_conf) if use_bias else None

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
        if self.enable_ebops and training:
            self._compute_ebops(ops.shape(inputs))
        return x

    def _compute_ebops(self, shape):
        bw_inp = self.iq.bits_((1,) + shape[1:])
        bw_ker = self.kq.bits_(ops.shape(self.kernel))
        ebops = ops.sum(ops.matmul(bw_inp, bw_ker))
        if self.bq is not None:
            bw_bias = self.bq.bits_(ops.shape(self.bias))
            ebops = ebops + ops.mean(bw_bias) * ops.prod(shape[1:])  # type: ignore

        self._ebops.assign(ops.cast(ebops, self._ebops.dtype))  # type: ignore
        self.add_loss(self.beta * ebops)

    def get_config(self):
        config = super().get_config()
        config.update({
            'kq_conf': serialize_keras_object(self.kq.config),
            'bq_conf': serialize_keras_object(self.bq.config) if self.bq is not None else None,
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
