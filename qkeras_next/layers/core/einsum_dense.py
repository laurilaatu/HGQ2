from keras import ops
from keras.api.layers import EinsumDense
from keras.api.saving import deserialize_keras_object, register_keras_serializable, serialize_keras_object
from keras.src.layers.core.einsum_dense import _analyze_einsum_string

from ...quantizer import Quantizer
from ...utils.config.quantizer import QuantizerConfig
from .base import QLayerBase


@register_keras_serializable(package='qkeras_next')
class QEinsumDense(QLayerBase, EinsumDense):
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
        self.kq = Quantizer(kq_conf)
        bq_conf = bq_conf or QuantizerConfig('default', 'bias')
        self.bq = None if bias_axes is None else Quantizer(bq_conf)

    def build(self, input_shape):
        super().build(input_shape)
        shape_data = _analyze_einsum_string(
            self.equation,
            self.bias_axes,
            input_shape,
            self.partial_output_shape,
        )
        kernel_shape, bias_shape, full_output_shape = shape_data
        self.full_output_shape = tuple(full_output_shape)

        self._kernel = self.add_weight(
            name="kernel",
            shape=tuple(kernel_shape),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        self.kq.build(kernel_shape)

        if bias_shape is not None:
            self._bias = self.add_weight(
                name="bias",
                shape=tuple(bias_shape),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
            assert self.bq is not None
            self.bq.build(bias_shape)
        else:
            self._bias = None
        self.built = True

    @property
    def kernel(self):
        return self.kq(self._kernel)

    @property
    def bias(self):
        if self._bias is None:
            return None
        assert self.bq is not None
        return self.bq(self._bias)

    def call(self, inputs, training=None):
        qkernel = self.kq(self._kernel, training=training)
        qinputs = self.iq(inputs, training=training)
        x = ops.einsum(self.equation, qinputs, qkernel)
        if self._bias is not None:
            assert self.bq is not None
            x += self.bq(self._bias, training=training)
        if self.activation is not None:
            x = self.activation(x)

        if self.enable_ebops:
            shape = tuple(ops.shape(inputs))
            bw_inp = self.iq.bits_((1,) + shape[1:])
            bw_ker = self.kq.bits_(ops.shape(self._kernel))
            ebops = ops.sum(ops.einsum(self.equation, bw_inp, bw_ker))
            if self.bq is not None:
                bw_bias = self.bq.bits_(bw_inp)
                ebops = ebops + ops.sum(bw_bias)  # type: ignore

            self.ebops.assign(ops.cast(ebops, self.ebops.dtype))
            self.add_loss(self.beta * ebops)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'kq_conf': serialize_keras_object(self.kq.config),
            'bq_conf': serialize_keras_object(self.bq.config) if self.bq is not None else None,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config = deserialize_keras_object(config)
        return cls(**config)
