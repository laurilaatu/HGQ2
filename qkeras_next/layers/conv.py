from keras import ops
from keras.api.layers import InputSpec
from keras.api.saving import deserialize_keras_object, register_keras_serializable, serialize_keras_object
from keras.src.layers.convolutional.base_conv import BaseConv
from keras.src.layers.core.einsum_dense import _analyze_einsum_string

from ..quantizer import Quantizer
from ..utils.config.quantizer import QuantizerConfig
from .core.base import QLayerBase


class QBaseConv(QLayerBase, BaseConv):
    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        parallelization_factor=-1,
        kq_conf: None | QuantizerConfig = None,
        iq_conf: None | QuantizerConfig = None,
        bq_conf: None | QuantizerConfig = None,
        **kwargs,
    ):
        super().__init__(
            rank=rank,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            lora_rank=None,
            iq_conf=iq_conf,
            **kwargs,
        )

        kq_conf = kq_conf or QuantizerConfig('default', 'weight')
        self._kq = Quantizer(kq_conf)
        bq_conf = bq_conf or QuantizerConfig('default', 'bias')
        self._bq = Quantizer(bq_conf) if use_bias else None
        self.parallelization_factor = parallelization_factor

    @property
    def kq(self):
        return self._kq

    @property
    def bq(self):
        return self._bq

    def build(self, input_shape):
        super().build(input_shape)
        self.kq.build(ops.shape(self.kernel))
        if self.use_bias:
            assert self.bq is not None
            self.bq.build(ops.shape(self.bias))

    @property
    def kernel(self):
        return self._kernel

    def call(self, inputs, training=None):

        qinputs = self.iq(inputs, training=training)
        qkernel = self.kq(self._kernel, training=training)
        outputs = self.convolution_op(
            qinputs,
            qkernel,
        )
        if self.use_bias:
            if self.data_format == "channels_last":
                bias_shape = (1,) * (self.rank + 1) + (self.filters,)
            else:
                bias_shape = (1, self.filters) + (1,) * self.rank
            qbias = self.bq(self.bias, training=training)  # type: ignore
            bias = ops.reshape(qbias, bias_shape)
            outputs += bias  # type: ignore

        if self.enable_ebops:
            self._compute_ebops(ops.shape(inputs))
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _compute_ebops(self, shape):
        bw_inp = self.iq.bits_((1,) + shape[1:])
        bw_ker = self.kq.bits_(ops.shape(self.kernel))
        if self.parallelization_factor < 0:
            ebops = ops.sum(self.convolution_op(bw_inp, bw_ker))
        else:
            reduce_axis_kernel = tuple(range(0, self.rank + 1))
            if self.data_format == "channels_last":
                reduce_axis_input = reduce_axis_kernel
            else:
                reduce_axis_input = (0,) + tuple(range(2, self.rank + 2))

            bw_inp = ops.max(bw_inp, axis=reduce_axis_input)  # Keep only maximum per channel
            ebops = ops.einsum('c,...co->', bw_inp, bw_ker)

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

    @classmethod
    def from_config(cls, config):
        config = deserialize_keras_object(config)
        return cls(**config)


@register_keras_serializable(package='qkeras_next')
class QConv1D(QBaseConv):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        groups=1,
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
        **kwargs
    ):
        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            kq_conf=kq_conf,
            iq_conf=iq_conf,
            bq_conf=bq_conf,
            **kwargs
        )

    def _compute_causal_padding(self):
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
        if self.data_format == "channels_last":
            causal_padding = [[0, 0], [left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0], [0, 0], [left_pad, 0]]
        return causal_padding

    def call(self, inputs, training=None):
        padding = self.padding
        if self.padding == "causal":
            # Apply causal padding to inputs.
            inputs = ops.pad(inputs, self._compute_causal_padding())
            padding = "valid"

        qinputs = self.iq(inputs, training=training)
        qkernel = self.kq(self._kernel, training=training)

        outputs = ops.conv(
            qinputs,
            qkernel,
            strides=list(self.strides),  # type: ignore
            padding=padding,
            dilation_rate=self.dilation_rate,  # type: ignore
            data_format=self.data_format,
        )

        if self.use_bias:
            if self.data_format == "channels_last":
                bias_shape = (1,) * (self.rank + 1) + (self.filters,)
            else:
                bias_shape = (1, self.filters) + (1,) * self.rank
            bias = ops.reshape(self.bias, bias_shape)
            outputs += bias  # type: ignore
        if self.enable_ebops:
            self._compute_ebops(ops.shape(inputs))

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


@register_keras_serializable(package='qkeras_next')
class QConv2D(QBaseConv):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
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
        **kwargs
    ):
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            kq_conf=kq_conf,
            iq_conf=iq_conf,
            bq_conf=bq_conf,
            **kwargs
        )


@register_keras_serializable(package='qkeras_next')
class QConv3D(QBaseConv):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
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
        **kwargs
    ):
        super().__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            kq_conf=kq_conf,
            iq_conf=iq_conf,
            bq_conf=bq_conf,
            **kwargs
        )
