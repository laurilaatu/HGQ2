from keras.api.layers import BatchNormalization
from keras.src import backend, ops
from keras.src.backend import standardize_dtype

from ..config.quantizer import QuantizerConfig
from ..quantizer import Quantizer
from .core.base import QLayerBaseSingleInput


class QBatchNormalization(QLayerBaseSingleInput, BatchNormalization):

    @property
    def beta(self):
        if self._beta is None:
            return backend.convert_to_tensor(0)
        return ops.cast(self._beta, ops.dtype(self._beta))

    @beta.setter
    def beta(self, value):
        """Workaround for setting the batch normalization beta during building."""
        if self.built:
            raise AttributeError("Cannot set beta after build.")
        self.bn_beta = value

    @property
    def gamma(self):
        raise AttributeError("Use `bn_gamma` instead of `gamma`.")

    @gamma.setter
    def gamma(self, value):
        """Workaround for setting the batch normalization gamma during building."""
        if self.built:
            raise AttributeError("Cannot set gamma after build.")
        self.bn_gamma = value

    def __init__(
        self,
        axis=-1,
        momentum=0.99,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        synchronized=False,
        kq_conf: None | QuantizerConfig = None,
        iq_conf: None | QuantizerConfig = None,
        bq_conf: None | QuantizerConfig = None,
        **kwargs,
    ):
        super().__init__(
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            synchronized=synchronized,
            iq_conf=iq_conf,
            **kwargs
        )
        kq_conf = kq_conf or QuantizerConfig('default', 'weight')
        self._kq = Quantizer(kq_conf, name=f"{self.name}_kq")
        bq_conf = bq_conf or QuantizerConfig('default', 'bias')
        self._bq = Quantizer(bq_conf, name=f"{self.name}_bq")

    @property
    def kq(self):
        return self._kq

    @property
    def bq(self):
        return self._bq

    def build(self, input_shape):
        super().build(input_shape)
        if self.bq is not None:
            self.bq.build(ops.shape(self.bn_beta))
        if self.kq is not None:
            self.kq.build(ops.shape(self.bn_gamma))
        shape = [1] * len(input_shape)
        shape[self.axis] = input_shape[self.axis]
        self._shape = tuple(shape)
        self.bn_beta.name = 'bn_beta'
        self.bn_beta.path = self.bn_beta.path.replace('/beta', '/bn_beta')
        self.bn_gamma.name = 'bn_gamma'
        self.bn_gamma.path = self.bn_gamma.path.replace('/gamma', '/bn_gamma')

    def _scaler_and_offset(self):
        mean = ops.cast(self.moving_mean, self.dtype)
        variance = ops.cast(self.moving_variance, self.dtype)

        shape = self._shape

        if self.scale:
            bn_gamma = ops.cast(self.bn_gamma, self.dtype)
        else:
            bn_gamma = 1

        if self.center:
            bn_beta = ops.cast(self.bn_beta, self.dtype)
        else:
            bn_beta = 1

        scale = bn_gamma / ops.sqrt(variance + self.epsilon)  # type: ignore
        offset = bn_beta - mean * scale

        return scale, offset

    @property
    def qscaler_and_qoffset(self):
        scale, offset = self._scaler_and_offset()
        return self.kq(scale, training=False), self.bq(offset, training=False)

    def _scaler_and_offset_train(self, qinputs, mask):

        moving_mean = ops.cast(self.moving_mean, ops.dtype(qinputs))
        moving_variance = ops.cast(self.moving_variance, ops.dtype(qinputs))

        mean, variance = self._moments(qinputs, mask)  # type: ignore
        self.moving_mean.assign(  # type: ignore
            moving_mean * self.momentum + mean * (1.0 - self.momentum)  # type: ignore
        )
        self.moving_variance.assign(  # type: ignore
            moving_variance * self.momentum + variance * (1.0 - self.momentum)  # type: ignore
        )

        if self.scale:
            bn_gamma = ops.cast(self.bn_gamma, ops.dtype(qinputs))
        else:
            bn_gamma = 1

        if self.center:
            bn_beta = ops.cast(self.bn_beta, ops.dtype(qinputs))
        else:
            bn_beta = 1

        scale = bn_gamma / ops.sqrt(variance + self.epsilon)  # type: ignore
        offset = bn_beta - mean * scale

        return scale, offset

    def call(self, inputs, training=None, mask=None):
        # Check if the mask has one less dimension than the inputs.
        if mask is not None:
            if len(mask.shape) != len(inputs.shape) - 1:
                # Raise a value error
                raise ValueError(
                    "The mask provided should be one dimension less "
                    "than the inputs. Received: "
                    f"mask.shape={mask.shape}, inputs.shape={inputs.shape}"
                )

        input_dtype = standardize_dtype(inputs.dtype)
        if input_dtype in ("float16", "bfloat16") and training:
            # BN is prone to overflowing for float16/bfloat16 inputs, so we opt
            # out BN for mixed precision.
            inputs = ops.cast(inputs, "float32")

        qinputs = self.iq(inputs, training=training)

        if training:
            scale, offset = self._scaler_and_offset_train(qinputs, mask)
        else:
            scale, offset = self._scaler_and_offset()

        shape = self._shape

        qscale = self.kq(scale, training=True)
        qoffset = self.bq(offset, training=True)

        qscale = ops.reshape(qscale, shape)
        qoffset = ops.reshape(qoffset, shape)

        outputs = qinputs * qscale + qoffset  # type: ignore

        if input_dtype in ("float16", "bfloat16"):
            outputs = ops.cast(outputs, input_dtype)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'kq_conf': self.kq.config,
            'bq_conf': self.bq.config,
        })
        return config

    def _compute_ebops(self, shape):
        bw_inp = self.iq.bits_(shape)
        bw_ker = self.kq.bits_(self._shape)
        bw_bias = self.bq.bits_(self._shape)
        size = ops.cast(ops.prod(shape), self.dtype)
        ebops = ops.sum(bw_inp * bw_ker) + ops.mean(bw_bias) * size  # type: ignore
        return ebops
