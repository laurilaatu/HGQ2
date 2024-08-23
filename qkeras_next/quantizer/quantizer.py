from typing import overload

from keras import ops
from keras.api.layers import Layer
from keras.api.saving import register_keras_serializable

from ..utils.config.quantizer import QuantizerConfig, all_quantizer_keys
from .fixed_point_quantizer import FixedPointQuantizerKBI, FixedPointQuantizerKIF
from .float_point_quantizer import FloatPointQuantizer


@register_keras_serializable(package='qkeras_next')
class Quantizer(Layer):
    """The generic quantizer layer. Supports float, fixed-point (KBI, KIF) quantization. Can be initialized with a QuantizerConfig object or with the quantizer type and its parameters.
    """

    @overload
    def __init__(self, config: QuantizerConfig, **kwargs):
        ...

    @overload
    def __init__(self, q_type='default', place='input', **kwargs):
        ...

    def __init__(self, *args, **kwargs):
        self.supports_masking = True
        self.config, kwargs = self.get_quantizer_config_kwargs(*args, **kwargs)
        super().__init__(**kwargs)
        if self.config.q_type == 'float':
            self.quantizer = FloatPointQuantizer(**self.config)
        elif self.config.q_type == 'kif':
            self.quantizer = FixedPointQuantizerKIF(**self.config)
        elif self.config.q_type == 'kbi':
            self.quantizer = FixedPointQuantizerKBI(**self.config)
        else:
            raise ValueError(f"Unknown quantizer type: {self.config.q_type}")

    def build(self, input_shape):
        self.quantizer.build(input_shape)
        super().build(input_shape)

    def get_quantizer_config_kwargs(self, *args, **kwargs):

        if len(args) == 1 and isinstance(args[0], QuantizerConfig):
            return args[0], kwargs
        config = kwargs.pop('config', None)
        if isinstance(config, QuantizerConfig):
            return config, kwargs

        _kwargs = {}
        for k in list(kwargs.keys()):
            if k in all_quantizer_keys:
                _kwargs[k] = kwargs.pop(k)
        config = QuantizerConfig(*args, **_kwargs)
        return config, kwargs

    def call(self, inputs, training=None):
        return self.quantizer.call(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config['config'] = self.config
        return config

    @property
    def bits(self):
        return self.quantizer.bits

    def __repr__(self):
        return f"{self.__class__.__name__}(q_type={self.config.q_type}, name={self.name}, built={self.built})"

    def bits_(self, shape):
        bits = self.bits
        return self.quantizer.bw_mapper.bw_to_x(bits, shape)
