from abc import ABCMeta
from collections.abc import Iterable, Sequence

from keras import ops
from keras.api.initializers import Constant, Initializer
from keras.api.layers import Concatenate, Layer
from keras.api.saving import deserialize_keras_object, serialize_keras_object
from keras.src import backend

from ...quantizer import Quantizer, QuantizerConfig, numbers
from ...utils.config.layer import global_config


class QLayerBase(Layer, metaclass=ABCMeta):
    def __init__(
            self,
            enable_ebops: bool | None = None,
            beta0: numbers | None | Initializer = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        if enable_ebops is None:
            enable_ebops = global_config['enable_ebops']
        beta0 = beta0 or global_config['beta0']
        beta0 = Constant(float(beta0)) if not isinstance(beta0, Initializer) else beta0
        self._enable_ebops = enable_ebops
        self._beta0 = beta0

    @property
    def beta(self):
        if self._beta is None:
            return backend.convert_to_tensor(0)
        return backend.convert_to_tensor(self._beta)

    @property
    def ebops(self):
        if self._ebops is None:
            return backend.convert_to_tensor(0)
        return backend.convert_to_tensor(self._ebops)

    @property
    def enable_ebops(self):
        return self._enable_ebops

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        if self.enable_ebops:
            self._beta = self.add_weight(
                name="beta",
                shape=(),
                initializer=self._beta0,
                trainable=False
            )
            self._ebops = self.add_weight(
                name="ebops",
                shape=(),
                initializer=Constant(0.),
                trainable=False,
                dtype='uint32'
            )
        else:
            self._beta = None
            self._ebops = None

    def get_config(self):
        config = super().get_config()
        config['enable_ebops'] = self.enable_ebops
        config['beta0'] = serialize_keras_object(self._beta0)
        return config

    def enable_lora(self, *args, **kwargs):
        raise NotImplementedError("LoRA is not supported in qkeras_next.")

    @classmethod
    def from_config(cls, config):
        config = deserialize_keras_object(config)
        return super().from_config(config)


class QLayerBaseSingleInput(QLayerBase):
    def __init__(
            self,
            iq_conf: QuantizerConfig | None,
            **kwargs
    ):
        super().__init__(**kwargs)
        iq_conf = iq_conf or QuantizerConfig('default', 'input')
        self._iq = Quantizer(iq_conf, name=f"{self.name}_iq")

    @property
    def iq(self):
        return self._iq

    def build(self, input_shape):
        super().build(input_shape)
        self.iq.build(input_shape)

    def get_config(self):
        config = super().get_config()
        config['iq_conf'] = serialize_keras_object(self.iq.config)
        return config


class _InvocableTuple(tuple):
    def __call__(self, x, **kwargs):
        assert len(self) == len(x), f"number of elements in InvocableList must match number of inputs, got {len(self)} != {len(x)}"
        return tuple(f(x_, **kwargs) for f, x_ in zip(self, x))


class QLayerBaseMultiInputs(QLayerBase):
    def __init__(
            self,
            iq_confs: Sequence[QuantizerConfig] | QuantizerConfig | None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self._iq_confs = iq_confs if iq_confs is not None else QuantizerConfig('default', 'input')

    @property
    def iqs(self):
        if not self.built:
            raise AttributeError("iqs is not available before build.")
        return self._iqs

    @property
    def iq_confs(self):
        return self._iq_confs

    def build(self, input_shape):
        super().build(input_shape)
        n_input = len(input_shape)
        for _input_shape in input_shape:
            assert isinstance(_input_shape, Iterable), f"each element of input_shape must be iterable, got {_input_shape}"

        if isinstance(self.iq_confs, QuantizerConfig):
            self._iq_confs = [self.iq_confs] * n_input
        assert len(self.iq_confs) == n_input, f"number of iq_confs must match number of inputs, got {len(self._iq_confs)} != {n_input}"

        _iqs = []
        for i, iq_conf in enumerate(self.iq_confs):
            iq = Quantizer(iq_conf, name=f"{self.name}_iq_{i}")
            iq.build(input_shape[i])
            _iqs.append(iq)
        self._iqs = _InvocableTuple(_iqs)

    def get_config(self):
        config = super().get_config()
        config['iq_confs'] = serialize_keras_object(self.iq_confs)
        return config
