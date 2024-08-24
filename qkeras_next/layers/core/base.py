from keras import ops
from keras.api.initializers import Constant, Initializer
from keras.api.layers import Concatenate, Layer
from keras.api.saving import deserialize_keras_object, serialize_keras_object
from keras.src import backend

from ...quantizer import Quantizer, QuantizerConfig, numbers
from ...utils.config.layer import global_config


class QLayerAbsBase(Layer):
    pass


class QLayerBase(QLayerAbsBase):
    def __init__(
            self,
            iq_conf: QuantizerConfig | None,
            enable_ebops: bool | None = None,
            beta0: numbers | None = None,
            **kwargs
    ):
        iq_conf = iq_conf or QuantizerConfig('default', 'input')
        super().__init__(**kwargs)
        if enable_ebops is None:
            enable_ebops = global_config['enable_ebops']
        self._enable_ebops = enable_ebops
        beta0 = beta0 or global_config['beta0']
        self._beta0 = Constant(float(beta0)) if not isinstance(beta0, Initializer) else beta0
        self._iq = Quantizer(iq_conf, name=f"{self.name}_iq")

    @property
    def iq(self):
        return self._iq

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

    def build(self, input_shape):
        super().build(input_shape)
        self.iq.build(input_shape)
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
        config['iq_conf'] = serialize_keras_object(self.iq.config)
        return config

    @classmethod
    def from_config(cls, config):
        config['iq_conf'] = deserialize_keras_object(config['iq_conf'])
        return super().from_config(config)

    def enable_lora(self, *args, **kwargs):
        raise NotImplementedError("LoRA is not supported in qkeras_next.")


class QLayerBaseMultiInputs(QLayerAbsBase):

    def __init__(
            self,
            iq_confs: list[QuantizerConfig] | QuantizerConfig | None = None,
            enable_ebops: bool | None = None,
            beta0: numbers | None = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.iq_confs = iq_confs
        self._enable_ebops = enable_ebops or global_config['enable_ebops']

        beta0 = beta0 or global_config['beta0']
        self._beta0 = Constant(float(beta0)) if not isinstance(beta0, Initializer) else beta0

    def build(self, input_shape):
        self.beta = self.add_weight(
            name="beta",
            shape=(1,),
            initializer=self._beta0,
            trainable=False
        )

        iq_confs = self.iq_confs or Quantizer(place='input')
        iq_confs = iq_confs if isinstance(self.iq_confs, list) else [iq_confs] * len(input_shape)

        assert len(iq_confs) == len(input_shape), f"If you specify a list of QuantizerConfig, it must have the same length as the number of inputs. Got {len(iq_confs)} QuantizerConfigs for {len(input_shape)} inputs."
        self.inpqs = [Quantizer(iq_conf) for iq_conf in iq_confs]
        for inpq, shape in zip(self.inpqs, input_shape):
            inpq.build(shape)

    @property
    def enable_ebops(self):
        return self._enable_ebops
