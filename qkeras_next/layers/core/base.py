import inspect
from abc import ABCMeta
from collections.abc import Callable, Iterable, Sequence
from functools import wraps
from typing import Any, overload

from keras import ops
from keras.api.initializers import Constant, Initializer
from keras.api.layers import Concatenate, Layer
from keras.api.saving import deserialize_keras_object, register_keras_serializable, serialize_keras_object
from keras.src import backend

from ...quantizer import Quantizer, QuantizerConfig, numbers
from ...utils.config.layer import global_config


class QLayerMeta(ABCMeta):
    def __new__(mcs: type, name: str, bases: tuple[type], attrs: dict[str, object], **kwargs):

        # ============ Compute ebops if _compute_ebops presents ==============
        # ====================================================================
        original_call: Callable = attrs.get('call', bases[0].call)  # type: ignore
        _compute_ebops = attrs.get('_compute_ebops', None)
        if _compute_ebops is None:
            _compute_ebops = bases[0]._compute_ebops  # type: ignore
        if original_call is not Layer.call and _compute_ebops is not QLayerBase._compute_ebops:

            signature = inspect.signature(original_call)

            VAR_KEYWORD = inspect.Parameter.VAR_KEYWORD
            KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY
            has_training = 'training' in signature.parameters
            has_var_keyword = any(v.kind == VAR_KEYWORD for v in signature.parameters.values())
            new_signature = signature
            if not has_training and not has_var_keyword:
                training_param = inspect.Parameter('training', KEYWORD_ONLY, default=None)
                new_params = signature.parameters.copy()
                new_params['training'] = training_param
                new_signature = signature.replace(parameters=new_params.values())  # type: ignore

            @wraps(original_call)
            def call(self, *args, **kwargs):
                if kwargs.get('training', None) and self.enable_ebops:
                    ebops = self._compute_ebops(*map(ops.shape, args))
                    self._ebops.assign(ops.cast(ebops, self._ebops.dtype))
                    self.add_loss(ebops * self.beta)
                return original_call(self, *args, **kwargs)
            call.__signature__ = new_signature  # type: ignore

            if _compute_ebops is not QLayerBase._compute_ebops:
                attrs['call'] = call
        # ====================================================================

        cls = super().__new__(mcs, name, bases, attrs, **kwargs)  # type: ignore

        # =========== Register as Keras serializable if possible =============
        # ====================================================================
        if cls.get_config is not Layer.get_config:
            original_get_config = cls.get_config

            def get_config(self):
                config = original_get_config(self)
                config = serialize_keras_object(config)
                return config
            cls.get_config = get_config
            cls = register_keras_serializable(package='qkeras_next')(cls)
        # ====================================================================

        return cls


class QLayerBase(Layer, metaclass=QLayerMeta):
    def __init__(
            self,
            enable_ebops: bool | None = None,
            beta0: numbers | None | Initializer = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        if enable_ebops is None:
            enable_ebops = global_config['enable_ebops']
        beta0 = beta0 if beta0 is not None else global_config['beta0']
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

    # @property
    # def ebops(self):
    #     if self._ebops is None:
    #         ebops = backend.convert_to_tensor(0)
    #     else:
    #         ebops = self._ebops
    #     for layer in self._flatten_layers():
    #         if isinstance(layer, QLayerBase):
    #             if layer._ebops is not None:
    #                 ebops = ebops + layer._ebops
    #     return ebops
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
        config['beta0'] = self._beta0
        return config

    def enable_lora(self, *args, **kwargs):
        raise NotImplementedError("LoRA is not supported in qkeras_next.")

    @classmethod
    def from_config(cls, config):
        config = deserialize_keras_object(config)
        return super().from_config(config)

    @property
    def _quantizers(self):
        quantizers = []
        for layer in self._flatten_layers():
            if isinstance(layer, Quantizer):
                quantizers.append(layer)
        return quantizers

    def _compute_ebops(self, *args, **kwargs):
        raise NotImplementedError("This method is abstract and should be implemented in subclasses.")


class QLayerBaseSingleInput(QLayerBase):
    def __init__(
            self,
            iq_conf: QuantizerConfig | None,
            disable_iq=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        if not disable_iq:
            iq_conf = iq_conf or QuantizerConfig('default', 'input')
            self._iq = Quantizer(iq_conf, name=f"{self.name}_iq")
        else:
            self._iq = None

    @property
    def iq(self):
        if self._iq is None:
            raise AttributeError("iq has been disabled.")
        return self._iq

    def build(self, input_shape):
        super().build(input_shape)
        if self._iq is not None:
            self.iq.build(input_shape)

    def get_config(self):
        config = super().get_config()
        config['iq_conf'] = self.iq.config
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
