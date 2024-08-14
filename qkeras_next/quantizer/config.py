from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict, overload

from keras.api.constraints import Constraint
from keras.api.initializers import Initializer
from keras.api.regularizers import Regularizer
from keras.api.saving import deserialize_keras_object, register_keras_serializable, serialize_keras_object

from qkeras_next.quantizer.base import BitwidthMapperBase
from qkeras_next.utils.constraints import Min, MinMax

from .base import BitwidthMapperBase, numbers


class QuantizerConfigBase(TypedDict):
    homogeneous_axis: Sequence[int]
    heterogeneous_axis: Sequence[int] | None
    bw_mapper: BitwidthMapperBase | None


def _serialize_config(config: QuantizerConfigBase):
    return {k: serialize_keras_object(v) for k, v in config.items()}


class KBIConfig(QuantizerConfigBase):
    k0: numbers | bool | Initializer
    b0: numbers | Initializer
    i0: numbers | Initializer
    round_mode: str
    overflow_mode: str
    bc: Constraint | None
    ic: Constraint | None
    br: Regularizer | None
    ir: Regularizer | None
    i_decay_speed: numbers


class KIFConfig(QuantizerConfigBase):
    k0: numbers | bool | Initializer
    i0: numbers | Initializer
    f0: numbers | Initializer
    round_mode: str
    overflow_mode: str
    ic: Constraint | None
    ir: Regularizer | None
    fc: Constraint | None
    fr: Regularizer | None
    i_decay_speed: numbers


class FloatConfig(QuantizerConfigBase):
    m0: numbers | Initializer
    e0: numbers | Initializer
    e00: numbers | Initializer
    mc: Constraint | None
    ec: Constraint | None
    e0c: Constraint | None
    mr: Regularizer | None
    er: Regularizer | None
    e0r: Regularizer | None


kbi_weight_default = KBIConfig(
    k0=True,
    b0=4,
    i0=2,
    round_mode='RND',
    overflow_mode='WRAP',
    bc=MinMax(0, 12),
    ic=None,
    br=None,
    ir=None,
    i_decay_speed=float('inf'),
    homogeneous_axis=(),
    heterogeneous_axis=None,
    bw_mapper=None
)

kbi_input_default = KBIConfig(
    k0=True,
    b0=4,
    i0=2,
    round_mode='RND',
    overflow_mode='SAT',
    bc=MinMax(0, 12),
    ic=None,
    br=None,
    ir=None,
    i_decay_speed=0.01,
    homogeneous_axis=(0,),
    heterogeneous_axis=None,
    bw_mapper=None
)

kif_weight_default = KIFConfig(
    k0=True,
    i0=4,
    f0=2,
    round_mode='RND',
    overflow_mode='SAT',
    ic=MinMax(-12, 12),
    ir=None,
    fc=MinMax(-12, 12),
    fr=None,
    i_decay_speed=float('inf'),
    homogeneous_axis=(),
    heterogeneous_axis=None,
    bw_mapper=None,
)

kif_input_default = KIFConfig(
    k0=True,
    i0=4,
    f0=2,
    round_mode='RND',
    overflow_mode='SAT',
    ic=MinMax(-12, 12),
    ir=None,
    fc=MinMax(-12, 12),
    fr=None,
    i_decay_speed=0.01,
    homogeneous_axis=(0,),
    heterogeneous_axis=None,
    bw_mapper=None,
)

float_weight_default = FloatConfig(
    m0=2,
    e0=2,
    e00=0,
    mc=Min(-1),
    ec=MinMax(0, 4),
    e0c=MinMax(-16, 16),
    mr=None,
    er=None,
    e0r=None,
    homogeneous_axis=(),
    heterogeneous_axis=None,
    bw_mapper=None,
)

float_weight_default = FloatConfig(
    m0=2,
    e0=2,
    e00=0,
    mc=Min(-1),
    ec=MinMax(0, 4),
    e0c=MinMax(-16, 16),
    mr=None,
    er=None,
    e0r=None,
    homogeneous_axis=(0,),
    heterogeneous_axis=None,
    bw_mapper=None,
)

default_configs: dict[tuple[str, str], QuantizerConfigBase] = {
    ('kbi', 'weight'): kbi_weight_default,
    ('kbi', 'input'): kbi_input_default,
    ('kif', 'weight'): kif_weight_default,
    ('kif', 'input'): kif_input_default,
    ('float', 'weight'): float_weight_default,
    ('float', 'input'): float_weight_default,
}

all_quantizer_keys = {k for v in default_configs.values() for k in v.keys()} | {'type', 'default'}


@register_keras_serializable(package='qkeras_next')
class QuantizerConfig(Mapping):

    @overload
    def __init__(
        self,
        type: str,
        default: str = 'weight',
        *,
        k0: numbers | bool | Initializer = True,
        b0: numbers | Initializer = 4,
        i0: numbers | Initializer = 2,
        round_mode: str = 'RND',
        overflow_mode: str = 'WRAP',
        bc: Constraint | None = MinMax(0, 12),
        ic: Constraint | None = None,
        br: Regularizer | None = None,
        ir: Regularizer | None = None,
        i_decay_speed: numbers = -1,
        homogeneous_axis: Sequence[int] | None = None,
        heterogeneous_axis: Sequence[int] | None = None,
        bw_mapper: BitwidthMapperBase | None = None,
    ) -> None:
        """Fixed point quantizer config with KBI parametrization.

        Parameters
        ----------
        type : str
            The type of the quantizer. 'kbi' for this implementation.
        default : str
            The default config to be loaded of the quantizer. One of 'weight', 'input'.
        k0 : numbers | bool | Initializer, optional
            If the quantizer allows negative values, by default True
        b0 : numbers | Initializer, optional
            The initial value of the number of bits (excl. sign), by default 4
        i0 : numbers | Initializer, optional
            The initial value of the number of integer bits (excl. sign), by default 2
        round_mode : str, optional
            Rounding mode. One of 'RND', 'RND_CONV', 'TRN', 'S_RND', 'S_RND_CONV', by default 'RND'
        overflow_mode : str, optional
            Overflow mode. One of 'WRAP', 'SAT', 'SAT_SYM', by default 'WRAP'
        bc : Constraint | None, optional
            Constraint for the number of bits, by default MinMax(0, 12)
        ic : Constraint | None, optional
            Constraint for the number of integer bits, by default None
        br : Regularizer | None, optional
            Regularizer for the number of bits, by default None
        ir : Regularizer | None, optional
            Regularizer for the number of integer bits, by default None
        i_decay_speed : numbers, optional
            The decay speed of the integer. Only used if `round_mode` is 'WRAP', by default -1
        homogeneous_axis : Sequence[int] | None, optional
            The axes that are quantized homogeneously. Mutually exclusive with `heterogeneous_axis`. Only used if `bw_mapper` is `DefaultBitwidthMapper`, by default None
        heterogeneous_axis : Sequence[int] | None, optional
            The axes that are quantized heterogeneously. Mutually exclusive with `homogeneous_axis`. Only used if `bw_mapper` is `DefaultBitwidthMapper`, by default None
        bw_mapper : BitwidthMapperBase | None, optional
            The bitwidth mapper to be used. Must be a subclass of `BitwidthMapperBase`. If None, the default bitwidth mapper is used with `homogeneous_axis` and `heterogeneous_axis` as arguments, by default None
        """
        ...

    @overload
    def __init__(
        self,
        type: str,
        default: str = 'weight',
        *,
        k0: numbers | bool | Initializer = True,
        i0: numbers | Initializer = 4,
        f0: numbers | Initializer = 2,
        round_mode: str = 'RND',
        overflow_mode: str = 'SAT',
        ic: Constraint | None = MinMax(-12, 12),
        ir: Regularizer | None = None,
        fc: Constraint | None = MinMax(-12, 12),
        fr: Regularizer | None = None,
        i_decay_speed: numbers = 0.01,
        homogeneous_axis: Sequence[int] | None = (0,),
        heterogeneous_axis: Sequence[int] | None = None,
    ) -> None:
        """Fixed point quantizer config with KIF parametrization.

        Parameters
        ----------
        type : str
            The type of the quantizer. 'kif' for this implementation.
        default : str
            The default config to be loaded of the quantizer. One of 'weight', 'input'.
        k0 : numbers | bool | Initializer, optional
            If the quantizer allows negative values, by default True
        i0 : numbers | Initializer, optional
            The initial value of the number of integer bits (excl. sign), by default 4
        f0 : numbers | Initializer, optional
            The initial value of the number of fraction bits, by default 2
        round_mode : str, optional
            Rounding mode. One of 'RND', 'RND_CONV', 'TRN', 'S_RND', 'S_RND_CONV', by default 'RND'
        overflow_mode : str, optional
            Overflow mode. One of 'WRAP', 'SAT', 'SAT_SYM', by default 'SAT'
        ic : Constraint | None, optional
            Constraint for the number of integer bits, by default MinMax(-12, 12)
        ir : Regularizer | None, optional
            Regularizer for the number of integer bits, by default None
        fc : Constraint | None, optional
            Constraint for the number of fraction bits, by default MinMax(-12, 12)
        fr : Regularizer | None, optional
            Regularizer for the number of fraction bits, by default None
        i_decay_speed : numbers, optional
            The decay speed of the integer. Only used if `round_mode` is 'WRAP', by default 0.01
        homogeneous_axis : Sequence[int] | None, optional
            The axes that are quantized homogeneously. Mutually exclusive with `heterogeneous_axis`. Only used if `bw_mapper` is `DefaultBitwidthMapper`, by default (0,)
        heterogeneous_axis : Sequence[int] | None, optional
            The axes that are quantized heterogeneously. Mutually exclusive with `homogeneous_axis`. Only used if `bw_mapper` is `DefaultBitwidthMapper`, by default None
        """
        ...

    @overload
    def __init__(
        self,
        type: str,
        default: str = 'weight',
        *,
        m0: numbers | Initializer = 2,
        e0: numbers | Initializer = 1,
        e00: numbers | Initializer = 0,
        mc: Constraint | None = Min(-1),
        ec: Constraint | None = MinMax(0, 4),
        e0c: Constraint | None = MinMax(-8, 8),
        mr: Regularizer | None = None,
        er: Regularizer | None = None,
        e0r: Regularizer | None = None,
        homogeneous_axis: Sequence[int] | None = (),
        heterogeneous_axis: Sequence[int] | None = None,
    ) -> None:
        """Floating point quantizer config.

        Parameters
        ----------
        type : str
            The type of the quantizer. 'float' for this implementation.
        default : str
            The default config to be loaded of the quantizer. One of 'weight', 'input'.
        m0 : numbers | Initializer, optional
            The initial value of the number of mantissa bits, by default 2
        e0 : numbers | Initializer, optional
            The initial value of the number of exponent bits, by default 1
        e00 : numbers | Initializer, optional
            The initial value of the number of exponent bits for the first axis, by default 0
        mc : Constraint | None, optional
            Constraint for the number of mantissa bits, by default Min(-1)
        ec : Constraint | None, optional
            Constraint for the number of exponent bits, by default MinMax(0, 4)
        e0c : Constraint | None, optional
            Constraint for the number of exponent bits for the first axis, by default MinMax(-8, 8)
        mr : Regularizer | None, optional
            Regularizer for the number of mantissa bits, by default None
        er : Regularizer | None, optional
            Regularizer for the number of exponent bits, by default None
        e0r : Regularizer | None, optional
            Regularizer for the number of exponent bits for the first axis, by default None
        homogeneous_axis : Sequence[int] | None, optional
            The axes that are quantized homogeneously. Mutually exclusive with `heterogeneous_axis`. Only used if `bw_mapper` is `DefaultBitwidthMapper`, by default ()
        heterogeneous_axis : Sequence[int] | None, optional
            The axes that are quantized heterogeneously. Mutually exclusive with `homogeneous_axis`. Only used if `bw_mapper` is `DefaultBitwidthMapper`, by default None
        """
        ...

    def __init__(self, type: str, default: str = 'weight', **kwargs) -> None:
        """Universal quantizer config. The type of the quantizer is specified by the `type` argument.

        Parameters
        ----------
        type : str
            The type of the quantizer. One of 'kbi', 'kif', 'float'
        default : str, optional
            The default config to be loaded of the quantizer. One of 'weight', 'input', by default 'weight'
        """
        type = type.lower()
        default = default.lower()
        assert type in ('kbi', 'kif', 'float')
        assert default in ('weight', 'input', 'all')

        assert kwargs.get('homogeneous_axis') is None or kwargs.get('heterogeneous_axis') is None, \
            "homogeneous_axis and heterogeneous_axis are mutually exclusive. Set only one of them."

        if kwargs.get('homogeneous_axis') is not None:
            kwargs['heterogeneous_axis'] = None
        if kwargs.get('heterogeneous_axis') is not None:
            kwargs['homogeneous_axis'] = None

        config = default_configs.get((type, default))
        assert config is not None, f"Default config for ({type}, {default}) not found."
        self.config = config.copy()

        if self.config is not None:
            for k, v in kwargs.items():
                if k not in self.config:
                    raise ValueError(f"{k} is not a valid parameter for {type} quantizer config.")
                self.config[k] = v

        self.kwargs = kwargs
        self.type = type
        self.default = default
        self._tmp_storage = {}

    def __getitem__(self, key):
        return self.config[key]

    def __iter__(self):
        return iter(self.config)

    def __len__(self) -> int:
        return len(self.config)

    def get_config(self):
        return {
            'type': self.type,
            'default': self.default,
            **_serialize_config(self.config)
        }

    @classmethod
    def from_config(cls, config):
        for k, v in config.items():
            config[k] = deserialize_keras_object(v)
        return cls(**config)


class QuantizerConfigScope:
    def __init__(self, type: str = 'all', default: str = 'all', **kwargs):
        """Override default quantizer config within a context.

        Parameters
        ----------
        type : str
            The type of the quantizer. One of 'kbi', 'kif', 'float', 'all'
        default : str
            The default config to be loaded of the quantizer. One of 'weight', 'input', 'all'

        """
        type = type.lower()
        default = default.lower()
        assert type in ('kbi', 'kif', 'float', 'all')
        assert default in ('weight', 'input', 'all')

        assert kwargs.get('homogeneous_axis') is None or kwargs.get('heterogeneous_axis') is None, \
            "homogeneous_axis and heterogeneous_axis are mutually exclusive. Set only one of them."

        if kwargs.get('homogeneous_axis') is not None:
            kwargs['heterogeneous_axis'] = None
        if kwargs.get('heterogeneous_axis') is not None:
            kwargs['homogeneous_axis'] = None

        i, f, b = kwargs.get('i0', None), kwargs.get('f0', None), kwargs.get('b0', None)
        if sum((i is not None, f is not None, b is not None)) == 2:
            if i is None:
                kwargs['i0'] = b - f
            if f is None:
                kwargs['f0'] = b - i
            if b is None:
                kwargs['b0'] = i + f

        for k in kwargs:
            if k not in all_quantizer_keys:
                raise ValueError(f"{k} is not a valid parameter for any known quantizer configs.")

        self.type = type
        self.default = default
        self.kwargs = kwargs
        self._tmp_storage = {}

    def __enter__(self):
        for (type, default), default_conf in default_configs.items():
            if (self.type == type or self.type == 'all') and \
               (self.default == default or self.default == 'all'):
                self._tmp_storage[(type, default)] = default_conf.copy()
                for k, v in self.kwargs.items():
                    if k in default_conf:
                        default_conf[k] = v

    def __exit__(self, exc_type, exc_value, traceback):
        for (type, default) in self._tmp_storage:
            default_configs[(type, default)].update(self._tmp_storage[(type, default)])
        self._tmp_storage.clear()

    def override(self):
        """Override the default quantizer config."""
        self.__enter__()
        self._tmp_storage.clear()
