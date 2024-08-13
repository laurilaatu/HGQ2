from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import overload

from keras.api.constraints import Constraint
from keras.api.initializers import Initializer
from keras.api.regularizers import Regularizer

from qkeras_next.quantizer.base import BitwidthMapperBase
from qkeras_next.utils.constraints import Min, MinMax

from .base import BitwidthMapperBase, numbers


class QuantizerConfigBase(Mapping):
    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)


@dataclass
class KBIDefaultWeight(QuantizerConfigBase):
    k0: numbers | bool | Initializer = True
    b0: numbers | Initializer = 4
    i0: numbers | Initializer = 2
    round_mode: str = 'RND'
    overflow_mode: str = 'WRAP'
    bc: Constraint | None = MinMax(0, 12)
    ic: Constraint | None = None
    br: Regularizer | None = None
    ir: Regularizer | None = None
    i_decay_speed: numbers = float('inf')
    homogeneous_axis = ()
    heterogeneous_axis = None
    bw_mapper: BitwidthMapperBase | None = None


@dataclass(frozen=True)
class KBIDefaultInput(QuantizerConfigBase):
    keep_negative = True
    k0: numbers | bool | Initializer = True
    b0: numbers | Initializer = 4
    i0: numbers | Initializer = 2
    round_mode: str = 'RND'
    overflow_mode: str = 'SAT'
    bc: Constraint | None = MinMax(0, 12)
    ic: Constraint | None = None
    br: Regularizer | None = None
    ir: Regularizer | None = None
    i_decay_speed: numbers = 0.01
    homogeneous_axis: Sequence[int] | None = (0,)
    heterogeneous_axis: Sequence[int] | None = None
    bw_mapper: BitwidthMapperBase | None = None


@dataclass(frozen=True)
class KIFDefaultWeight(QuantizerConfigBase):
    k0: numbers | bool | Initializer = True
    i0: numbers | Initializer = 4
    f0: numbers | Initializer = 2
    round_mode: str = 'RND'
    overflow_mode: str = 'SAT'
    ic: Constraint | None = MinMax(-12, 12)
    ir: Regularizer | None = None
    fc: Constraint | None = MinMax(-12, 12)
    fr: Regularizer | None = None
    i_decay_speed: numbers = float('inf')
    homogeneous_axis: Sequence[int] | None = ()
    heterogeneous_axis: Sequence[int] | None = None
    bw_mapper: BitwidthMapperBase | None = None


@dataclass(frozen=True)
class KIFDefaultInput(QuantizerConfigBase):
    k0: numbers | bool | Initializer = True
    i0: numbers | Initializer = 4
    f0: numbers | Initializer = 2
    round_mode: str = 'RND'
    overflow_mode: str = 'SAT'
    ic: Constraint | None = MinMax(-12, 12)
    ir: Regularizer | None = None
    fc: Constraint | None = MinMax(-12, 12)
    fr: Regularizer | None = None
    i_decay_speed: numbers = 0.01
    homogeneous_axis: Sequence[int] | None = (0,)
    heterogeneous_axis: Sequence[int] | None = None
    bw_mapper: BitwidthMapperBase | None = None


@dataclass(frozen=True)
class FloatDefaultWeight(QuantizerConfigBase):
    m0: numbers | Initializer = 2
    e0: numbers | Initializer = 1
    e00: numbers | Initializer = 0
    mc: Constraint | None = Min(-1)
    ec: Constraint | None = MinMax(0, 4)
    e0c: Constraint | None = MinMax(-8, 8)
    mr: Regularizer | None = None
    er: Regularizer | None = None
    e0r: Regularizer | None = None
    homogeneous_axis: Sequence[int] | None = ()
    heterogeneous_axis: Sequence[int] | None = None
    bw_mapper: BitwidthMapperBase | None = None


@dataclass(frozen=True)
class FloatDefaultInput(QuantizerConfigBase):
    m0: numbers | Initializer = 2
    e0: numbers | Initializer = 1
    e00: numbers | Initializer = 0
    mc: Constraint | None = Min(-1)
    ec: Constraint | None = MinMax(0, 4)
    e0c: Constraint | None = MinMax(-8, 8)
    mr: Regularizer | None = None
    er: Regularizer | None = None
    e0r: Regularizer | None = None
    homogeneous_axis: Sequence[int] | None = (0,)
    heterogeneous_axis: Sequence[int] | None = None
    bw_mapper: BitwidthMapperBase | None = None


class QuantizerConfig(Mapping):

    @overload
    def __init__(
        self,
        type: str,
        default: str = 'weight',
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

    def __init__(self, type: str, default: str = 'weight', **kwargs) -> None:  # type: ignore
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
        assert type in ['kbi', 'kif', 'float']
        match (type, default):
            case ('kbi', 'weight'):
                self.config = KBIDefaultWeight(**kwargs)
            case ('kbi', 'input'):
                self.config = KBIDefaultInput(**kwargs)
            case ('kif', 'weight'):
                self.config = KIFDefaultWeight(**kwargs)
            case ('kif', 'input'):
                self.config = KIFDefaultInput(**kwargs)
            case ('float', 'weight'):
                self.config = FloatDefaultWeight(**kwargs)
            case ('float', 'input'):
                self.config = FloatDefaultInput(**kwargs)
            case _:
                raise ValueError(f"Invalid combination of type and default: {type}, {default}")

    def __getitem__(self, key):
        return self.config[key]

    def __iter__(self):
        return iter(self.config)

    def __len__(self) -> int:
        return len(self.config)
