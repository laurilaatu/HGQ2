from collections.abc import Sequence
from math import prod
from warnings import warn

from keras import ops
from keras.src import backend

from ..quantizer import QuantizerConfig
from .activation import QUnaryFunctionLUT
from .core import QLayerBaseSingleInput


class QSoftmax(QLayerBaseSingleInput):
    def __init__(
        self,
        axis: int | Sequence[int] = -1,
        iq_conf: None | QuantizerConfig = None,
        stable=True,
        exp_iq_conf: None | QuantizerConfig = None,
        exp_oq_conf: None | QuantizerConfig = None,
        inv_iq_conf: None | QuantizerConfig = None,
        inv_oq_conf: None | QuantizerConfig = None,
        allow_heterogeneous_table: bool = False,
        input_scaler: float = 1.0,
        parallelization_factor: int = -1,
        **kwargs,
    ):
        self.supports_masking = True
        super().__init__(iq_conf=iq_conf, **kwargs)  # type: ignore
        self.stable = stable
        self.axis = tuple(axis) if isinstance(axis, Sequence) else (axis,)
        self.parallelization_factor = parallelization_factor

        assert not allow_heterogeneous_table, 'No hls4ml support; remove this check if you know what you are doing.'
        self._allow_heterogeneous_table = allow_heterogeneous_table

        self.input_scaler = input_scaler

        def _inv(x):
            return 1.0 / (x + backend.epsilon())

        def _exp(x):
            if self.stable:
                return ops.exp(-x * self.input_scaler)
            else:
                return ops.exp(x * self.input_scaler)

        inv_iq_conf = inv_iq_conf or QuantizerConfig('default', 'datalane')
        exp_iq_conf = exp_iq_conf or QuantizerConfig('default', 'datalane')
        exp_oq_conf = exp_oq_conf or QuantizerConfig('default', 'table')
        inv_oq_conf = inv_oq_conf or QuantizerConfig('default', 'table')
        if not self._allow_heterogeneous_table:
            inv_iq_conf.config['heterogeneous_axis'] = ()
            inv_iq_conf.config['homogeneous_axis'] = None
            exp_iq_conf.config['heterogeneous_axis'] = ()
            exp_iq_conf.config['homogeneous_axis'] = None

        if 'k0' in inv_oq_conf.config:
            inv_oq_conf.config['k0'] = 0
        if 'k0' in exp_oq_conf.config:
            exp_oq_conf.config['k0'] = 0

        if inv_oq_conf.config.get('overflow_mode', None) == 'WRAP':
            warn('WRAP overflow mode on inverse table will likely cause hugh table size. Set to SAT instead.')
            inv_oq_conf.config['overflow_mode'] = 'SAT'  # type: ignore

        self.inv_table = QUnaryFunctionLUT(
            _inv,
            inv_iq_conf,
            inv_oq_conf,
            enable_iq=True,
            enable_oq=True,
            allow_heterogeneous_table=allow_heterogeneous_table,
            name=f'{self.name}_inv_table',
            enable_ebops=self.enable_ebops,
            beta0=self._beta0.clone(),
        )
        self.exp_table = QUnaryFunctionLUT(
            _exp,
            exp_iq_conf,
            exp_oq_conf,
            enable_iq=self.stable,
            enable_oq=True,
            allow_heterogeneous_table=allow_heterogeneous_table,
            name=f'{self.name}_exp_table',
            enable_ebops=self.enable_ebops,
            beta0=self._beta0.clone(),
        )

    def build(self, input_shape):
        self.exp_table.build(input_shape)

        inv_shape = list(input_shape)
        for i in self.axis:
            inv_shape[i] = 1
        self.inv_table.build(tuple(inv_shape))

        super().build(input_shape)

    def call(self, inputs, training=None, mask=None):  # type: ignore
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)

        if self.stable:
            inputs = ops.max(inputs, axis=self.axis, keepdims=True) - inputs

        exp_inp = self.exp_table(inputs, training=training)

        if mask is not None:
            exp_inp = backend.cast(mask, ops.dtype(inputs)) * exp_inp

        sums = ops.sum(exp_inp, axis=self.axis, keepdims=True)  # type: ignore
        divisor = self.inv_table(sums, training=training)

        return exp_inp * divisor

    def _compute_ebops(self, shape):
        accum_shape = tuple(1 if i in self.axis else s for i, s in enumerate(shape))
        max_instance = prod(accum_shape)
        n_instance = self.parallelization_factor if self.parallelization_factor > 0 else max_instance
        factor = n_instance / max_instance

        if self.stable:
            inp_bits = self.iq.bits_(shape)
            substract_ebops = 1.55 * ops.sum(inp_bits)  # type: ignore # TODO: better ebops cost model for add and max
        else:
            substract_ebops = 0

        exp_bits = self.exp_table.oq.bits_(shape)
        inv_bits = self.inv_table.oq.bits_(accum_shape)

        accum_ebops = ops.sum(exp_bits) - ops.sum(ops.min(exp_bits, axis=self.axis))  # type: ignore
        mult_ebops = ops.sum(exp_bits * inv_bits)  # type: ignore

        ebops = substract_ebops + accum_ebops + mult_ebops
        return ebops * factor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'axis': self.axis,
                'stable': self.stable,
                'exp_oq_conf': self.exp_table.oq.config,
                'exp_iq_conf': self.exp_table.iq.config if self.stable else None,
                'inv_oq_conf': self.inv_table.oq.config,
                'inv_iq_conf': self.inv_table.iq.config,
                'allow_heterogeneous_table': self._allow_heterogeneous_table,
                'input_scaler': self.input_scaler,
                'parallelization_factor': self.parallelization_factor,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    @property
    def ebops(self):
        ebops = sum(
            (  # type: ignore
                ops.convert_to_tensor(self._ebops),
                self.exp_table.ebops,
                self.inv_table.ebops,
            )
        )
        return round(ops.convert_to_numpy(ebops).item())  # type: ignore
