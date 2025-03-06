from collections.abc import Sequence
from math import log2, prod

from keras import ops

from ...quantizer.config import QuantizerConfig
from ...utils.misc import warn_no_synth
from ..core.base import QLayerBaseSingleInput


class QSum(QLayerBaseSingleInput):
    def __init__(
        self,
        iq_conf: QuantizerConfig | None = None,
        axis: int | Sequence[int] = -1,
        scale: float = 1.0,
        keepdims: bool = False,
        **kwargs,
    ):
        super().__init__(iq_conf=iq_conf, **kwargs)
        self.axis = tuple(axis) if isinstance(axis, Sequence) else (axis,)
        assert log2(scale).is_integer(), 'Scale must be a power of 2.'
        self._scale = scale
        self._keepdims = keepdims

    def build(self, input_shape):
        super().build(input_shape)
        axis = sorted(i if i >= 0 else i + len(input_shape) for i in self.axis)
        self.axis = tuple(axis)
        cond = all(i1 - i0 > 1 for i0, i1 in zip(axis[:-1], axis[1:]))
        warn_no_synth(cond, 'Softmax axis is not contiguous, hls4ml will not be able to synthesize this layer.')

    @property
    def scale(self):
        return self._scale

    @property
    def keepdims(self):
        return self._keepdims

    def _compute_ebops(self, shape):
        bits = self.iq.bits_(shape)
        ebops = ops.sum(bits) - ops.sum(ops.min(bits, axis=self.axis))  # type: ignore
        ebops = ebops * 0.65  # TODO: better ebops cost model for accumulators
        return ebops

    def call(self, inputs, training=None):
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)
        r = ops.sum(inputs, axis=self.axis, keepdims=self.keepdims) * self.scale  # type: ignore
        return r


class QMeanPow2(QSum):
    def __init__(
        self,
        iq_conf: QuantizerConfig | None = None,
        axis: int | Sequence[int] = -1,
        keepdims: bool = False,
        **kwargs,
    ):
        super().__init__(iq_conf=iq_conf, axis=axis, keepdims=keepdims, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        scale = 1.0 / prod([input_shape[i] for i in self.axis])
        self._scale = round(2.0 ** log2(scale))
