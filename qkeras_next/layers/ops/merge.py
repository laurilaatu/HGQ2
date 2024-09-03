from math import log2

from keras import ops
from keras.api.layers import Add, Average, Dot, Maximum, Minimum, Multiply, Subtract
from keras.src.layers.merging.base_merge import Merge

from ..core.base import QLayerBaseMultiInputs


class QMerge(Merge, QLayerBaseMultiInputs):  # type: ignore
    def call(self, inputs, training=None):
        if self.enable_ebops and training:
            self._compute_ebops([ops.shape(inp) for inp in inputs])
        qinputs = self.iqs(inputs, training=training)
        r = super().call(qinputs)
        return r

    def _compute_ebops(self, shapes):
        raise NotImplementedError


def _ebops_from_sum_bits_excl_max(self: QMerge, shapes, scale):
    bitss = [iq.bits_((1,) + shape[1:]) for iq, shape in zip(self.iqs, shapes)]
    _ebops = sum(bitss)
    _min = bitss[0]
    for bits in bitss[1:]:
        _min = ops.minimum(_min, bits)
    ebops = _ebops - ops.sum(_min)  # type: ignore
    ebops = ebops * scale  # TODO: better ebops cost model for accumulators
    self._ebops.assign(ops.cast(ebops, self._ebops.dtype))  # type: ignore
    self.add_loss(self.beta * ebops)


class QAdd(QMerge, Add):
    def _compute_ebops(self, shapes):
        _ebops_from_sum_bits_excl_max(self, shapes, scale=0.65)


class QAveragePow2(QAdd, Average):
    def build(self, input_shape):
        super().build(input_shape)
        self._scale = float(2. ** log2(1.0 / len(input_shape)))

    def _merge_function(self, inputs):
        r = super()._merge_function(inputs)


class QSubtract(QMerge, Subtract):
    def _compute_ebops(self, shapes):
        bits0, bits1 = (iq.bits_((1,) + shape[1:]) for iq, shape in zip(self.iqs, shapes))
        ebops = ops.sum(ops.maximum(bits0, bits1))
        ebops = ebops  # TODO: better ebops cost model for subtract
        self._ebops.assign(ops.cast(ebops, self._ebops.dtype))  # type: ignore
        self.add_loss(self.beta * ebops)


class QMultiply(QMerge, Multiply):
    def _compute_ebops(self, shapes):
        bitss = [iq.bits_((1,) + shape[1:]) for iq, shape in zip(self.iqs, shapes)]
        _ebops = bitss[0]
        for bits in bitss[1:]:
            _ebops = ops.multiply(_ebops, bits)
        ebops = ops.sum(_ebops)
        self._ebops.assign(ops.cast(ebops, self._ebops.dtype))  # type: ignore
        self.add_loss(self.beta * ebops)


class QDot(QMerge, Dot):
    def _compute_ebops(self, shapes):
        bits0, bits1 = (iq.bits_((1,) + shape[1:]) for iq, shape in zip(self.iqs, shapes))
        ebops = ops.sum(bits0 * bits1)
        self._ebops.assign(ops.cast(ebops, self._ebops.dtype))  # type: ignore
        self.add_loss(self.beta * ebops)


class QMaximum(QMerge, Maximum):
    def _compute_ebops(self, shapes):
        _ebops_from_sum_bits_excl_max(self, shapes, scale=1.)


class QMinimum(QMerge, Minimum):
    def _compute_ebops(self, shapes):
        _ebops_from_sum_bits_excl_max(self, shapes, scale=1.)


class QMatmul(QMerge):
    def build(self, input_shape):
        assert len(input_shape) == 2, "QMatmul requires exactly 2 inputs."
        super().build(input_shape)

    def _merge_function(self, inputs):
        inp1, inp2 = inputs
        return ops.matmul(inp1, inp2)

    def _compute_ebops(self, shapes):
        bits0, bits1 = (iq.bits_((1,) + shape[1:]) for iq, shape in zip(self.iqs, shapes))
        ebops = ops.matmul(bits0, bits1)
        self._ebops.assign(ops.cast(ebops, self._ebops.dtype))  # type: ignore
        self.add_loss(self.beta * ebops)


class QEinsum(QMerge):
    def __init__(self, equation, **kwargs):
        super().__init__(**kwargs)
        self.equation = equation
        self._ebops_equation = equation.split('->', 1)[0] + '->'

    def _merge_function(self, inputs):
        return ops.einsum(self.equation, *inputs)

    def _compute_ebops(self, shapes):
        bitss = [iq.bits_((1,) + shape[1:]) for iq, shape in zip(self.iqs, shapes)]
        ebops = ops.einsum(self._ebops_equation, *bitss)
        self._ebops.assign(ops.cast(ebops, self._ebops.dtype))  # type: ignore
        self.add_loss(self.beta * ebops)
