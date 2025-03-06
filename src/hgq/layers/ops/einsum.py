from keras.api import ops

from ..core import QLayerBaseMultiInputs


class QMatmul(QLayerBaseMultiInputs):
    def build(self, input_shape):
        assert len(input_shape) == 2, 'QMatmul requires exactly 2 inputs.'
        super().build(input_shape)

    def call(self, inputs, training=None):
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)
        inp1, inp2 = inputs
        return ops.matmul(inp1, inp2)

    def _compute_ebops(self, shapes):
        bits0, bits1 = (iq.bits_(shape) for iq, shape in zip(self.iq, shapes))
        ebops = ops.sum(ops.matmul(bits0, bits1))
        return ebops


class QEinsum(QLayerBaseMultiInputs):
    def __init__(self, equation, **kwargs):
        super().__init__(**kwargs)
        self.equation = equation
        self._ebops_equation = equation.split('->', 1)[0] + '->'

    def call(self, inputs, training=None):
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)
        return ops.einsum(self.equation, *inputs)

    def _compute_ebops(self, shapes):
        bitss = [iq.bits_(shape) for iq, shape in zip(self.iq, shapes)]
        ebops = ops.einsum(self._ebops_equation, *bitss)
        return ebops
