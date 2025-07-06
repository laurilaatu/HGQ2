from keras import ops
from keras.layers import (
    AveragePooling1D,
    AveragePooling2D,
    AveragePooling3D,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    GlobalAveragePooling3D,
    GlobalMaxPooling1D,
    GlobalMaxPooling2D,
    GlobalMaxPooling3D,
    MaxPooling1D,
    MaxPooling2D,
    MaxPooling3D,
)

from .core.base import QLayerBaseSingleInput


class QBasePooling(QLayerBaseSingleInput):
    def _compute_ebops(self, shape):
        bw_inp = self.iq.bits_(shape)
        return ops.sum(bw_inp)

    def call(self, inputs, training=None):
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)
        return super().call(inputs)


class QAveragePooling1D(QBasePooling, AveragePooling1D):
    pass


class QAveragePooling2D(QBasePooling, AveragePooling2D):
    pass


class QAveragePooling3D(QBasePooling, AveragePooling3D):
    pass


class QMaxPooling1D(QBasePooling, MaxPooling1D):
    pass


class QMaxPooling2D(QBasePooling, MaxPooling2D):
    pass


class QMaxPooling3D(QBasePooling, MaxPooling3D):
    pass


class QGlobalAveragePooling1D(QBasePooling, GlobalAveragePooling1D):  # type: ignore
    pass


class QGlobalAveragePooling2D(QBasePooling, GlobalAveragePooling2D):
    pass


class QGlobalAveragePooling3D(QBasePooling, GlobalAveragePooling3D):
    pass


class QGlobalMaxPooling1D(QBasePooling, GlobalMaxPooling1D):
    pass


class QGlobalMaxPooling2D(QBasePooling, GlobalMaxPooling2D):
    pass


class QGlobalMaxPooling3D(QBasePooling, GlobalMaxPooling3D):
    pass


__all__ = [
    'QMaxPooling1D',
    'QMaxPooling2D',
    'QMaxPooling3D',
    'QAveragePooling1D',
    'QAveragePooling2D',
    'QAveragePooling3D',
    'QGlobalMaxPooling1D',
    'QGlobalMaxPooling2D',
    'QGlobalMaxPooling3D',
    'QGlobalAveragePooling1D',
    'QGlobalAveragePooling2D',
    'QGlobalAveragePooling3D',
]
