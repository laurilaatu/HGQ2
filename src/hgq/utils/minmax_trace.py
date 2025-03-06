import keras


def set_trace_mode(layer: keras.Model, trace: bool):
    from ..quantizer.internal import FixedPointQuantizerBase

    if isinstance(layer, FixedPointQuantizerBase):
        if hasattr(layer, '_i_decay_speed'):
            i_decay_speed = layer.i_decay_speed
            if not trace ^ (i_decay_speed < 0):
                return
            # Tracemode is on when i_decay_speed < 0
            i_decay_speed = -(i_decay_speed + 1)
            layer._i_decay_speed.assign(i_decay_speed)
            if trace:
                layer._i.assign(keras.ops.full_like(layer._i, -1e9))
                layer._k.assign(keras.ops.zeros_like(layer._k))
    for sublayer in layer._layers:
        sublayer: keras.layers.Layer
        set_trace_mode(sublayer, trace)


class trace_mode:
    def __init__(self, model: keras.Model):
        self.model = model

    def __enter__(self):
        set_trace_mode(self.model, True)

    def __exit__(self, *args):
        set_trace_mode(self.model, False)
