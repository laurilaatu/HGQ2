from keras import ops
from keras.api import Model
from keras.api.callbacks import Callback

from qkeras_next.layers import QLayerBase


class FreeEBOPs(Callback):
    def on_epoch_end(self, epoch, logs=None):
        assert logs is not None
        assert isinstance(self.model, Model)
        ebops = 0
        for layer in self.model._flatten_layers():
            if isinstance(layer, QLayerBase):
                ebops += int(ops.convert_to_numpy(layer.ebops))
        logs['ebops'] = ebops
