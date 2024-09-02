from collections.abc import Callable, Sequence

import numpy as np
from keras import Variable, ops
from keras.api.callbacks import Callback
from keras.api.models import Model


class BetaScheduler(Callback):
    """Schedule the beta value of the Q Layers.

    Parameters
    ----------
    beta_fn : Callable[[int], float]
        A function that takes the current epoch and returns the beta value.
    """

    def __init__(self, beta_fn: Callable[[int], float]):
        self.beta_fn = beta_fn

    def on_epoch_begin(self, epoch, logs=None):
        assert isinstance(self.model, Model)

        beta = self.beta_fn(epoch)
        for layer in self.model._flatten_layers():
            if hasattr(layer, '_beta'):
                layer._beta.assign(ops.convert_to_tensor(beta, dtype=layer._beta.dtype))

    def on_epoch_end(self, epoch, logs=None):
        assert isinstance(logs, dict)
        logs['beta'] = self.beta_fn(epoch)


class PieceWiseSchedule:
    """Get interpolated schedule from key points.

    Parameters
    ----------
    intervals : sequence of tuple[int, float, str]
        The key points of the schedule. Each tuple contains the starting epoch, beta, and interpolation for the interval.
        # Example: [(0, 0, 'linear'), (10, 1e-5, 'log'), (20, 1e-3, 'linear')] will start with beta=0, then increase to 1e-5 in 10 epochs linearly, and increase to 1e-3 in another 10 epochs logarithmically. beta will stay at 1e-3 after 20 epochs.
    total_epochs : int
        The total number of epochs. Much be greater or equal to the last epoch in the intervals.
    """

    def __init__(self, intervals: Sequence[tuple[int, float, str]], total_epochs):
        epochs = []
        betas = []
        interpolations = []
        for epoch, beta, interp in intervals:
            epochs.append(epoch)
            betas.append(beta)
            assert interp in ['linear', 'log']
            interpolations.append(interp == 'log')
        epochs = np.array(epochs + [total_epochs])
        assert np.all(np.diff(epochs) >= 0)
        betas = np.array(betas)
        interpolations = np.array(interpolations)

        self.epochs = epochs
        self.betas = betas
        self.interpolations = interpolations
        self.total_epochs = total_epochs

    def __call__(self, epoch):
        if epoch >= self.total_epochs:
            return self.betas[-1, -1]
        idx = np.searchsorted(self.epochs, epoch, side='right') - 1
        beta0, beta1 = self.betas[idx]
        epoch0, epoch1 = self.epochs[idx], self.epochs[idx + 1]
        if self.interpolations[idx]:
            beta = beta0 * (beta1 / beta0) ** ((epoch - epoch0) / (epoch1 - epoch0))
        else:
            beta = beta0 + (beta1 - beta0) * (epoch - epoch0) / (epoch1 - epoch0)
        return float(beta)
