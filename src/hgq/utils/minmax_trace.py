from collections.abc import Sequence

import keras
from keras import ops
from keras.utils import PyDataset
from numpy.typing import ArrayLike
from tqdm import tqdm

from .dataset import Dataset


def _reset_minmax(layer: keras.Layer):
    if hasattr(layer, '_i_decay_speed'):
        # WRAP-like overflow mode
        shape, dtype = layer._i.shape, layer._i.dtype
        layer._i.assign(keras.ops.full(shape, 1e-9, dtype=dtype))
        shape, dtype = layer._k.shape, layer._k.dtype
        layer._k.assign(keras.ops.zeros(shape, dtype=dtype))
    for sublayer in layer._layers:
        _reset_minmax(sublayer)


class TrainingFlagWrapper:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.value == other

    def __bool__(self):
        return self.value is True


def trace_minmax(
    model: keras.Model,
    data: ArrayLike | Sequence[ArrayLike] | PyDataset,
    reset=True,
    batch_size=1024,
    verbose: int | bool = 0,
    return_results=False,
):
    n_outputs = len(model.outputs)

    if not isinstance(data, PyDataset):
        data = Dataset(data, batch_size=batch_size, device='none')

    if reset:
        _reset_minmax(model)
    record: dict[str, int] = {}

    results = []
    use_pbar = verbose is True or verbose > 1
    n_batch = len(data)  # type: ignore
    n_outputs = len(model.outputs)

    with tqdm(total=n_batch, leave=False, disable=not use_pbar) as pbar:  # type: ignore
        for i in range(n_batch):
            r = model(data[i][0], training=TrainingFlagWrapper('tracing'))
            if return_results:
                results.append(r)
            pbar.update(1)

    if verbose:
        record = {}
        for layer in model.layers:
            if getattr(layer, 'enable_ebops', False):
                record[layer.name] = int(layer.ebops)  # type: ignore
        width = max(max(map(len, record.keys())), 5)
        for k, v in record.items():
            print(f'{k:{width}}: {v}')
        print(f'Total: {sum(record.values())}')

    if return_results:
        if n_outputs == 1:
            return ops.stop_gradient(ops.concatenate(results))
        return tuple(ops.stop_gradient(ops.concatenate([r[i] for r in results])) for i in range(n_outputs))
