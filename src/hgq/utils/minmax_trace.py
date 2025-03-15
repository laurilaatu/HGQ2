import keras
from keras.api.utils import PyDataset
from numpy.typing import ArrayLike
from tqdm import tqdm


def _reset_minmax(layer: keras.Layer):
    if hasattr(layer, '_i_decay_speed'):
        # WRAP-like overflow mode
        layer._i.assign(keras.ops.full_like(layer._i, -1e9))
        layer._k.assign(keras.ops.zeros_like(layer._k))
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
    model: keras.Model, data: ArrayLike | PyDataset, reset=True, batch_size=1024, verbose: int | bool = 0, return_results=False
):
    if not isinstance(data, PyDataset):
        _data = []
        for i in range(0, len(data), batch_size):  # type: ignore
            _data.append((data[i : i + batch_size], None))  # type: ignore
        data = _data

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
            return keras.ops.concatenate(results)
        return tuple(keras.ops.concatenate([r[i] for r in results]) for i in range(n_outputs))
