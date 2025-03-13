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
    if reset:
        _reset_minmax(model)
    record: dict[str, int] = {}
    if not isinstance(data, PyDataset):
        data = []
        for i in range(0, len(data), batch_size):
            data.append((data[i : i + batch_size], None))
    results = []
    use_pbar = verbose is True or verbose > 1
    with tqdm(data, leave=False, disable=not use_pbar) as pbar:  # type: ignore
        for i, (x, _) in enumerate(pbar):  # type: ignore
            if len(data) == i:  # type: ignore
                break  # keras dataloader iterator is infinite
            r = model(x, training=TrainingFlagWrapper('tracing'))
            if return_results:
                results.append(r)

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
        return keras.ops.concatenate(results)
