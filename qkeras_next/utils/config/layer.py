from typing import TypedDict, overload

from keras.api.initializers import Initializer


class GlobalConfig(TypedDict):
    beta0: float
    enable_ebops: bool


global_config = GlobalConfig(beta0=1e-5, enable_ebops=True)


class LayerConfigScope:
    @overload
    def __init__(self, *, beta0: float | None | Initializer = None, enable_ebops: bool | None = None):
        ...

    @overload
    def __init__(self, **kwargs):
        ...

    def __init__(self, **kwargs):
        self._override = kwargs

    def __enter__(self):
        for k, v in self._override.items():
            global_config[k] = v
