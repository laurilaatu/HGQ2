from collections.abc import Callable
from functools import singledispatch
from typing import Any, Optional, TypeVar

import keras
from keras import ops

round_mode_registry: dict[str, Callable[[Any, bool | None], Any]] = {}
round_mode_registry_scaled: dict[str, Callable[[Any, Any, bool | None], Any]] = {}
saturation_mode_registry: dict[str, Callable[[Any, Any, Any, Any, bool | None], Any]] = {}

T = TypeVar('T')


@ops.custom_gradient
def _clip(x, min_value, max_value):
    r = ops.clip(x, min_value, max_value)

    def grad(dy):
        dx = ops.where(x == r, dy, 0.)
        dmax = ops.where(x > max_value, dy, 0.)
        dmin = ops.where(x < min_value, dy, 0.)
        return dx, dmin, dmax
    return r, grad


def rnd_mode(names: str | list[str] | tuple[str, ...]):
    names = (names,) if isinstance(names, str) else names

    def inner(func):
        @keras.ops.custom_gradient
        def wrapper(x, f, training=None):
            scale = 2.**f
            sx = x * scale
            sxq = func(sx, training)
            xq = sxq / scale
            delta = xq - x

            def grad(dy):
                dx = dy
                df = - ops.log(2.) * delta * dy  # type: ignore
                return dx, df, None
            return xq, grad

        @keras.ops.custom_gradient
        def ste_wrapper(x, training=None):
            r = func(x, training)

            def grad(dy):
                return dy, None
            return r, grad

        for name in names:
            assert name not in round_mode_registry_scaled, f"Rounding mode '{name}' already exists."
            round_mode_registry_scaled[name] = wrapper
        round_mode_registry_scaled[func.__name__.upper()] = wrapper

        for name in names:
            assert name not in round_mode_registry, f"Rounding mode '{name}' already exists."
            round_mode_registry[name] = ste_wrapper
        round_mode_registry[func.__name__.upper()] = ste_wrapper

        return ste_wrapper
    return inner


@rnd_mode('TRN')
def floor(x, training=None):
    return ops.floor(x)


@rnd_mode('RND')
def round(x, training=None):
    return ops.floor(x + 0.5)


@rnd_mode('S_RND')
def stochastic(x, training=None):
    if training:
        return ops.floor(x + keras.random.uniform(x.shape))
    else:
        return ops.floor(x + 0.5)


@rnd_mode('RND_CONV')
def round_conv(x, training=None):
    return ops.round(x)


@rnd_mode('S_RND_CONV')
def stochastic_conv(x, training=None):
    if training:
        return ops.floor(x + keras.random.uniform(x.shape))
    else:
        return ops.round(x)


@singledispatch
def sat_mode(func: Callable):
    saturation_mode_registry[func.__name__.upper()] = func
    return func


@sat_mode.register
def _(name: str | list | tuple):
    names = (name,) if isinstance(name, str) else name

    def inner(func):
        for name in names:
            assert name not in saturation_mode_registry, f"Saturation mode '{name}' already exists."
            saturation_mode_registry[name] = func
        saturation_mode_registry[func.__name__.upper()] = func
        return func
    return inner


@sat_mode
def wrap(x, k, i, f, training=None):
    if training:
        return x

    xs = x
    bk = i + k
    bias = k * 2**(bk - 1)
    return ((xs + bias) % (2**bk) - bias)


@sat_mode('SAT')
def sat(x, k, i, f, training=None):
    f_eps = 2**(-f)
    __max = 2**i
    _max = __max - f_eps
    _min = -__max * k
    r = _clip(x, _min, _max)
    return r


@sat_mode('SAT_SYM')
def sat_sym(x, k, i, f, training=None):
    f_eps = 2**(-f)
    _max = 2**i - f_eps
    _min = -_max * k
    r = _clip(x, _min, _max)
    return r


def get_fixed_quantizer(round_mode: str = 'TRN', overflow_mode: str = 'WRAP') -> Callable[[T, Any, Any, Any, bool | None], T]:
    """Get a stateless fixed-point quantizer given the round and overflow mode.
    The quantizer is differentiable w.r.t. to the input and f, also i if using saturation overflow mode.

    Args:
        round_mode: round mode, one of
    """
    round_mode = round_mode.upper()
    overflow_mode = overflow_mode.upper()
    round_fn_scaled = round_mode_registry_scaled.get(round_mode, None)
    if round_fn_scaled is None:
        raise ValueError(f"Unknown rounding mode: {round_mode}")  # pragma: no cover
    sat_fn = saturation_mode_registry.get(overflow_mode, None)
    if sat_fn is None:
        raise ValueError(f"Unknown saturation mode: {overflow_mode}")  # pragma: no cover

    def quantizer(x, k, i, f, training: bool | None = None):
        """Stateless fixed-point quantizer.
        Args:
            x: input tensor
            k: number of fractional bits
            i: number of integer bits
            f: number of fractional bits
            training: training mode
        """
        qi = round(i, True)
        qf = round(f, True)

        xq = round_fn_scaled(x, qf, training)
        xq = sat_fn(xq, k, qi, qf, training)
        return xq
    return quantizer
