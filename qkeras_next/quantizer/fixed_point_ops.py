from collections.abc import Callable
from functools import singledispatch
from typing import Any, Optional, TypeVar

import keras
from keras import ops
from keras.api.random import SeedGenerator

round_mode_registry: dict[str, Callable[[Any, bool | None], Any]] = {}
round_mode_registry_scaled: dict[str, Callable[[Any, Any, bool | None, SeedGenerator | None], Any]] = {}
saturation_mode_registry: dict[str, Callable[[Any, Any, Any, Any, bool | None], Any]] = {}

T = TypeVar('T')


@ops.custom_gradient
def _clip(x, min_value, max_value):
    r = ops.clip(x, min_value, max_value)

    def grad(*args, upstream=None):
        mask1 = x > max_value
        mask2 = x < min_value
        if upstream is None:
            (upstream,) = args
        dy = upstream
        dx = ops.where(~(mask1 | mask2), dy, 0.)
        dmax = ops.where(mask1, dy, 0.)
        dmin = ops.where(mask2, dy, 0.)
        return dx, dmin, dmax
    return r, grad


def rnd_mode(names: str | list[str] | tuple[str, ...]):
    names = (names,) if isinstance(names, str) else names

    def inner(func):
        @keras.ops.custom_gradient
        def inner_wrapper(x, f):
            scale = 2.**f
            sx = x * scale
            sxq = func(sx)
            xq = sxq / scale
            delta = xq - x

            def grad(*args, upstream=None):
                if upstream is None:
                    (upstream,) = args
                dy = upstream
                dx = dy
                df = - ops.log(2.) * delta * dy  # type: ignore
                return dx, df
            return ops.stop_gradient(xq), grad

        @keras.ops.custom_gradient
        def inner_ste_wrapper(x):
            r = func(x)

            def grad(*args, upstream=None):
                if upstream is None:
                    (upstream,) = args
                dy = upstream
                return dy
            return ops.stop_gradient(r), grad

        def wrapper(x, f, training=None, seed_gen=None):  # type: ignore
            return inner_wrapper(x, f)

        def ste_wrapper(x, training=None):  # type: ignore
            return inner_ste_wrapper(x)

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
def floor(x):
    return ops.floor(x)


@rnd_mode('RND')
def round(x):
    return ops.floor(x + 0.5)


@rnd_mode('RND_CONV')
def round_conv(x):
    return ops.round(x)


def sat_mode(name: str | list | tuple):
    names = (name,) if isinstance(name, str) else name

    def inner(func):
        for name in names:
            assert name not in saturation_mode_registry, f"Saturation mode '{name}' already exists."
            saturation_mode_registry[name] = func
        saturation_mode_registry[func.__name__.upper()] = func
        return func
    return inner


@sat_mode('WRAP')
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


def get_fixed_quantizer(round_mode: str = 'TRN', overflow_mode: str = 'WRAP') -> Callable[[T, Any, Any, Any, bool | None, SeedGenerator | None], T]:
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

    def quantizer(x: T, k: T, i: T, f: T, training: bool | None = None, seed_gen: SeedGenerator | None = None) -> T:
        """Stateless fixed-point quantizer.
        Args:
            x: input tensor
            k: number of fractional bits
            i: number of integer bits
            f: number of fractional bits
            training: training mode
        """
        i = ops.stop_gradient(ops.maximum(i, -f)) + (i - ops.stop_gradient(i))  # type: ignore

        # Workaround for gradient computation around 0.
        # When have small rounded to boundary, grad on f presents despite the value will be clipped off anyway.
        # Thus have saturation before rounding, except for wrap mode, which doesn't round during training.
        if overflow_mode != 'WRAP':
            x = sat_fn(x, k, i, f, training)
        x = round_fn_scaled(x, f, training, seed_gen)
        if overflow_mode == 'WRAP':
            x = sat_fn(x, k, i, f, training)
        return x
    return quantizer


# ======================= STOCHASTIC ROUNDING =======================
def register_stochastic_rounding():
    @keras.ops.custom_gradient
    def stochastic_scaled(x, f, noise):
        scale = 2.**f
        sx = x * scale
        sxq = ops.floor(sx + noise)
        xq = sxq / scale
        delta = xq - x

        def grad(*args, upstream=None):
            if upstream is None:
                (upstream,) = args
            dy = upstream
            dx = dy
            df = - ops.log(2.) * delta * dy  # type: ignore
            return dx, df, None
        return ops.stop_gradient(xq), grad

    rnd_scaled = round_mode_registry_scaled['RND']
    rnd_conv_scaled = round_mode_registry_scaled['RND_CONV']

    def outer_rnd(x, f, training=None, seed_gen=None):
        if training:
            noise = keras.random.uniform(ops.shape(x), seed=seed_gen)
            return stochastic_scaled(x, f, noise)
        else:
            return rnd_scaled(x, f, training, seed_gen)

    def outer_rnd_conv(x, f, training=None, seed_gen=None):
        if training:
            noise = keras.random.uniform(ops.shape(x), seed=seed_gen)
            return stochastic_scaled(x, f, noise)
        else:
            return rnd_conv_scaled(x, f, training, seed_gen)

    round_mode_registry_scaled['S_RND'] = outer_rnd
    round_mode_registry_scaled['S_RND_CONV'] = outer_rnd_conv


register_stochastic_rounding()
