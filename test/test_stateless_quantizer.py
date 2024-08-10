import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qkeras_next.quantizer.fixed_point_ops import get_fixed_quantizer
from qkeras_next.quantizer.float_point_ops import float_decompose, float_quantize


@pytest.fixture(scope='module')
def x_kif():
    N = (100000,)
    key = jax.random.PRNGKey(int(os.environ.get('PYTEST_SEED', 0)))
    k1, k2, k3, k4 = jax.random.split(key, 4)
    k = jax.random.uniform(k1, N) > 0.5
    i = jax.random.randint(k2, N, -4, 8).astype(jnp.float32)
    f = jax.random.randint(k3, N, -4, 8).astype(jnp.float32)
    x = jax.random.normal(k4, N).astype(jnp.float32)
    f = jnp.maximum(f, -i)
    return x, k, i, f


@pytest.fixture(scope='module')
def xxx_mee0():
    N = (100000,)
    key = jax.random.PRNGKey(int(os.environ.get('PYTEST_SEED', 0)))
    k1, k2, k3, k4 = jax.random.split(key, 4)
    M = jax.random.randint(k1, N, 1, 9).astype(jnp.float32)
    E = jax.random.randint(k2, N, 1, 4).astype(jnp.float32)
    E0 = jax.random.randint(k3, N, -8, 8).astype(jnp.float32)
    x = jax.random.uniform(k4, N, jnp.float32, -1., 1.)

    sign = jnp.sign(x)
    x = jnp.abs(x)
    m_eps = 2.**-M
    _max = (2 - m_eps) * 2.**(2**(E - 1) - 1 + E0)
    _min_pos_normal = 2.**(-2**(E - 1) + E0 + 1)
    _min_pos_subnormal = m_eps * 2.**(-2**(E - 1) + E0)
    log_subnormal = jnp.log2(_min_pos_subnormal)
    log_normal = jnp.log2(_min_pos_normal)
    log_overflow = jnp.log2(_max)

    x_normal = 2.**(x * (log_overflow - log_normal) + log_normal) * sign
    x_subnormal = 2.**(x * (log_normal - log_subnormal) + log_subnormal - 1) * sign
    x_overflow = 2.**(x * 2 + log_overflow + 1) * sign
    xxx = (x_normal, x_subnormal, x_overflow)
    return xxx, M, E, E0


@pytest.mark.parametrize('round_mode', ['TRN', 'RND', 'S_RND', 'RND_CONV', 'S_RND_CONV'])
@pytest.mark.parametrize('overflow_mode', ['WRAP', 'SAT', 'SAT_SYM'])
def test_fixed_quantizer_inference(round_mode, overflow_mode, x_kif):
    quantizer = get_fixed_quantizer(round_mode, overflow_mode)
    assert callable(quantizer)
    x, k, i, f = x_kif
    upper = 2.**i - 2.**-f

    if overflow_mode != 'SAT_SYM':
        lower = -2.**i * k
    else:
        lower = -upper * k
    no_overflow = (x >= lower) & (x <= upper)
    overflow_up = x > upper
    overflow_down = x < lower
    xq = quantizer(x, k, i, f, False)
    xq_2 = quantizer(xq, k, i, f, False)
    assert np.all(xq >= lower) and np.all(xq <= upper)
    assert np.all(xq == xq_2)

    delta = 2.**-f[no_overflow]
    diff_no_overflow = (xq - x)[no_overflow]
    if round_mode == 'TRN':
        assert np.all(diff_no_overflow >= -delta)
        assert np.all(diff_no_overflow <= 0)
    if round_mode == 'RND':
        assert np.all(np.abs(diff_no_overflow) <= delta / 2)

    if overflow_mode in ('SAT', 'SAT_SYM'):
        assert np.all(xq[overflow_up] == upper[overflow_up])
        assert np.all(xq[overflow_down] == lower[overflow_down])


@pytest.mark.parametrize('round_mode', ['TRN', 'RND', 'S_RND', 'RND_CONV', 'S_RND_CONV'])
@pytest.mark.parametrize('overflow_mode', ['WRAP', 'SAT', 'SAT_SYM'])
def test_fixed_quantizer_train(round_mode, overflow_mode, x_kif):
    quantizer = get_fixed_quantizer(round_mode, overflow_mode)
    assert callable(quantizer)
    x, k, i, f = x_kif

    xq = quantizer(x, k, i, f, True)
    if 'overflow_mode' in ('SAT', 'SAT_SYM') and not round_mode.startswith('S_'):
        xq1 = quantizer(x, k, i, f, False)
        assert jnp.all(xq == xq1)

    def abs_quantization_err(x, k, i, f):
        xq = quantizer(x, k, i, f, True)
        err = jnp.abs(x - xq)
        return jnp.sum(err)

    dx, di, df = jax.grad(abs_quantization_err, (0, 2, 3))(x, 1, i, f)

    if overflow_mode == 'WRAP':
        assert jnp.all(dx == 0), f'X grad Error'
        assert jnp.all(di == 0), f'I grad Error'
        assert jnp.all(df < 0), f'F grad Error'
    elif overflow_mode in ('SAT', 'SAT_SYM'):
        for _dx in jnp.unique(dx):
            assert _dx in (-1, 0, 1), f'X grad Error'
        assert jnp.all(df <= 0), f'F grad Error'
        assert jnp.all(di <= 0), f'I grad Error'
        assert jnp.all((df < 0) | (di < 0) | (x == xq)), f'Grad Error for sat mode'


def test_float_quantizer_grad(xxx_mee0):
    xxx, M, E, E0 = xxx_mee0
    x_normal, x_subnormal, x_overflow = xxx

    def abs_quantization_err(x, M, E, E0):
        xq = float_quantize(x, M, E, E0)
        err = jnp.abs(x - xq)
        return jnp.sum(err)

    dx, dm, de, de0 = jax.grad(abs_quantization_err, range(4))(x_normal, M, E, E0)
    xq = float_quantize(x_normal, M, E, E0)
    mask = x_normal != xq
    assert jnp.all(dx == 0), f'Normal Number X grad Error'
    assert jnp.all(dm[mask] < 0), f'Normal Number M grad Error: max={jnp.max(dm[mask])}>0'
    assert jnp.all(de == 0), f'Normal Number E grad Error'
    assert jnp.all(de0 == 0), f'Normal Number E0 grad Error'

    dx, dm, de, de0 = jax.grad(abs_quantization_err, range(4))(x_subnormal, M, E, E0)
    xq = float_quantize(x_subnormal, M, E, E0)
    mask = x_subnormal != xq
    assert jnp.all(dx == 0), f'Subnormal Number X grad Error'
    assert jnp.all(dm[mask] < 0), f'Subnormal Number M grad Error: max={jnp.max(dm[mask])}>0'
    assert jnp.all(de[mask] < 0), f'Subnormal Number E grad Error: max={jnp.max(de[mask])}>0'
    assert jnp.all(de0[mask] > 0), f'Subnormal Number E0 grad Error: max={jnp.max(de0[mask])}>0'

    dx, dm, de, de0 = jax.grad(abs_quantization_err, range(4))(x_overflow, M, E, E0)
    assert jnp.all(dx == 0), f'Overflow Number X grad Error'
    assert jnp.all(dm == 0), f'Overflow Number M grad Error'
    assert jnp.all(de < 0), f'Overflow Number E grad Error: max={jnp.max(de)}>0'
    assert jnp.all(de0 < 0), f'Overflow Number E0 grad Error: max={jnp.max(de0)}>0'


def test_float_decompose(xxx_mee0):

    xxx, M, E, E0 = xxx_mee0
    x_normal, x_subnormal, x_overflow = xxx

    mm, ee = float_decompose(x_normal, M, E, E0)
    xq_ = mm * 2.**ee  # type: ignore
    xq = float_quantize(x_normal, M, E, E0)
    assert jnp.all(xq == xq_), f'Float Decompose Error @ Normal Number'
    assert jnp.all(jnp.abs(mm) < 2), f'Mantissa Error @ Normal Number'  # type: ignore
    assert jnp.all(jnp.abs(mm) >= 1), f'Mantissa Error @ Normal Number'  # type: ignore

    mm, ee = float_decompose(x_subnormal, M, E, E0)
    xq_ = mm * 2.**ee  # type: ignore
    xq = float_quantize(x_subnormal, M, E, E0)
    assert jnp.all(xq == xq_), f'Float Decompose Error @ Subnormal Number'
    assert jnp.all(jnp.abs(mm) < 1), f'Mantissa Error @ Subnormal Number'  # type: ignore

    mm, ee = float_decompose(x_overflow, M, E, E0)
    xq_ = mm * 2.**ee  # type: ignore
    xq = float_quantize(x_overflow, M, E, E0)
    assert jnp.all(xq == xq_), f'Float Decompose Error'
    assert jnp.all(jnp.abs(mm) < 2), f'Mantissa Error @ Overflow Number'  # type: ignore
    assert jnp.all(jnp.abs(mm) >= 1), f'Mantissa Error @ Overflow Number'  # type: ignore
