from pathlib import Path

import cppyy
import numpy as np
import pytest

from squark.quantizer.internal import float_quantize, get_fixed_quantizer


@pytest.fixture(scope='session')
def register_cpp():
    path = Path(__file__).parent
    cppyy.add_include_path(f'{path}/cpp_source')
    cppyy.add_include_path(f'{path}/cpp_source/ap_types')
    cppyy.include('quantizers.h')


@pytest.fixture(scope='session')
def data():
    arr = np.random.randint(-2**15, 2**15, 1000) * 2**-8
    return arr.astype(np.float32)


def c_quantize_fixed(x, k, i, f, round_mode, overflow_mode):
    W, I = k + i + f, k + i
    c_round_mode = getattr(cppyy.gbl, f'AP_{round_mode}')
    c_overflow_mode = getattr(cppyy.gbl, f'AP_{overflow_mode}')

    fn = cppyy.gbl.qkn_test.fixedq_vec[W, I, k, c_round_mode, c_overflow_mode]

    r = fn(x, len(x))
    return r


def c_quantize_float(x, M, E, E0):
    fn = cppyy.gbl.qkn_test.floatq_vec[M, E, E0]
    r = fn(x, len(x))
    return r


@pytest.mark.parametrize('fixed_round_mode', ['TRN'])
@pytest.mark.parametrize('fixed_overflow_mode', ['WRAP', 'SAT_SYM'])
@pytest.mark.parametrize('k', [0, 1])
@pytest.mark.parametrize('i', [3])
@pytest.mark.parametrize('f', [4])
@pytest.mark.parametrize('M', [2, 4])
@pytest.mark.parametrize('E', [2, 4])
@pytest.mark.parametrize('E0', [1, 8])
def test_fixed_quantizer_inference(fixed_round_mode, fixed_overflow_mode, k, i, f, M, E, E0, data, register_cpp):

    quantizer = get_fixed_quantizer(fixed_round_mode, fixed_overflow_mode)
    arr_py_fixed = np.array(quantizer(data, float(k), float(i), float(f), False, None))
    arr_py_float = np.array(float_quantize(data, M, E, E0))
    arr_c_fixed = c_quantize_fixed(data, k, i, f, fixed_round_mode, fixed_overflow_mode)
    arr_c_float = c_quantize_float(data, M, E, E0)
    arr_c_fixed_np = np.array([float(x) for x in arr_c_fixed])
    arr_c_float_np = np.array([float(x) for x in arr_c_float])

    mismatch = np.where(arr_py_fixed != arr_c_fixed_np)[0]
    assert len(mismatch) == 0, f'''Fixed quantizer has inconsistent behavior with C++ implementation:
        [* {len(mismatch)} mismatches, {min(len(mismatch), 5)} shown *]
        C++: {arr_c_fixed_np[mismatch][:5]}
        Py: {arr_py_fixed[mismatch][:5]}
        in: {data[mismatch][:5]}
    '''

    mismatch = np.where(arr_py_float != arr_c_float_np)[0]
    assert len(mismatch) == 0, f'''Float quantizer has inconsistent behavior with C++ implementation:
        [* {len(mismatch)} mismatches, {min(len(mismatch), 5)} shown *]
        [* Up to 5 shown]
        C++: {arr_c_float_np[mismatch][:5]}
        Py: {arr_py_float[mismatch][:5]}
        in: {data[mismatch][:5]}
    '''

    mult_c = np.array([(fp * fx).to_double() for fx, fp in zip(arr_c_fixed, arr_c_float)])
    mult_py = arr_c_fixed_np * arr_c_float_np

    mismatch = np.where(mult_py != mult_c)[0]
    print(len(mismatch))
    assert len(mismatch) == 0, f'''Multiplication has inconsistent behavior with C++ implementation:
        [* {len(mismatch)} mismatches, {min(len(mismatch), 5)} shown *]
        C++: {mult_c[mismatch][:5]}
        Py: {mult_py[mismatch][:5]}
        fx in: {arr_c_fixed_np[mismatch][:5]}
        fp in: {arr_c_float_np[mismatch][:5]}
    '''
