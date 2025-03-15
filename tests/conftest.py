import os
import random
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='session', autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility"""
    np.random.seed(42)
    random.seed(42)
    backend = os.environ.get('KERAS_BACKEND', 'tensorflow')
    match backend:
        case 'tensorflow':
            import tensorflow as tf

            tf.random.set_seed(42)
        case 'torch':
            import torch

            torch.manual_seed(42)
        case 'jax':
            pass
        case _:
            raise ValueError(f'Unknown backend: {backend}')


@pytest.fixture(scope='session', autouse=True)
def set_hls4ml_configs():
    """Set default hls4ml configuration"""
    os.environ['HLS4ML_BACKEND'] = 'Vivado'


@pytest.fixture(scope='function')
def temp_directory(request: pytest.FixtureRequest):
    root = Path(os.environ.get('HGQ2_TEST_DIR', '/tmp/hgq2_test'))
    root.mkdir(exist_ok=True)

    test_name = request.node.name
    cls_name = request.cls.__name__ if request.cls else None
    if cls_name is None:
        test_dir = root / test_name
    else:
        test_dir = root / f'{cls_name}.{test_name}'
    test_dir.mkdir(exist_ok=True)
    return str(test_dir)
