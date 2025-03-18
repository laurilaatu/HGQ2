import os
from collections.abc import Sequence
from typing import Any

import keras
import numpy as np
import pytest
from hls4ml.converters import convert_from_keras_model
from keras import ops

from hgq.config import QuantizerConfigScope
from hgq.layers import QLayerBase
from hgq.quantizer.internal import FixedPointQuantizerKBI, FixedPointQuantizerKIF
from hgq.utils import trace_minmax


class CtxGlue:
    def __init__(self, *ctxs):
        self.ctxs = ctxs

    def __enter__(self):
        for ctx in self.ctxs:
            ctx.__enter__()

    def __exit__(self, *args):
        for ctx in self.ctxs:
            ctx.__exit__(*args)


def _assert_equal(a: np.ndarray, b: np.ndarray):
    a, b = np.asanyarray(a).ravel(), np.asanyarray(b).ravel()
    # if keras.backend.backend() != 'torch':
    mismatches = np.where(a != b)[0]
    # else:
    #     # Torch has some extra numerical errors by default...
    #     mismatches = np.where(np.abs(a - b) > 1e-6)[0]
    a_sample = a[mismatches[:5]]
    b_sample = b[mismatches[:5]]
    msg = f"""keras - c synth mismatch. {len(mismatches)} out of {len(a)} samples are different
    Sample:
    {a_sample}
    vs
    {b_sample}
    """
    assert len(mismatches) == 0, msg


class LayerTestBase:
    """Base class for testing HGQ layers"""

    layer_cls: type[QLayerBase] = QLayerBase
    custom_objects = {}

    @pytest.fixture(params=[True, False])
    def use_parallel_io(self, request) -> bool:
        return request.param

    @pytest.fixture(params=['kif', 'kbi'])
    def q_type(self, request) -> str:
        return request.param

    @pytest.fixture(params=['WRAP', 'SAT'])
    def overflow_mode(self, request) -> str:
        return request.param

    @pytest.fixture(params=['TRN', 'RND'])
    def round_mode(self, request) -> str:
        return request.param

    @pytest.fixture
    def ctx_scope(self, use_parallel_io: bool, q_type: str, overflow_mode: str, round_mode: str):
        heterogeneous_axis, homogeneous_axis = (None, (0,)) if use_parallel_io else ((), None)
        scope_w = QuantizerConfigScope(
            default_q_type=q_type,
            heterogeneous_axis=None,
            homogeneous_axis=(),
            overflow_mode=overflow_mode,
            round_mode=round_mode,
        )
        scope_a = QuantizerConfigScope(
            default_q_type=q_type,
            place='datalane',
            heterogeneous_axis=heterogeneous_axis,
            homogeneous_axis=homogeneous_axis,
            overflow_mode=overflow_mode,
            round_mode=round_mode,
        )
        return CtxGlue(scope_w, scope_a)

    @pytest.fixture
    def layer_kwargs(self, *args, **kwargs) -> dict[str, Any]:
        """Override this to return kwargs needed for layer construction"""
        return {}

    @pytest.fixture
    def input_shapes(self, *args, **kwargs) -> tuple[tuple[int, ...], ...] | tuple[int, ...]:
        """Override this to return test input shape"""
        raise NotImplementedError

    @pytest.fixture
    def input_data(self, input_shapes, N: int = 5000) -> np.ndarray | tuple[np.ndarray, ...]:
        """Generate test input data"""
        if isinstance(input_shapes[0], int):
            return np.random.randn(N, *input_shapes).astype(np.float32)
        dat = tuple(np.random.randn(N, *shape).astype(np.float32) for shape in input_shapes)  # type: ignore
        if len(dat) == 1:
            return dat[0]
        return dat

    @pytest.fixture
    def layer(self, layer_kwargs, ctx_scope):
        """Create layer instance"""
        with ctx_scope:
            return self.layer_cls(**layer_kwargs)

    @pytest.fixture
    def model(self, layer, input_shapes, use_parallel_io):
        """Create test model with the layer"""
        if isinstance(input_shapes[0], int):
            input_shapes = (input_shapes,)
        inputs = [keras.layers.Input(shape=shape) for shape in input_shapes]
        if len(inputs) == 1:
            inputs = inputs[0]
        outputs = layer(inputs)
        model = keras.Model(inputs, outputs)

        self.perturbe_bw(use_parallel_io, model)
        return model

    def perturbe_bw(self, use_parallel_io, model):
        if use_parallel_io:
            for _layer in model._flatten_layers(False):
                if isinstance(_layer, FixedPointQuantizerKBI):
                    b = np.random.randint(0, 8, _layer._b.shape)
                    i = np.array(ops.stop_gradient(_layer.i))
                    b = np.minimum(b, 12 - i)
                    if np.all(b == 0):
                        b.ravel()[0] = 1
                    _layer._b.assign(ops.array(b))
                if isinstance(_layer, FixedPointQuantizerKIF):
                    f = np.random.randint(2, 8, _layer._f.shape)
                    i = np.array(ops.stop_gradient(_layer.i))
                    f = np.minimum(f, 12 - i)
                    if np.all(i + f == 0):
                        f.ravel()[0] = 1
                    _layer._f.assign(ops.array(f))
        for _layer in model._flatten_layers(False):
            # Randomize activation values
            if hasattr(_layer, 'bias') and isinstance(_layer.bias, keras.Variable):
                bias = np.random.randn(*_layer.bias.shape)
                _layer.bias.assign(ops.array(bias))

    @pytest.mark.parametrize('format', ['h5', 'keras'])
    def test_serialization(self, model, input_data, format: str, temp_directory: str):
        """Test layer serialization/deserialization"""

        model.compile(optimizer='adam', loss='mean_squared_error')
        original_output = ops.stop_gradient(model(input_data))

        save_path = f'{temp_directory}/model.{format}'
        model.save(save_path)
        loaded_model = keras.models.load_model(save_path, custom_objects=self.custom_objects)
        loaded_output = ops.stop_gradient(loaded_model(input_data))  # type: ignore

        np.testing.assert_array_equal(original_output, loaded_output)  # type: ignore

        os.system(f"rm -rf '{temp_directory}'")

    def test_hls4ml_conversion(
        self, model: keras.Model, input_data: np.ndarray, temp_directory: str, use_parallel_io: bool, q_type: str
    ):
        """Test hls4ml conversion and bit-exactness"""

        trace_keras_output_0 = trace_minmax(model, input_data, return_results=True, verbose=2)

        hls_model = convert_from_keras_model(
            model,
            output_dir=temp_directory,
            backend='Vitis',
            io_type='io_parallel' if use_parallel_io else 'io_stream',
        )

        hls_model.compile()
        if q_type == 'kif':
            keras_output_0 = model.predict(input_data, batch_size=5000)
            self.assert_equal(keras_output_0, trace_keras_output_0)

        if isinstance(input_data, Sequence):
            input_data = tuple(_input_data * 2 for _input_data in input_data)  # type: ignore
        else:
            input_data = input_data * 2  # Test for overflow in the same time

        keras_output = model.predict(input_data, batch_size=5000)
        hls_output: np.ndarray = hls_model.predict(input_data).reshape(keras_output.shape)  # type: ignore
        self.assert_equal(keras_output, hls_output)

        os.system(f"rm -rf '{temp_directory}'")

    def assert_equal(self, keras_output, hls_output):
        _assert_equal(keras_output, hls_output)

    def test_training(self, model: keras.Model, input_data: np.ndarray, overflow_mode: str, *args, **kwargs):
        """Test basic training step"""
        # Add a loss layer for testing
        model_wrap = keras.Sequential([model, keras.layers.Flatten(), keras.layers.Dense(1)])

        initial_weights_np = [w.numpy() for w in model.trainable_variables]

        opt = keras.optimizers.SGD(learning_rate=1.0)
        loss = keras.losses.MeanSquaredError()
        model(input_data, training=True)  # Adapt init bitwidth

        data_len = len(input_data[0]) if isinstance(input_data, Sequence) else len(input_data)
        labels = ops.array(np.random.rand(data_len), dtype='float32')
        model_wrap.compile(optimizer=opt, loss=loss)  # type: ignore
        r0 = model_wrap.train_on_batch(input_data, labels)

        trained_weights = [w for w in model.trainable_variables]

        r1 = loss(labels, model_wrap(input_data))

        assert r1 != r0, f'Loss did not change: {r0} -> {r1}'

        boom = []
        for w0, w1 in zip(initial_weights_np, trained_weights):
            if w1.name in 'bif':
                continue
            if np.prod(w1.shape) < 10 and overflow_mode == 'SAT':
                # Overflowing weight doesn't receive grad in SAT mode
                # Chance of all overflow is high for small-sized weights, skip them
                continue
            if np.array_equal(w0, w1.numpy()):
                # if w1.path == 'q_multi_head_attention/key/bias':
                #     continue
                boom.append(f'{w1.path}')
        assert not boom, f"Weight {' AND '.join(boom)} did not change"
        assert any(np.any(w0 != w1.numpy()) for w0, w1 in zip(initial_weights_np, trained_weights) if w1.name in 'bif')
