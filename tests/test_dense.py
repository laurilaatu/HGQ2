import pytest

from hgq.layers import QBatchNormDense, QDense, QEinsumDense, QEinsumDenseBatchnorm
from tests.base import LayerTestBase


class TestDense(LayerTestBase):
    layer_cls = QDense

    @pytest.fixture(params=[8])  # Test different output sizes
    def units(self, request):
        return request.param

    @pytest.fixture(params=[None, 'relu'])  # Test with and without activation
    def activation(self, request):
        return request.param

    @pytest.fixture(params=[(8, 8), (12,)])
    def input_shapes(self, request):
        return request.param

    @pytest.fixture
    def layer_kwargs(self, units, activation):
        return {'units': units, 'activation': activation}


class TestBatchNormDense(TestDense):
    layer_cls = QBatchNormDense


class TestEinsumDense(LayerTestBase):
    layer_cls = QEinsumDense

    @pytest.fixture(params=['Bij,ijk->Bik', 'Babc,bcd->Babd'])
    def equation(self, request):
        return request.param

    @pytest.fixture
    def input_shapes(self, equation: str):
        if equation == 'Bij,ijk->Bik':
            return (3, 4)
        return (3, 4, 5)

    @pytest.fixture
    def output_shape(self, equation: str):
        if equation == 'Bij,ijk->Bik':
            return (3, 6)
        return (3, 4, 2)

    @pytest.fixture(params=[False, True])
    def bias_axes(self, equation: str, request) -> str | None:
        if not request.param:
            return None
        if equation == 'Bij,ijk->Bik':
            return 'ik'
        return 'ad'

    @pytest.fixture
    def layer_kwargs(self, equation: str, output_shape: tuple[int, ...], bias_axes: str | None):
        return {'equation': equation, 'output_shape': output_shape, 'bias_axes': bias_axes, 'activation': None}

    @pytest.fixture(params=[True])
    def use_parallel_io(self, request) -> bool:
        return request.param


class TestEinsumDenseBatchnorm(TestEinsumDense):
    layer_cls = QEinsumDenseBatchnorm

    @pytest.fixture()
    def bias_axes(self, equation: str, request):
        if equation == 'Bij,ijk->Bik':
            return 'ik'
        return 'bd'
