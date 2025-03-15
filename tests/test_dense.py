import pytest

from hgq.layers import QDense
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
