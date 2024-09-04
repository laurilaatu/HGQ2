import inspect
import math
from collections.abc import Sized
from typing import Any

from keras.api.layers import Dropout, MultiHeadAttention
from keras.src.layers.attention.multi_head_attention import _build_attention_equation, _build_proj_equation

from ..utils.config.quantizer import QuantizerConfig
from .core.base import QLayerBase
from .core.einsum_dense import QEinsumDense
from .softmax import QSoftmax


def gather_vars_to_kwargs() -> dict[str, Any]:
    vars = inspect.getouterframes(inspect.currentframe(), 2)[1][0].f_locals
    kwarg = vars.pop('kwargs', {})
    kwarg.update(vars)
    for k in list(kwarg.keys()):
        if k.startswith('__') and k.endswith('__'):
            del kwarg[k]
    return kwarg


def _get_output_shape(output_rank, known_last_dims, input_shape):
    n = (output_rank - len(known_last_dims))
    return list(input_shape[1:n + 1]) + list(known_last_dims)


class QMultiHeadAttention(MultiHeadAttention, QLayerBase):
    def __init__(
        self,
        num_heads,
        key_dim,
        value_dim=None,
        dropout=0.0,
        use_bias=True,
        output_shape=None,
        attention_axes=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        seed=None,
        stable_softmax=False,
        qkv_iq_conf: QuantizerConfig | None = None,
        qkv_kq_conf: QuantizerConfig | None = None,
        qkv_bq_conf: QuantizerConfig | None = None,
        softmax_iq_conf: QuantizerConfig | None = None,
        softmax_exp_iq_conf: QuantizerConfig | None = None,
        softmax_exp_oq_conf: QuantizerConfig | None = None,
        softmax_inv_iq_conf: QuantizerConfig | None = None,
        softmax_inv_oq_conf: QuantizerConfig | None = None,
        **kwargs,
    ):
        kwargs = gather_vars_to_kwargs()
        del kwargs['self']

        self._qkvo_iq_conf = kwargs.pop('qkv_iq_conf') or QuantizerConfig(place='input')
        self._qkv_kq_conf = kwargs.pop('qkv_kq_conf') or QuantizerConfig(place='weight')
        self._qkv_bq_conf = kwargs.pop('qkv_bq_conf') or QuantizerConfig(place='bias')
        self._softmax_iq_conf = kwargs.pop('softmax_iq_conf') or QuantizerConfig(place='input')
        self._softmax_exp_iq_conf = kwargs.pop('softmax_exp_iq_conf') or QuantizerConfig(place='input')
        self._softmax_exp_oq_conf = kwargs.pop('softmax_exp_oq_conf') or QuantizerConfig(place='table')
        self._softmax_inv_iq_conf = kwargs.pop('softmax_inv_iq_conf') or QuantizerConfig(place='input')
        self._softmax_inv_oq_conf = kwargs.pop('softmax_inv_oq_conf') or QuantizerConfig(place='table')
        self._stable_softmax = kwargs.pop('stable_softmax')

        super().__init__(**kwargs)

    def _get_common_kwargs_for_sublayer(self):
        common_kwargs: dict = super()._get_common_kwargs_for_sublayer()
        # Inject quantizer and ebops configs to sub QEinsumDense layers.
        common_kwargs.update({
            'iq_conf': self._qkvo_iq_conf,
            'kq_conf': self._qkv_kq_conf,
            'bq_conf': self._qkv_bq_conf,
            'enable_ebops': self.enable_ebops,
            'beta0': self._beta0.clone(),
        })
        return common_kwargs

    def build(
        self,
        query_shape,
        value_shape,
        key_shape=None,
    ):
        """Builds layers and variables.

        Args:
            query_shape: Shape of the `query` tensor.
            value_shape: Shape of the `value` tensor.
            key: Optional shape of the `key` tensor.
        """

        # Copied and modified from keras MultiHeadAttention, substituted EinsumDense with QEinsumDense and added sequence length (shape) to its output shape when initializing, if known.
        key_shape = value_shape if key_shape is None else key_shape

        if query_shape[-1] != value_shape[-1]:
            raise ValueError(
                "The last dimension of `query_shape` and `value_shape` "
                f"must be equal, but are {query_shape[-1]}, {value_shape[-1]}. "
                "Received: query_shape={query_shape}, value_shape={value_shape}"
            )

        if value_shape[1:-1] != key_shape[1:-1]:
            raise ValueError(
                "All dimensions of `value` and `key`, except the last one, "
                f"must be equal. Received: value_shape={value_shape} and "
                f"key_shape={key_shape}"
            )

        query_rank = len(query_shape)
        value_rank = len(value_shape)
        key_rank = len(key_shape)
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            query_rank - 1, bound_dims=1, output_dims=2
        )
        self._query_dense = QEinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._key_dim], query_shape
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name="query",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._query_dense.build(query_shape)
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            key_rank - 1, bound_dims=1, output_dims=2
        )
        self._key_dense = QEinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._key_dim], key_shape
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name="key",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._key_dense.build(key_shape)
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            value_rank - 1, bound_dims=1, output_dims=2
        )
        self._value_dense = QEinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._value_dim], value_shape
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name="value",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._value_dense.build(value_shape)

        # Builds the attention computations for multi-head dot product
        # attention.  These computations could be wrapped into the keras
        # attention layer once it supports multi-head einsum computations.
        self._build_attention(output_rank)
        self._output_dense = self._make_output_dense(
            query_shape,
            self._get_common_kwargs_for_sublayer(),
            "attention_output",
        )
        output_dense_input_shape = list(
            self._query_dense.compute_output_shape(query_shape)
        )
        output_dense_input_shape[-1] = self._value_dim
        self._output_dense.build(tuple(output_dense_input_shape))
        self.built = True

    def _make_output_dense(self, query_shape, common_kwargs, name=None):
        """Builds the output projection matrix.

        Args:
            free_dims: Number of free dimensions for einsum equation building.
            common_kwargs: Common keyword arguments for einsum layer.
            name: Name for the projection layer.

        Returns:
            Projection layer.
        """

        # Copied and modified from keras MultiHeadAttention, substituted EinsumDense with QEinsumDense and added sequence length (shape) to its output shape when initializing, if known.
        query_rank = len(query_shape)
        if self._output_shape:
            if not isinstance(self._output_shape, Sized):
                output_shape = [self._output_shape]
            else:
                output_shape = self._output_shape
        else:
            output_shape = [query_shape[-1]]
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            query_rank - 1, bound_dims=2, output_dims=len(output_shape)
        )
        return QEinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(output_rank - 1, output_shape, query_shape),
            bias_axes=bias_axes if self._use_bias else None,
            name=name,
            **common_kwargs,
        )

    def _build_attention(self, rank):
        """Builds multi-head dot-product attention computations.

        This function builds attributes necessary for `_compute_attention` to
        customize attention computation to replace the default dot-product
        attention.

        Args:
            rank: the rank of query, key, value tensors.
        """

        # Copied and modified from keras MultiHeadAttention, substituted Softmax with QSoftmax.
        if self._attention_axes is None:
            self._attention_axes = tuple(range(1, rank - 2))
        else:
            self._attention_axes = tuple(self._attention_axes)
        (
            self._dot_product_equation,
            self._combine_equation,
            attn_scores_rank,
        ) = _build_attention_equation(rank, attn_axes=self._attention_axes)
        norm_axes = tuple(
            range(
                attn_scores_rank - len(self._attention_axes), attn_scores_rank
            )
        )
        self._softmax = QSoftmax(
            axis=norm_axes,
            dtype=self.dtype_policy,
            stable=self._stable_softmax,
            iq_conf=self._softmax_iq_conf,
            exp_iq_conf=self._softmax_exp_iq_conf,
            exp_oq_conf=self._softmax_exp_oq_conf,
            inv_iq_conf=self._softmax_inv_iq_conf,
            inv_oq_conf=self._softmax_inv_oq_conf,
        )
        self._dropout_layer = Dropout(
            rate=self._dropout, dtype=self.dtype_policy, seed=self.seed
        )
        self._inverse_sqrt_key_dim = 1.0 / math.sqrt(float(self._key_dim))

    def get_config(self):
        config = super().get_config()
        config.update({
            'qkv_iq_conf': self._qkvo_iq_conf,
            'qkv_kq_conf': self._qkv_kq_conf,
            'qkv_bq_conf': self._qkv_bq_conf,
            'softmax_iq_conf': self._softmax_iq_conf,
            'softmax_exp_iq_conf': self._softmax_exp_iq_conf,
            'softmax_exp_oq_conf': self._softmax_exp_oq_conf,
            'softmax_inv_iq_conf': self._softmax_inv_iq_conf,
            'softmax_inv_oq_conf': self._softmax_inv_oq_conf,
            'stable_softmax': self._stable_softmax,
        })
        return config
