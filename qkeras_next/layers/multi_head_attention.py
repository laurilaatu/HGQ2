from abc import ABCMeta
from collections.abc import Iterable, Sequence
from inspect import signature

from keras import ops
from keras.api.initializers import Constant, Initializer
from keras.api.layers import Concatenate, Layer, MultiHeadAttention
from keras.api.saving import deserialize_keras_object, serialize_keras_object
from keras.src import backend

from ..quantizer import Quantizer, QuantizerConfig, numbers
from ..utils.config.layer import global_config
from .core.base import QLayerBase

three_q_conf = tuple[QuantizerConfig, QuantizerConfig, QuantizerConfig]
four_q_conf = tuple[QuantizerConfig, QuantizerConfig, QuantizerConfig, QuantizerConfig]


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
        to_q_confs: None | three_q_conf = None,
        to_k_confs: None | three_q_conf = None,
        to_v_confs: None | three_q_conf = None,
        softmax_confs: None | three_q_conf | four_q_conf = None,
        **kwargs,
    ):
        bound = signature(MultiHeadAttention.__init__).bind()
        bound.apply_defaults()
        bound.arguments.update(kwargs)
        bound.arguments.pop("stable_softmax")
        bound.arguments.pop("to_q_confs")
        bound.arguments.pop("to_k_confs")
        bound.arguments.pop("to_v_confs")
        super().__init__(**bound.arguments)

        if stable_softmax:
            assert softmax_confs is None or len(softmax_confs) == 4, (
                "If stable_softmax is True, softmax_confs must be a tuple of length 4 if provided."
            )

        default_to_qkv_q_conf = (
            QuantizerConfig("default", "input"),
            QuantizerConfig("default", "kernel"),
            QuantizerConfig("default", "bias")
        )

        iq_q_confs = to_q_confs or default_to_qkv_q_conf
        iq_k_confs = to_k_confs or default_to_qkv_q_conf
        iq_v_confs = to_v_confs or default_to_qkv_q_conf

        default_softmax_conf = (
            QuantizerConfig("default", "input"),
            QuantizerConfig("default", "input"),
            QuantizerConfig("default", "input"),
            QuantizerConfig("default", "input"),
        )

        softmax_confs = softmax_confs or default_softmax_conf
