import inspect
from collections.abc import Sequence

from squark import layers
from squark.config import QuantizerConfig, QuantizerConfigScope

from ..initializers import QInitializer  # TODO: use qkeras initializers when called from here
from ..quantizers import get_quantizer

kw_map = {
    'kq_conf': ('kernel_quantizer', 'kq'),
    'bq_conf': ('bias_quantizer', 'bq'),
    'oq_conf': ('output_quantizer', 'oq'),
    'iq_conf': ('input_quantizer', 'iq'),
    'exp_iq_conf': ('exp_input_quantizer', 'exp_iq'),
    'exp_oq_conf': ('exp_output_quantizer', 'exp_oq'),
    'inv_iq_conf': ('inv_input_quantizer', 'inv_iq'),
    'inv_oq_conf': ('inv_output_quantizer', 'inv_oq'),
}

kw_map_inv = {vv: k for k, v in kw_map.items() for vv in v}


def qkeras_layer_wrap(cls: type):
    # base_cls = cls.__bases__[0]
    original_init = cls.__init__
    signature = inspect.signature(original_init)
    params = signature.parameters
    new_params = []
    for v in params.values():
        if v.name in kw_map:
            new_params.append(v.replace(name=kw_map[v.name][0]))
        else:
            new_params.append(v)
    new_signature = signature.replace(parameters=new_params)

    original_init = cls.__init__

    def __init__(self, *args, **kwargs):
        for k, v in list(kwargs.items()):
            if k not in kw_map_inv:
                continue
            new_k = kw_map_inv[k]
            assert new_k not in kwargs, f"Duplicate key {new_k}."
            del kwargs[k]
            v = v if not isinstance(v, str) else get_quantizer(v)
            kwargs[new_k] = v

        # Disable EBOPS; only explicitly enabled quantizers will be used
        kwargs['enable_ebops'] = False
        kwargs['enable_iq'] = kwargs.get('iq_conf') is not None
        kwargs['enable_oq'] = kwargs.get('oq_conf') is not None
        with QuantizerConfigScope(default_q_type='dummy'):
            return original_init(self, *args, **kwargs)

    __init__.__signature__ = new_signature  # type: ignore
    cls.__init__ = __init__
    return cls


for name, obj in layers.__dict__.items():
    if not isinstance(obj, type):
        continue
    if issubclass(obj, layers.QLayerBase):
        globals()[name] = qkeras_layer_wrap(obj)


# @qkeras_layer_wrap
# class QDense(layers.QDense):
#     def __init__(
#         self,
#         units,
#         activation=None,
#         use_bias=True,
#         kernel_initializer="glorot_uniform",
#         bias_initializer="zeros",
#         kernel_regularizer=None,
#         bias_regularizer=None,
#         activity_regularizer=None,
#         kernel_constraint=None,
#         bias_constraint=None,
#         kernel_quantizer: None | QuantizerConfig | str = None,
#         input_quantizer: None | QuantizerConfig | str = None,
#         bias_quantizer: None | QuantizerConfig | str = None,
#         output_quantizer: None | QuantizerConfig | str = None,
#         **kwargs,
#     ):
#         ...


# @qkeras_layer_wrap
# class QEinsumDense(layers.QEinsumDense):
#     def __init__(
#         self,
#         equation,
#         output_shape,
#         activation=None,
#         bias_axes=None,
#         kernel_initializer="glorot_uniform",
#         bias_initializer="zeros",
#         kernel_regularizer=None,
#         bias_regularizer=None,
#         kernel_constraint=None,
#         bias_constraint=None,
#         kernel_quantizer: None | QuantizerConfig | str = None,
#         input_quantizer: None | QuantizerConfig | str = None,
#         bias_quantizer: None | QuantizerConfig | str = None,
#         output_quantizer: None | QuantizerConfig | str = None,
#         **kwargs,
#     ):
#         ...


# @qkeras_layer_wrap
# class QConv1D(layers.QConv1D):
#     def __init__(
#         self,
#         filters,
#         kernel_size,
#         strides=1,
#         padding="valid",
#         data_format=None,
#         dilation_rate=1,
#         groups=1,
#         activation=None,
#         use_bias=True,
#         kernel_initializer="glorot_uniform",
#         bias_initializer="zeros",
#         kernel_regularizer=None,
#         bias_regularizer=None,
#         activity_regularizer=None,
#         kernel_constraint=None,
#         bias_constraint=None,
#         kernel_quantizer: None | QuantizerConfig | str = None,
#         input_quantizer: None | QuantizerConfig | str = None,
#         bias_quantizer: None | QuantizerConfig | str = None,
#         output_quantizer: None | QuantizerConfig | str = None,
#         **kwargs,
#     ):
#         ...


# @qkeras_layer_wrap
# class QConv2D(layers.QConv2D):
#     def __init__(
#         self,
#         filters,
#         kernel_size,
#         strides=(1, 1),
#         padding="valid",
#         data_format=None,
#         dilation_rate=(1, 1),
#         groups=1,
#         activation=None,
#         use_bias=True,
#         kernel_initializer="glorot_uniform",
#         bias_initializer="zeros",
#         kernel_regularizer=None,
#         bias_regularizer=None,
#         activity_regularizer=None,
#         kernel_constraint=None,
#         bias_constraint=None,
#         kernel_quantizer: None | QuantizerConfig | str = None,
#         input_quantizer: None | QuantizerConfig | str = None,
#         bias_quantizer: None | QuantizerConfig | str = None,
#         output_quantizer: None | QuantizerConfig | str = None,
#         **kwargs,
#     ):
#         ...


# @qkeras_layer_wrap
# class QConv3D(layers.QConv3D):
#     def __init__(
#         self,
#         filters,
#         kernel_size,
#         strides=(1, 1, 1),
#         padding="valid",
#         data_format=None,
#         dilation_rate=(1, 1, 1),
#         groups=1,
#         activation=None,
#         use_bias=True,
#         kernel_initializer="glorot_uniform",
#         bias_initializer="zeros",
#         kernel_regularizer=None,
#         bias_regularizer=None,
#         activity_regularizer=None,
#         kernel_constraint=None,
#         bias_constraint=None,
#         kernel_quantizer: None | QuantizerConfig | str = None,
#         input_quantizer: None | QuantizerConfig | str = None,
#         bias_quantizer: None | QuantizerConfig | str = None,
#         output_quantizer: None | QuantizerConfig | str = None,
#         **kwargs,
#     ):
#         ...


# @qkeras_layer_wrap
# class QSoftmax(layers.QSoftmax):
#     def __init__(
#         self,
#         axis: int | Sequence[int] = -1,
#         iq_conf: None | QuantizerConfig = None,
#         stable=False,
#         exp_input_quantizer: None | QuantizerConfig | str = None,
#         exp_output_quantizer: None | QuantizerConfig | str = None,
#         inv_input_quantizer: None | QuantizerConfig | str = None,
#         inv_output_quantizer: None | QuantizerConfig | str = None,
#         allow_heterogeneous_table: bool = False,
#         input_scaler: float = 1.0,
#         **kwargs
#     ):
#         ...


# @qkeras_layer_wrap
# class QActivation(layers.QUnaryFunctionLUT):
#     def __init__(
#         self,
#         function,
#         input_quantizer: None | QuantizerConfig | str = None,
#         output_quantizer: None | QuantizerConfig | str = None,
#         allow_heterogeneous_table: bool = False,
#         **kwargs
#     ):
#         ...


# @qkeras_layer_wrap
# class QMeanPow2(layers.QMeanPow2):
#     pass


# @qkeras_layer_wrap
# class QSum(layers.QSum):
#     pass


# @qkeras_layer_wrap
# class QAdd(layers.QAdd):
#     pass


# @qkeras_layer_wrap
# class QAveragePow2(layers.QAveragePow2):
#     pass


# @qkeras_layer_wrap
# class QDot(layers.QDot):
#     pass


# @qkeras_layer_wrap
# class QEinsum(layers.QEinsum):
#     pass


# @qkeras_layer_wrap
# class QMatmul(layers.QMatmul):
#     pass


# @qkeras_layer_wrap
# class QMaximum(layers.QMaximum):
#     pass


# @qkeras_layer_wrap
# class QMinimum(layers.QMinimum):
#     pass


# @qkeras_layer_wrap
# class QMultiply(layers.QMultiply):
#     pass


# @qkeras_layer_wrap
# class QSubtract(layers.QSubtract):
#     pass
