from keras.api import ops
from keras.api.saving import deserialize_keras_object, register_keras_serializable, serialize_keras_object
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.rnn.simple_rnn import RNN, SimpleRNN, SimpleRNNCell

from ...config import QuantizerConfig
from ...quantizer import Quantizer
from ..core.base import QLayerBaseSingleInput


class QSimpleRNNCell(QLayerBaseSingleInput, SimpleRNNCell):
    __no_wrap_call__ = True

    @property
    def kq(self):
        "Kernel Quantizer"
        return self._kq

    @property
    def rkq(self):
        "Recurrent Kernel Quantizer"
        return self._rkq

    @property
    def iq(self):
        "Input Quantizer"
        return self._iq

    @property
    def sq(self):
        "State Quantizer"
        return self._sq

    @property
    def bq(self):
        "Bias Quantizer"
        return self._bq

    @property
    def qkernel(self):
        return self.kq(self.kernel)

    @property
    def qrecurrent_kernel(self):
        return self.rkq(self.recurrent_kernel)

    @property
    def qbias(self):
        return self.bq(self.bias)

    def __init__(
        self,
        units,
        activation='tanh',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros',
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        iq_conf: QuantizerConfig | None = None,
        sq_conf: QuantizerConfig | None = None,
        kq_conf: QuantizerConfig | None = None,
        rkq_conf: QuantizerConfig | None = None,
        bq_conf: QuantizerConfig | None = None,
        **kwargs,
    ):
        iq_conf = iq_conf or QuantizerConfig(place='datalane')
        sq_conf = sq_conf or QuantizerConfig(place='datalane')
        kq_conf = kq_conf or QuantizerConfig(place='weight')
        rkq_conf = rkq_conf or QuantizerConfig(place='weight')
        bq_conf = bq_conf or QuantizerConfig(place='bias')
        # self._iq = Quantizer(iq_conf, name='iq')
        self._sq = Quantizer(sq_conf, name='sq')
        self._kq = Quantizer(kq_conf, name='kq')
        self._rkq = Quantizer(rkq_conf, name='rkq')
        self._bq = Quantizer(bq_conf, name='bq')

        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            iq_conf=iq_conf,
            seed=seed,
            **kwargs,
        )

    def build(self, input_shape):
        self._sq.build((None, self.units))
        self._kq.build((input_shape[-1], self.units))
        self._rkq.build((self.units, self.units))
        self._bq.build((self.units,))
        super().build(input_shape)

    def call(self, sequence, states, training=False):
        prev_output = states[0] if isinstance(states, (list, tuple)) else states
        dp_mask = self.get_dropout_mask(sequence)
        rec_dp_mask = self.get_recurrent_dropout_mask(prev_output)

        if training and dp_mask is not None:
            sequence = sequence * dp_mask
        if training and rec_dp_mask is not None:
            prev_output = prev_output * rec_dp_mask

        qkernel = self.kq(self.kernel, training=training)
        qrecurrent_kernel = self.rkq(self.recurrent_kernel, training=training)
        qstate = self.sq(prev_output, training=training)
        qsequence = self.iq(sequence, training=training)

        h = ops.matmul(qsequence, qkernel)
        if self.bias is not None:
            h += self.bq(self.bias, training=training)

        output = h + ops.matmul(qstate, qrecurrent_kernel)  # type: ignore
        if self.activation is not None:
            output = self.activation(output)

        new_state = [output] if isinstance(states, (list, tuple)) else output
        return output, new_state

    def get_config(self):
        conf = super().get_config()
        conf.update(
            {
                'iq_conf': self._iq.config,
                'sq_conf': self._sq.config,
                'kq_conf': self._kq.config,
                'rkq_conf': self._rkq.config,
                'bq_conf': self._bq.config,
            }
        )
        return conf

    def _compute_ebops(self, shape):
        bw_inp = self.iq.bits_(shape)
        bw_ker = self.kq.bits_(ops.shape(self.kernel))
        ebops1 = ops.sum(ops.matmul(bw_inp, bw_ker))
        bw_state = self.sq.bits_((1, self.units))
        bw_rker = self.rkq.bits_(ops.shape(self.recurrent_kernel))
        ebops2 = ops.sum(ops.matmul(bw_state, bw_rker))
        ebops = ebops1 + ebops2  # type: ignore
        if self.bq is not None:
            bw_bias = self.bq.bits_(ops.shape(self.bias))
            size = ops.cast(ops.prod(shape), self.dtype)
            ebops = ebops + ops.mean(bw_bias) * size  # type: ignore

        return ebops


@register_keras_serializable(package='squark')
class QSimpleRNN(SimpleRNN):
    """Fully-connected RNN where the output is to be fed back as the new input.

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer uses
            a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `"glorot_uniform"`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation of the recurrent
            state.  Default: `"orthogonal"`.
        bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector.
            Default: `None`.
        activity_regularizer: Regularizer function applied to the output of the
            layer (its "activation"). Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix.  Default: `None`.
        bias_constraint: Constraint function applied to the bias vector.
            Default: `None`.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for the linear transformation
            of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for the linear transformation of the
            recurrent state. Default: 0.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence. Default: `False`.
        return_state: Boolean. Whether to return the last state
            in addition to the output. Default: `False`.
        go_backwards: Boolean (default: `False`).
            If `True`, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default: `False`). If `True`, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default: `False`).
            If `True`, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
        iq_conf: QuantizerConfig, optional
            Input Quantizer configuration
        sq_conf: QuantizerConfig, optional
            State Quantizer configuration
        kq_conf: QuantizerConfig, optional
            Kernel Quantizer configuration
        rkq_conf: QuantizerConfig, optional
            Recurrent Kernel Quantizer configuration
        bq_conf: QuantizerConfig, optional
            Bias Quantizer configuration

    Call arguments:
        sequence: A 3D tensor, with shape `[batch, timesteps, feature]`.
        mask: Binary tensor of shape `[batch, timesteps]` indicating whether
            a given timestep should be masked. An individual `True` entry
            indicates that the corresponding timestep should be utilized,
            while a `False` entry indicates that the corresponding timestep
            should be ignored.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode.
            This argument is passed to the cell when calling it.
            This is only relevant if `dropout` or `recurrent_dropout` is used.
        initial_state: List of initial state tensors to be passed to the first
            call of the cell.

    Example:

    ```python
    inputs = np.random.random((32, 10, 8))
    simple_rnn = keras.layers.SimpleRNN(4)
    output = simple_rnn(inputs)  # The output has shape `(32, 4)`.
    simple_rnn = keras.layers.SimpleRNN(4, return_sequences=True, return_state=True)
    # whole_sequence_output has shape `(32, 10, 4)`.
    # final_state has shape `(32, 4)`.
    whole_sequence_output, final_state = simple_rnn(inputs)
    ```
    """

    def __init__(
        self,
        units,
        activation='tanh',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros',
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        seed=None,
        iq_conf: QuantizerConfig | None = None,
        sq_conf: QuantizerConfig | None = None,
        kq_conf: QuantizerConfig | None = None,
        rkq_conf: QuantizerConfig | None = None,
        bq_conf: QuantizerConfig | None = None,
        parallelization_factor=-1,
        **kwargs,
    ):
        cell = QSimpleRNNCell(
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            seed=seed,
            dtype=kwargs.get('dtype', None),
            trainable=kwargs.get('trainable', True),
            name='simple_rnn_cell',
            iq_conf=iq_conf,
            sq_conf=sq_conf,
            kq_conf=kq_conf,
            rkq_conf=rkq_conf,
            bq_conf=bq_conf,
        )
        RNN.__init__(
            self,
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs,
        )
        self.input_spec = [InputSpec(ndim=3)]
        self.parallelization_factor = parallelization_factor

    def get_config(self):  # type: ignore
        conf = super().get_config()
        conf.update(
            {
                'iq_conf': self.cell._iq.config,
                'sq_conf': self.cell._sq.config,
                'kq_conf': self.cell._kq.config,
                'rkq_conf': self.cell._rkq.config,
                'bq_conf': self.cell._bq.config,
                'parallelization_factor': self.parallelization_factor,
            }
        )
        return serialize_keras_object(conf)

    @classmethod
    def from_config(cls, config):
        config = deserialize_keras_object(config)
        pass
        return super().from_config(config)

    def build(self, sequences_shape, initial_state_shape=None):
        seq_len = sequences_shape[1]
        if self.parallelization_factor == -1:
            self.parallelization_factor = seq_len
        return super().build(sequences_shape, initial_state_shape)

    def call(self, sequences, initial_state=None, mask=None, training=False):
        ebops = self.cell._compute_ebops((1,) + ops.shape(sequences[0])[1:]) * self.parallelization_factor
        self.cell._ebops.assign(ops.cast(ebops, self.cell._ebops.dtype))  # type: ignore
        self.add_loss(ebops * self.cell.beta)
        return super().call(sequences, mask=mask, training=training, initial_state=initial_state)
