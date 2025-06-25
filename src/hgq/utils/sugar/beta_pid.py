from typing import Optional
from keras.callbacks import Callback
from keras.models import Model

class BetaPID(Callback):
    """
    Control the beta value of the Q Layers using a PID controller.

    Parameters
    ----------
    p : float
        Proportional gain.
    i : float
        Integral gain.
    d : float
        Derivative gain.
    target_ebops : float
        Target EBOPs to reach.
    warmup : int, optional
        Number of epochs to wait before starting to adjust beta. Default is 10.
    init_beta : float, optional
        Initial beta value. Default is 1e-10.
    max_beta : float, optional
        Maximum beta value to allow. Default is 1e-6.
    damp_beta_on_target : float, optional
        If set and the model EBOPs fall below the target EBOPs, the current beta
        will be multiplied by this value. For example, 0.1 will reduce beta by 90%.
        Useful for fine-tuning once the target is reached. If None, no damping is applied.

    Notes
    -----
    The beta value is adjusted based on the PID controller output. This helps
    guide the training process by dynamically controlling the regularization strength.
    """

    def __init__(
        self,
        p: float,
        i: float,
        d: float,
        target_ebops: float,
        warmup: int = 10,
        init_beta: float = 1e-10,
        max_beta: float = 1e-6,
        damp_beta_on_target: Optional[float] = None,
    ) -> None:
        self._beta: float = init_beta
        self.warmup: int = warmup
        self.p: float = p
        self.i: float = i
        self.d: float = d
        self.target_ebops: float = target_ebops
        self.max_beta: float = max_beta

        self.init_ebops: Optional[float] = None
        self.damp_beta_on_target: Optional[float] = damp_beta_on_target

        self.integral: float = 0.0
        self.prev_error: float = 0.0

    def get_model_ebops(self) -> float:
        ebops: float = 0.0
        for layer in self.model.layers:
            if hasattr(layer, "ebops"):
                ebops += layer.ebops
        return ebops

    def update_beta(self) -> float:
        current_value: float = self.get_model_ebops()

        error: float = 1 - self.current_value / self.target_ebops
        self.integral += error
        derivative: float = error - self.prev_error

        pid_out: float = self.p * error + self.i * self.integral + self.d * derivative
        self.prev_error = error

        return -pid_out

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        if epoch == 1:
            if self.init_ebops is None:
                self.init_ebops = float(self.get_model_ebops())

        if epoch > self.warmup:
            new_beta: float = float(self.update_beta())

            self._beta = max(0.0, new_beta)
            self._beta = min(self._beta, self.max_beta)

            if (
                self.damp_beta_on_target is not None
                and self.get_model_ebops() < self.target_ebops
            ):
                self._beta *= self.damp_beta_on_target

            assert isinstance(self.model, Model)

            for layer in self.model._flatten_layers():
                if hasattr(layer, '_beta'):
                    layer._beta.assign(ops.convert_to_tensor(self._beta, dtype=layer._beta.dtype))

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        assert isinstance(logs, dict)
        logs['beta'] = self._beta