from keras.callbacks import Callback
from tqdm import tqdm


class PBar(Callback):
    def __init__(self, metric='loss: {loss:.2f}/{val_loss:.2f}', disable_ebops=False):
        self.pbar = None
        self.template = metric
        self.disable_ebops = disable_ebops

    def on_epoch_begin(self, epoch, logs=None):
        assert isinstance(self.params, dict)
        if self.pbar is None:
            self.pbar = tqdm(total=self.params['epochs'], unit='epoch')

    def on_epoch_end(self, epoch, logs=None):
        assert isinstance(self.pbar, tqdm)
        assert isinstance(logs, dict)
        self.pbar.update(1)
        string = self.template.format(**logs)
        if not self.disable_ebops and 'ebops' in logs:
            string += f' - EBOPs: {logs["ebops"]:,.0f}'
        self.pbar.set_description(string)

    def on_train_end(self, logs=None):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
