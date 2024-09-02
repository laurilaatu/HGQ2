from math import ceil

import keras
from keras import ops
from keras.api.utils import PyDataset


class Dataset(PyDataset):
    def __init__(self, x_set, y_set, batch_size, device='cpu:0', drop_last=False, **kwargs):
        super().__init__(**kwargs)
        with keras.device(device):
            self.x = ops.convert_to_tensor(x_set)
            self.y = ops.convert_to_tensor(y_set)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __len__(self):
        if self.drop_last:
            return len(self.x) // self.batch_size
        return ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.x))
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]
        return batch_x, batch_y
