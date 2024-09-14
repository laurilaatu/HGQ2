# %%
import argparse
import os

import keras
import numpy as np
from HGQ.layers import HDenseBatchNorm, HQuantize
from HGQ.layers.batchnorm_base import HBatchNormalization
from HGQ.layers.passive_layers import PLayerBase, PReshape
from keras.layers import Add, BatchNormalization, Dense, Dropout, GlobalAveragePooling1D, GlobalMaxPooling1D, LayerNormalization, MultiHeadAttention

parser = argparse.ArgumentParser()
parser.add_argument('--constituent', '-c', type=int, default=50)
parser.add_argument('--epoch', '-e', type=int, default=60)
parser.add_argument('--batch', '-b', type=int, default=512)
args = parser.parse_args()


from HGQ import get_default_kq_conf, get_default_paq_conf, set_default_kq_conf, set_default_paq_conf

kq_conf = get_default_kq_conf()
paq_conf = get_default_paq_conf()
kq_conf['init_bw'] = 6
paq_conf['init_bw'] = 8
set_default_kq_conf(kq_conf)
set_default_paq_conf(paq_conf)


class PPermute(PLayerBase, keras.layers.Permute):
    pass


# %%

import h5py as h5

with h5.File('/data/massive2/fastml_data/jet150/150c-train.h5') as f:
    X_train = np.array(f['feature'])
    y_train = np.array(f['label'])
with h5.File('/data/massive2/fastml_data/jet150/150c-test.h5') as f:
    X_test = np.array(f['feature'])
    y_test = np.array(f['label'])
labels = 'gqWZt'

# %%

X_train = np.array(X_train[:, :args.constituent], dtype=np.float16)
X_test = np.array(X_test[:, :args.constituent], dtype=np.float16)


# %%
inp = keras.layers.Input((args.constituent, 16))
inp_q_mask = keras.backend.any(inp, axis=-1)

dim = 8
h = 4

beta = 0.

x = HQuantize(beta=beta)(inp)
# x = HBatchNormalization()(x)
# inp_q_mask = MaskDropout(0.1)(inp_q_mask)

x = HDenseBatchNorm(16, activation='relu', beta=beta)(x)
x = HDenseBatchNorm(16, activation='relu', beta=beta)(x)
x = PPermute((2, 1))(x)

x = HDenseBatchNorm(32, activation='relu', beta=beta)(x)
x = PPermute((2, 1))(x)

x = HDenseBatchNorm(16, activation='relu', beta=beta)(x)
x = HDenseBatchNorm(16, activation='relu', beta=beta)(x)
x = PPermute((2, 1))(x)

x = HDenseBatchNorm(1, activation='relu', beta=beta)(x)
x = PPermute((2, 1))(x)

x = HDenseBatchNorm(5, activation='relu', beta=beta)(x)
out = PReshape((5,))(x)
# x = keras.layers.Permute((2, 1))(x)
# x = Dense(1)(x)
# x = BatchNormalization()(x)
# x = keras.layers.Activation('relu')(x)
# x = keras.layers.Permute((2, 1))(x)


model = keras.models.Model(inp, out)


# %%
# x.shape

# %%


opt = keras.optimizers.AdamW()
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# lr_schedule = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
cosine = keras.optimizers.schedules.CosineDecayRestarts(3e-3, 20, 1, 0.95)
lr_schedule = keras.callbacks.LearningRateScheduler(lambda *args: float(cosine(*args)))
reset = ResetMinMax()
bops = FreeBOPs()

# %%
# use bf16 for training
model.compile(opt, loss, metrics=['accuracy'])

# model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test))
# model.compile(opt, loss, metrics=['accuracy'])

# %%
model.summary()

# %%


# %%

callbacks = [lr_schedule, bops, reset]
model.fit(X_train, y_train, validation_split=0.1, epochs=600, batch_size=1024, callbacks=callbacks)

# %%
model.evaluate(X_test, y_test, batch_size=1024)

pred = model.predict(X_test, batch_size=1024, verbose=0)  # type: ignore
pred = np.array(keras.backend.softmax(pred))
# %%

from sklearn.metrics import roc_curve

disp_fpr = (0.1, 0.01)
fpr = {}
tpr = {}
for i in range(5):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, pred[:, i])
    auc = np.trapz(tpr[i], fpr[i])
    print(f'{labels[i]}: auc = {auc:.3f}')
    for _fpr in disp_fpr:
        idx = np.argmax(fpr[i] > _fpr)
        print(f'  tpr@fpr={_fpr}: {tpr[i][idx]:.3f}')


# %%


# %%


# %%
