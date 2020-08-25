import numpy as np, sys, talos
import tensorflow as tf

from keras.layers import ReLU, LeakyReLU, ELU
from keras.optimizers import RMSprop, Adam

from tests.networks_test import Unet
from utils.other_utils import get_data
from config.net_config import NetworkConfig
from utils_network.metrics import iou, iou_loss, dice_coef, dice_coef_loss, phi_coef, balanced_cross_entropy

PATH_TRAIN = ['/home/michele/Documents/PhD_Sussex/output/ML/outputs/outputs_Segnet/inputs/data2D_128_180320_train/', 
              '/home/michele/Documents/PhD_Sussex/output/ML/outputs/outputs_Segnet/inputs/data2D_128_180320_train/']
PATH_OUT = 'tests/'

IM_SHAPE = (128, 128)
#IM_SHAPE = (128, 128, 128)

# Load Data
print('Load images ...') 
X_train, y_train = get_data(PATH_TRAIN[0]+'data/', IM_SHAPE, shuffle=True)
size_train_dataset = X_train.shape[0]
#print('Load masks ...') 
#X_valid, y_valid = get_data(PATH_TRAIN[1]+'data/', IM_SHAPE, shuffle=True)
#size_valid_dataset = X_valid.shape[0]


# Network Hyperparameters
p = {'coarse_dim': [256, 512],
     'dropout':[0.05, 0.15, 0.3],
     'kernel_size':[3, 4, 6],
     'batch_size':[16, 32, 64],
     'activation': [ReLU(), LeakyReLU(), ELU()],
     'lr':[10**(-i) for i in range(3,8)],
     'optimizer': [RMSprop(), Adam()],
     'epochs':[100],
     'loss':[balanced_cross_entropy, 'binary_crossentropy']
    }

# Network to optimize
def TestModel(x_train, y_train, x_val, y_val, par):
    model = Unet(img_shape=np.append(IM_SHAPE, 1), params=par, path=PATH_OUT)
    model.compile(optimizer=par['optimizer'], loss=par['loss'])
    results = model.fit(x=x_train, y=y_train, batch_size=par['batch_size'], epochs=par['epochs'], validation_data=(x_val, y_val), shuffle=True)
    return results, model

scan_object = talos.Scan(X_train, y_train, params=p, model=TestModel,
                         experiment_name=PATH_OUT+'opt_unet', fraction_limit=.001)
