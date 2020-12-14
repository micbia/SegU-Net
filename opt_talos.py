import numpy as np, sys, talos
import tensorflow as tf

from keras.layers import ReLU, LeakyReLU, ELU
from keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import load_model

from config.net_config import NetworkConfig
from tests.networks_test import Unet
from sklearn.model_selection import train_test_split
from utils.other_utils import get_data, get_batch
from config.net_config import NetworkConfig
from utils_network.metrics import iou, iou_loss, dice_coef, dice_coef_loss, phi_coef, balanced_cross_entropy

avail_metrics = {'binary_accuracy':'binary_accuracy', 'iou':iou, 'dice_coef':dice_coef, 'iou_loss':iou_loss, 'dice_coef_loss':dice_coef_loss, 'phi_coef':phi_coef, 'mse':'mse', 'mae':'mae', 'binary_crossentropy':'binary_crossentropy', 'balanced_cross_entropy':balanced_cross_entropy}
load_trained_model = False

RANDOM_SEED = 2020
#PATH_TRAIN = '/home/michele/Documents/PhD_Sussex/output/ML/dataset/inputs/data2D_128_030920/'
PATH_TRAIN = '/home/michele/Documents/PhD_Sussex/output/ML/dataset/inputs/data2D_128_030920.zip'
PATH_OUT = 'tests/'

# Load Data
IM_SHAPE = (128, 128)
if(isinstance(PATH_TRAIN, str)):
    print('Load data ...')
    #X, y = get_data(PATH_TRAIN+'data/', IM_SHAPE, shuffle=True)
    X, y = get_batch(path=PATH_TRAIN, img_shape=IM_SHAPE, size=1200, dataset_size=30000)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
else:
    print('Load images ...') 
    X_train, y_train = get_data(PATH_TRAIN[0]+'data/', IM_SHAPE, shuffle=True)
    print('Load masks ...') 
    X_valid, y_valid = get_data(PATH_TRAIN[1]+'data/', IM_SHAPE, shuffle=True)

size_train_dataset = X_train.shape[0]
size_valid_dataset = X_valid.shape[0]

# Network Hyperparameters
p = {'coarse_dim': [256, 512],
     'dropout':[0.05, 0.15, 0.3],
     'kernel_size':[3, 4, 6],
     'batch_size':[16, 32, 64],
     'activation': [ReLU(), LeakyReLU(), ELU()],
     'lr':[10**(-i) for i in range(3,8)],
     #'optimizer': [RMSprop(), Adam()],
     'optimizer': [Adam()],
     'epochs':[100],
     'loss':[balanced_cross_entropy, 'binary_crossentropy']
    }

# Network to optimize
def TestModel(x_train, y_train, x_val, y_val, par):
    opt_model = Unet(img_shape=np.append(IM_SHAPE, 1), params=par, path=PATH_OUT)
    
    if(load_trained_model):
        # load trained model
        conf = NetworkConfig(PATH_OUT+'runs/net2D_021020.ini')
        avail_metrics = {'binary_accuracy':'binary_accuracy', 'iou':iou, 'dice_coef':dice_coef, 'iou_loss':iou_loss, 'dice_coef_loss':dice_coef_loss, 'phi_coef':phi_coef, 'mse':'mse', 'mae':'mae', 'binary_crossentropy':'binary_crossentropy', 'balanced_cross_entropy':balanced_cross_entropy} 
        PATH_MODEL = conf.resume_path+'checkpoints/'
        MODEL_EPOCH = conf.best_epoch
        METRICS = [avail_metrics[m] for m in np.append(conf.loss, conf.metrics)]
        cb = {func.__name__:func for func in METRICS if not isinstance(func, str)}
        trained_model = load_model('%smodel-sem21cm_ep%d.h5' %(PATH_MODEL, MODEL_EPOCH), custom_objects=cb)

        # copy weight in optimization model
        opt_model.set_weights(trained_model.get_weights()) 

    opt_model.compile(optimizer=par['optimizer'], loss=par['loss'])
    results = opt_model.fit(x=x_train, y=y_train, batch_size=par['batch_size'], epochs=par['epochs'], validation_data=(X_valid, y_valid), shuffle=True)
    
    return results, opt_model

scan_object = talos.Scan(X_train, y_train, params=p, model=TestModel, experiment_name=PATH_OUT+'opt_unet', fraction_limit=.001)
