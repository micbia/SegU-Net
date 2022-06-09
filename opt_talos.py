import talos
from talos.utils import lr_normalizer

import numpy as np, sys, pickle
import tensorflow as tf

from tensorflow.keras.layers import ReLU, LeakyReLU, ELU
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import logcosh

from config.net_config import NetworkConfig
from tests.networks_test import Unet
from sklearn.model_selection import train_test_split
from utils.other_utils import get_data, get_batch, get_data_lc
from config.net_config import NetworkConfig
from utils_network.metrics import r2score, precision, recall, iou, iou_loss, dice_coef, dice_coef_loss, matthews_coef, balanced_cross_entropy

with open('utils_network/avail_metrics.pkl', 'rb') as data:
    avail_metrics = pickle.loads(data.read())

#===================================
config_file = sys.argv[1]
conf = NetworkConfig(config_file)

load_trained_model = False
GPU = conf.gpus
#===================================
RANDOM_SEED = 2021

IO_PATH = conf.io_path
DATASET_PATH = conf.dataset_path
if isinstance(DATASET_PATH, list):
    PATH_TRAIN = IO_PATH+'inputs/'+DATASET_PATH[0]
    PATH_VALID = IO_PATH+'inputs/'+DATASET_PATH[1]
else:
    PATH_TRAIN = IO_PATH+'inputs/'+DATASET_PATH
    PATH_VALID = PATH_TRAIN
PATH_OUT = 'tests/test2/'

# Load Data
IM_SHAPE = conf.img_shape
if(isinstance(DATASET_PATH, str)):
    print('Load data ...')
    #X, y = get_data(PATH_TRAIN+'data/', IM_SHAPE, shuffle=True)
    X, y = get_batch(path=PATH_TRAIN, img_shape=IM_SHAPE, size=1200, dataset_size=30000)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
else:
    print('Load images ...') 
    #X_train, y_train = get_data_lc(path=PATH_TRAIN, i=0, shuffle=True)
    for i in range(9):
        if(i == 0):
            X_train, y_train = get_data_lc(path=PATH_TRAIN, i=i, shuffle=True)
        else:
            X_tmp, y_tmp = get_data_lc(path=PATH_TRAIN, i=i, shuffle=True)
            X_train, y_train = np.vstack((X_train, X_tmp)), np.vstack((y_train, y_tmp))
    print('Load masks ...') 
    #X_valid, y_valid = get_data_lc(path=PATH_VALID, i=10, shuffle=True)
    for i in range(9, 11):
        if(i == 9):
            X_valid, y_valid = get_data_lc(path=PATH_VALID, i=i, shuffle=True)
        else:
            X_tmp, y_tmp = get_data_lc(path=PATH_VALID, i=i, shuffle=True)
            X_valid, y_valid = np.vstack((X_valid, X_tmp)), np.vstack((y_valid, y_tmp))

size_train_dataset = X_train.shape[0]
size_valid_dataset = X_valid.shape[0]
print(size_train_dataset, size_valid_dataset)

# Network Hyperparameters
p = {'coarse_dim': [128, 256, 512],
     'dropout':[0.05, 0.1, 0.15],
     'kernel_size':[3, 4, 6],
     'batch_size':[16*GPU, 32*GPU, 64*GPU],
     'activation': [ReLU(), LeakyReLU(), ELU()],
     'final_activation': ['sigmoid', 'softmax'],
     'lr':[10**(-i) for i in range(5,7)],
     'optimizer': [Adam, RMSprop, Nadam],
     'epochs':[200],
     #'loss':[avail_metrics['balanced_cross_entropy'], avail_metrics['binary_crossentropy']],
     'depth': [3,4]
    }

# save hyperparemeters sample
with open(PATH_OUT+'hyperparameters_space.json', 'w') as file:
    par = p.copy()
    for var in par:
        if(var == 'activation' or var == 'optimizer'):
            par[var] = str(p[var])
    file.write(json.dumps(par))


# Network to optimize
def TestModel(x_train, y_train, x_val, y_val, par):
    from utils_network.metrics import r2score, precision, recall, iou, iou_loss, dice_coef, dice_coef_loss, phi_coef, balanced_cross_entropy

    with tf.device("/cpu:0"):
        opt_model = Unet(img_shape=np.append(IM_SHAPE, 1), params=par, path=PATH_OUT)

        if(load_trained_model):
            # load trained model
            conf = NetworkConfig(PATH_OUT+'runs/net2D_021020.ini')  #TODO: GENERALISE THIS
            PATH_MODEL = conf.resume_path+'checkpoints/'
            MODEL_EPOCH = conf.best_epoch
            METRICS = [avail_metrics[m] for m in np.append(conf.loss, conf.metrics)]
            cb = {func.__name__:func for func in METRICS if not isinstance(func, str)}
            trained_model = load_model('%smodel-sem21cm_ep%d.h5' %(PATH_MODEL, MODEL_EPOCH), custom_objects=cb)

            # copy weight in optimization model
            opt_model.set_weights(trained_model.get_weights())

    opt_model = multi_gpu_model(opt_model, gpus=GPU)
    
    METRICS = [matthews_coef, precision, recall, iou] # [matthews_coef, precision, recall, iou]
    opt_model.compile(optimizer=par['optimizer'](lr=lr_normalizer(1e-7, par['optimizer'])), loss=avail_metrics['balanced_cross_entropy'], metrics=METRICS)
    
    results = opt_model.fit(x=x_train, y=y_train, batch_size=32*GPU, epochs=100, validation_data=[X_valid, y_valid], shuffle=True)
    
    return results, opt_model

scan_object = talos.Scan(X_train, y_train, 
                         params=p, 
                         model=TestModel, 
                         experiment_name=PATH_OUT,
                         reduction_metric='val_loss',
                         #fraction_limit=1,
                         random_method='latin_improved')
#scan_object = talos.Scan(X_train, y_train, params=p, model=TestModel, experiment_no='1', grid_downsample=0.01)

# accessing the results data frame
print(scan_object.data.head())

# access the summary details
print(scan_object.details)

scan_object.learning_entropy.to_csv(PATH_OUT+'entropy.csv')
