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
from utils_network.networks import Unet, LSTM_Unet, Unet_Reg
from utils_network.data_generator import LightConeGenerator, LightConeGenerator_Reg
from utils.other_utils import get_data, get_batch, get_data_lc
from config.net_config import NetworkConfig
from utils_network.metrics import get_avail_metris

#===================================
config_file = sys.argv[1]
conf = NetworkConfig(config_file)
BATCH_SIZE = conf.BATCH_SIZE
METRICS = [get_avail_metris(m) for m in conf.METRICS]
#===================================
RANDOM_SEED = 2021

if isinstance(conf.DATASET_PATH, list):
    PATH_TRAIN = conf.IO_PATH+'inputs/'+conf.DATASET_PATH[0]
    PATH_VALID = conf.IO_PATH+'inputs/'+conf.DATASET_PATH[1]
else:
    PATH_TRAIN = conf.IO_PATH+'inputs/'+conf.dataset_path
    PATH_VALID = PATH_TRAIN

PATH_OUT = 'tests/test2/'

# Load Data
size_train_dataset, size_valid_dataset = 10000, 1500
train_idx = np.arange(0, size_train_dataset, dtype=int)
valid_idx = np.arange(0, size_valid_dataset, dtype=int)

train_generator = LightConeGenerator(path=PATH_TRAIN, data_temp=train_idx, data_shape=conf.IM_SHAPE, zipf=ZIPFILE, batch_size=BATCH_SIZE, tobs=1000, shuffle=True)
valid_generator = LightConeGenerator(path=PATH_VALID, data_temp=valid_idx, data_shape=conf.IM_SHAPE, zipf=ZIPFILE, batch_size=BATCH_SIZE, tobs=1000, shuffle=True)

# Define generator functional
def generator_train():
    multi_enqueuer = tf.keras.utils.OrderedEnqueuer(train_generator, use_multiprocessing=True)
    multi_enqueuer.start(workers=10, max_queue_size=10)
    while True:
        batch_xs, batch_ys = next(multi_enqueuer.get()) 
        yield batch_xs, batch_ys

def generator_valid():
    multi_enqueuer = tf.keras.utils.OrderedEnqueuer(valid_generator, use_multiprocessing=True)
    multi_enqueuer.start(workers=10, max_queue_size=10)
    while True:
        batch_xs, batch_ys = next(multi_enqueuer.get()) 
        yield batch_xs, batch_ys

# Create dataset from data generator
train_dataset = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32, tf.float32), output_shapes=(tf.TensorShape([None]*(len(conf.IM_SHAPE)+2)), tf.TensorShape([None]*(len(conf.IM_SHAPE)+2))))
valid_dataset = tf.data.Dataset.from_generator(generator_valid, output_types=(tf.float32, tf.float32), output_shapes=(tf.TensorShape([None]*(len(conf.IM_SHAPE)+2)), tf.TensorShape([None]*(len(conf.IM_SHAPE)+2))))

# Define GPU distribution strategy
strategy = tf.distribute.MirroredStrategy()
NR_GPUS = strategy.num_replicas_in_sync
print ('Number of devices: %d' %NR_GPUS)
BATCH_SIZE *= NR_GPUS

# Distribute the dataset to the devices
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
valid_dist_dataset = strategy.experimental_distribute_dataset(valid_dataset)

# Set the sharding policy to DATA
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
train_dataset.with_options(options)
valid_dataset.with_options(options)
print(size_train_dataset, size_valid_dataset)

# Network Hyperparameters
p = {'coarse_dim': [128, 256, 512],
     'dropout':[0.05, 0.1, 0.15],
     'kernel_size':[3, 4, 6],
     'activation': [ReLU(), LeakyReLU(), ELU()],
     'final_activation': ['sigmoid', 'softmax'],
     'lr':[10**(-i) for i in range(5,7)],
     'optimizer': [Adam, RMSprop, Nadam],
     'epochs':[200],
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

    with strategy.scope():
        opt_model = Unet(img_shape=np.append(conf.IM_SHAPE, 1), params=par, path=PATH_OUT)

        model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

        opt_model.compile(optimizer=par['optimizer'](lr=lr_normalizer(1e-7, par['optimizer'])), loss=get_avail_metris(conf.LOSS), metrics=METRICS)
    
    results = opt_model.fit(x=x_train, y=y_train, batch_size=32*GPU, epochs=100, validation_data=[X_valid, y_valid], shuffle=True)
    # model fit
    results = model.fit(x=train_dist_dataset,
                        batch_size=BATCH_SIZE, 
                        epochs=conf.EPOCHS,
                        steps_per_epoch=size_train_dataset//BATCH_SIZE,
                        callbacks=callbacks, 
                        validation_data=valid_dist_dataset,
                        validation_steps=size_valid_dataset//BATCH_SIZE,
                        shuffle=True)

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
