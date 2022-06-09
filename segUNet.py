import os, random, numpy as np, sys
import tensorflow as tf

from sklearn.model_selection import train_test_split
from datetime import datetime
from glob import glob

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import ELU, LeakyReLU, PReLU, ReLU

from config.net_config import NetworkConfig
from utils_network.networks import Unet, LSTM_Unet, Unet_Reg
from utils_network.metrics import get_avail_metris
from utils_network.callbacks import HistoryCheckpoint, SaveModelCheckpoint, ReduceLR
from utils_network.data_generator import LightConeGenerator, LightConeGenerator_Reg
from utils.other_utils import get_data, get_data_lc
from utils_plot.plotting import plot_loss

# title
print('  _____              _    _ _   _      _   \n / ____|            | |  | | \ | |    | |  \n| (___   ___  __ _  | |  | |  \| | ___| |_ \n \___ \ / _ \/ _` | | |  | | . ` |/ _ \ __|\n ____) |  __/ (_| | | |__| | |\  |  __/ |_ \n|_____/ \___|\__, |  \____/|_| \_|\___|\__|\n              __/ |                        \n             |___/                         \n')

config_file = sys.argv[1]
conf = NetworkConfig(config_file)

# --------------------- NETWORK & RESUME OPTIONS ---------------------
RANDOM_SEED = 2021
BATCH_SIZE = conf.batch_size
DATA_AUGMENTATION = conf.augment
IM_SHAPE = conf.img_shape
COARSE_DIM = conf.chansize
DROPOUT = conf.dropout
KS = conf.kernel_size
EPOCHS = conf.epochs
LOSS = get_avail_metris(conf.loss)
OPTIMIZER = Adam(lr=conf.learn_rate)
ACTIVATION = 'relu'
METRICS = [get_avail_metris(m) for m in conf.metrics]
RECOMPILE = conf.recomplile
GPU = conf.gpus
IO_PATH = conf.io_path
DATASET_PATH = conf.dataset_path
if isinstance(DATASET_PATH, list):
    PATH_TRAIN = IO_PATH+'inputs/'+DATASET_PATH[0]
    PATH_VALID = IO_PATH+'inputs/'+DATASET_PATH[1]
else:
    PATH_TRAIN = IO_PATH+'inputs/'+conf.dataset_path
    PATH_VALID = PATH_TRAIN
ZIPFILE = (0 < len(glob(PATH_TRAIN+'data/*tar.gz')) and 0 < len(glob(PATH_VALID+'data/*tar.gz')))
BEST_EPOCH = conf.best_epoch
# if you want to restart from the previous best model set RESUME_EPOCH = BEST_EPOCH
RESUME_EPOCH = conf.resume_epoch
RESUME_PATH = conf.resume_path
# -------------------------------------------------------------------

if(BEST_EPOCH != 0 and RESUME_EPOCH !=0):
    PATH_OUT = RESUME_PATH
    RESUME_MODEL = '%scheckpoints/model-sem21cm_ep%d.h5' %(RESUME_PATH, BEST_EPOCH)
else:
    RESUME_MODEL = './foo'
    if(len(IM_SHAPE) == 3):
        PATH_OUT = '%soutputs/%s_%dcube/' %(IO_PATH, datetime.now().strftime('%d-%mT%H-%M-%S'), IM_SHAPE[0])
    elif(len(IM_SHAPE) == 2):
        PATH_OUT = '%soutputs/%s_%dslice/' %(IO_PATH, datetime.now().strftime('%d-%mT%H-%M-%S'), IM_SHAPE[0])
    else:
        print('!!! Wrong data dimension !!!')
    os.makedirs(PATH_OUT)
    os.makedirs(PATH_OUT+'/outputs')
    os.makedirs(PATH_OUT+'/source')
    os.makedirs(PATH_OUT+'/checkpoints')

random.seed(RANDOM_SEED)

# copy code to source directory
os.system('cp *.py %s/source' %PATH_OUT)
os.system('cp -r utils %s/source' %PATH_OUT)
os.system('cp -r utils_network %s/source' %PATH_OUT)
os.system('cp -r utils_plot %s/source' %PATH_OUT)
os.system('cp -r config %s/source' %PATH_OUT)
os.system('cp %s %s' %(config_file, PATH_OUT))

# Load data
if not isinstance(DATA_AUGMENTATION, str):
    if isinstance(DATASET_PATH, (list, np.ndarray)):
        print('Load images ...') 
        X_train, y_train = get_data(PATH_TRAIN+'data/', IM_SHAPE, shuffle=True)
        size_train_dataset = X_train.shape[0]
        print('Load masks ...') 
        X_valid, y_valid = get_data(PATH_VALID+'data/', IM_SHAPE, shuffle=True)
        size_valid_dataset = X_valid.shape[0]
    else:
        print('Load dataset ...') 
        X, y = get_data_lc(path=PATH_TRAIN, fname='lc_256Mpc_train', shuffle=True)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=RANDOM_SEED)
        size_train_dataset = X_train.shape[0]
else:
    if isinstance(DATASET_PATH, (list, np.ndarray)):
        print('Data will ber extracted in batches...')
        size_train_dataset, size_valid_dataset = 10000, 1500
        train_idx = np.arange(0, size_train_dataset, dtype=int)
        valid_idx = np.arange(0, size_valid_dataset, dtype=int)
    else:
        print('Data will ber extracted in batches...')
        test_size, datasize = 0.15, 10000
        train_idx = np.arange(0, datasize*(1-test_size), dtype=int)
        size_train_dataset = train_idx.size
        valid_idx = np.arange(datasize*(1-test_size), datasize, dtype=int)

"""
# Create data generator from tensorflow.keras.utils.Sequence
train_generator = LightConeGenerator_Reg(path=PATH_TRAIN, data_temp=train_idx, data_shape=IM_SHAPE, batch_size=BATCH_SIZE, shuffle=True)
valid_generator = LightConeGenerator_Reg(path=PATH_VALID, data_temp=valid_idx, data_shape=IM_SHAPE, batch_size=BATCH_SIZE, shuffle=True)

# Define generator functional
def generator_train():
    multi_enqueuer = tf.keras.utils.OrderedEnqueuer(train_generator, use_multiprocessing=True)
    multi_enqueuer.start(workers=10, max_queue_size=10)
    while True:
        batch_xs, batch_ys1, batch_ys2 = next(multi_enqueuer.get()) 
        #yield(batch_xs, {'output1': batch_ys1, 'output2': batch_ys2})
        yield batch_xs, [batch_ys1, batch_ys2]
        
def generator_valid():
    multi_enqueuer = tf.keras.utils.OrderedEnqueuer(valid_generator, use_multiprocessing=True)
    multi_enqueuer.start(workers=10, max_queue_size=10)
    while True:
        batch_xs, batch_ys1, batch_ys2 = next(multi_enqueuer.get()) 
        #yield(batch_xs, {'output1': batch_ys1, 'output2': batch_ys2})
        yield batch_xs, [batch_ys1, batch_ys2]

# Create dataset from data generator
train_dataset = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32, tf.float32, tf.float32), output_shapes=(tf.TensorShape([None]*(len(IM_SHAPE)+2)), tf.TensorShape([None]*(len(IM_SHAPE)+2)), tf.TensorShape([None]*2)))
valid_dataset = tf.data.Dataset.from_generator(generator_valid, output_types=(tf.float32, tf.float32, tf.float32), output_shapes=(tf.TensorShape([None]*(len(IM_SHAPE)+2)), tf.TensorShape([None]*(len(IM_SHAPE)+2)), tf.TensorShape([None]*2)))
"""

train_generator = LightConeGenerator(path=PATH_TRAIN, data_temp=train_idx, data_shape=IM_SHAPE, zipf=ZIPFILE, batch_size=BATCH_SIZE, tobs=1000, shuffle=True)
valid_generator = LightConeGenerator(path=PATH_VALID, data_temp=valid_idx, data_shape=IM_SHAPE, zipf=ZIPFILE, batch_size=BATCH_SIZE, tobs=1000, shuffle=True)

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
train_dataset = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32, tf.float32), output_shapes=(tf.TensorShape([None]*(len(IM_SHAPE)+2)), tf.TensorShape([None]*(len(IM_SHAPE)+2))))
valid_dataset = tf.data.Dataset.from_generator(generator_valid, output_types=(tf.float32, tf.float32), output_shapes=(tf.TensorShape([None]*(len(IM_SHAPE)+2)), tf.TensorShape([None]*(len(IM_SHAPE)+2))))

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

# Define model or load model
with strategy.scope():
    if(os.path.exists(RESUME_MODEL)):
        print('\nLoaded existing model\n')
        ''' NOTE:
            load_model() is a compiled model ready to be used (unless the saved model was not compiled).
            Therefore re-compiling the model will reset the state of the loaded model. '''
        try:
            model = load_model(RESUME_MODEL)
        except:
            custom_metrics = {m:get_avail_metris(m) for m in np.append(conf.loss, conf.metrics)}
            model = load_model(RESUME_MODEL, custom_objects=cb)

        if(RECOMPILE):
            RESUME_LR = np.loadtxt('%soutputs/lr_ep-%d.txt' %(RESUME_PATH, RESUME_EPOCH))[RESUME_EPOCH-1]
            model.compile(optimizer=Adam(lr=RESUME_LR), loss=LOSS, metrics=METRICS)
            resume_metrics = model.evaluate(X_valid, y_valid, verbose=1)
            RESUME_LOSS = resume_metrics[0]
            msg = '\nScore resumed model:\n'
            for i, res_val in enumerate(resume_metrics):
                msg += ' %s: %.3f   ' %(model.metrics_names[i], res_val) 
            print(msg)
            print("Resume Learning rate: %.3e\n" %(tf.keras.backend.get_value(model.optimizer.lr)))
        else:
            tf.keras.backend.set_value(model.optimizer.lr, RESUME_LR)       # resume learning rate
            RESUME_LOSS = np.loadtxt('%soutputs/val_loss_ep-%d.txt' %(RESUME_PATH, RESUME_EPOCH))[BEST_EPOCH-1]
            print('\nScore resumed model:\n loss: %.3f' %RESUME_LOSS)
    else: 
        print('\nModel on %d GPU\n' %NR_GPUS)
        RESUME_LOSS = None
        hyperpar = {'coarse_dim': COARSE_DIM,
                    'dropout': DROPOUT,
                    'kernel_size': KS,
                    'activation': ACTIVATION,
                    'final_activation': None,
                    'depth': 4}

        #model = Unet_Reg(img_shape=np.append(IM_SHAPE, 1), params=hyperpar, path=PATH_OUT)
        model = Unet(img_shape=np.append(IM_SHAPE, 1), params=hyperpar, path=PATH_OUT)
        #model = LSTM_Unet(img_shape=np.append(IM_SHAPE, 1), coarse_dim=COARSE_DIM, ks=KS, dropout=DROPOUT, path=PATH_OUT)
        #model = Unet3D_time(img_shape=np.append(IM_SHAPE, 1), coarse_dim=COARSE_DIM, ks=KS, dropout=DROPOUT, path=PATH_OUT)
        
        model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
        #model.compile(optimizer=OPTIMIZER, loss=[LOSS, LOSS], loss_weights=[1., 1.], metrics=[METRICS, METRICS])

# define callbacks
callbacks = [EarlyStopping(patience=60, verbose=1),
             ReduceLR(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1, wait=int(RESUME_EPOCH-BEST_EPOCH), best=RESUME_LOSS),
             SaveModelCheckpoint(PATH_OUT+'checkpoints/model-sem21cm_ep{epoch:d}.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, best=RESUME_LOSS),
             HistoryCheckpoint(filepath=PATH_OUT+'/outputs/', verbose=0, save_freq=1, in_epoch=RESUME_EPOCH)]


# model fit
results = model.fit(x=train_dist_dataset,
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS,
                    steps_per_epoch=size_train_dataset//BATCH_SIZE,
                    initial_epoch=RESUME_EPOCH,
                    callbacks=callbacks, 
                    validation_data=valid_dist_dataset,
                    validation_steps=size_valid_dataset//BATCH_SIZE,
                    shuffle=True)


# write info for prediction on config_file
best_model_epoch = max([int(sf[sf.rfind('p')+1:sf.rfind('.')]) for sf in glob(PATH_OUT+'checkpoints/model-sem21cm_ep*.h5')]) 
f = open(glob('%s*.ini' %PATH_OUT)[0], 'a')
f.write('\n\n[PREDICTION]')
f.write('\nMODEL_EPOCH = %d' %best_model_epoch)
f.write('\nMODEL_PATH = %s' %(PATH_OUT))
f.write('\nTTA_WRAP = False')
f.write('\nAUGMENT = False')
f.write('\nEVAL = True')
f.write('\nINDEXES = 0') 
f.close()

# Plot Loss
#plot_loss(output=results, path=PATH_OUT+'outputs/')
os.system('python utils_plot/postpros_plot.py %s/outputs/' %PATH_OUT)

# Write accuracy in the output directory name
#os.system('mv %s %s_acc%d' %(PATH_OUT[:-1], PATH_OUT[:-1], 100*np.max(results.history["val_binary_accuracy"])))
