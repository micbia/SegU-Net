import os, random, numpy as np, sys
import tensorflow as tf
from glob import glob

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import ELU, LeakyReLU, PReLU, ReLU

from config.net_config import NetworkConfig
from utils_network.networks import Unet, SERENEt, FullSERENEt
from utils_network.metrics import get_avail_metris
from utils_network.callbacks import HistoryCheckpoint, SaveModelCheckpoint, ReduceLR
from utils_network.data_generator import LightConeGenerator, LightConeGenerator_SERENEt, LightConeGenerator_FullSERENEt
from utils.other_utils import get_data, get_data_lc, config_paths
from utils_plot.plotting import plot_loss

# title
print('  _____              _    _ _   _      _   \n / ____|            | |  | | \ | |    | |  \n| (___   ___  __ _  | |  | |  \| | ___| |_ \n \___ \ / _ \/ _` | | |  | | . ` |/ _ \ __|\n ____) |  __/ (_| | | |__| | |\  |  __/ |_ \n|_____/ \___|\__, |  \____/|_| \_|\___|\__|\n              __/ |                        \n             |___/                         \n')

config_file = sys.argv[1]
conf = NetworkConfig(config_file)

# --------------------- NETWORK & RESUME OPTIONS ---------------------
FREEZE = False
TYPE_NET = conf.AUGMENT
RANDOM_SEED = 2022
BATCH_SIZE = conf.BATCH_SIZE
METRICS = [get_avail_metris(m) for m in conf.METRICS]
if(isinstance(conf.LOSS, list)):
    LOSS = [get_avail_metris(loss) for loss in conf.LOSS]
    LOSS = {"out_imgSeg": LOSS[0], "out_imgRec": LOSS[1]}
    LOSS_WEIGHTS = {"out_imgSeg": 0.5, "out_imgRec": 0.5}
else:
    LOSS = get_avail_metris(conf.LOSS)
OPTIMIZER = Adam(lr=conf.LR)
ACTIVATION = 'relu'
if isinstance(conf.DATASET_PATH, list):
    PATH_TRAIN = conf.IO_PATH+'inputs/'+conf.DATASET_PATH[0]
    PATH_VALID = conf.IO_PATH+'inputs/'+conf.DATASET_PATH[1]
else:
    PATH_TRAIN = conf.IO_PATH+'inputs/'+conf.dataset_path
    PATH_VALID = PATH_TRAIN
ZIPFILE = (0 < len(glob(PATH_TRAIN+'data/*tar.gz')) and 0 < len(glob(PATH_VALID+'data/*tar.gz')))
# TODO: if you want to restart from the previous best model set conf.RESUME_EPOCH = conf.BEST_EPOCH and loss need to be cut accordingly
# -------------------------------------------------------------------
random.seed(RANDOM_SEED)
path_scratch = '/scratch/snx3000/mibianco/output_segunet/'
output_prefix = ''
PATH_OUT, RESUME_MODEL = config_paths(conf=conf, path_scratch=path_scratch, prefix=output_prefix)

if not (os.path.exists(PATH_OUT+'source')):
    # copy code to source directory
    os.system('cp *.py %s/source' %PATH_OUT)
    os.system('cp -r utils %s/source' %PATH_OUT)
    os.system('cp -r utils_network %s/source' %PATH_OUT)
    os.system('cp -r utils_plot %s/source' %PATH_OUT)
    os.system('cp -r config %s/source' %PATH_OUT)
os.system('cp %s %s' %(config_file, PATH_OUT))

# Define GPU distribution strategy
strategy = tf.distribute.MirroredStrategy()
NR_GPUS = strategy.num_replicas_in_sync
print ('Number of GPU devices: %d' %NR_GPUS)
BATCH_SIZE *= NR_GPUS

# Load data
size_train_dataset, size_valid_dataset = 10000, 1500
train_idx = np.arange(0, size_train_dataset, dtype=int)
valid_idx = np.arange(0, size_valid_dataset, dtype=int)
#train_idx = np.loadtxt(PATH_TRAIN+'good_data.txt')
#valid_idx = np.loadtxt(PATH_VALID+'good_data.txt')

# Create data generator from tensorflow.keras.utils.Sequence
if(TYPE_NET == 'full_serene'):
    train_generator = LightConeGenerator_FullSERENEt(path=PATH_TRAIN, data_temp=train_idx, data_shape=conf.IM_SHAPE, batch_size=BATCH_SIZE, shuffle=True)
    valid_generator = LightConeGenerator_FullSERENEt(path=PATH_VALID, data_temp=valid_idx, data_shape=conf.IM_SHAPE, batch_size=BATCH_SIZE, shuffle=True)

    # Define generator functional
    def generator_train():
        multi_enqueuer = tf.keras.utils.OrderedEnqueuer(train_generator, use_multiprocessing=False)
        multi_enqueuer.start(workers=10, max_queue_size=10)
        while True:
            batch_xs, batch_ys1, batch_ys2 = next(multi_enqueuer.get()) 
            yield (batch_xs, {'out_imgSeg':batch_ys1, 'out_imgRec':batch_ys2})
            
    def generator_valid():
        multi_enqueuer = tf.keras.utils.OrderedEnqueuer(valid_generator, use_multiprocessing=False)
        multi_enqueuer.start(workers=10, max_queue_size=10)
        while True:
            batch_xs, batch_ys1, batch_ys2 = next(multi_enqueuer.get()) 
            yield (batch_xs, {'seg_out_img':batch_ys1, 'out_imgRec':batch_ys2})

    # Create dataset from data generator
    train_dataset = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32, {'rec_out_img': tf.float32, 'seg_out_img': tf.float32}))
    valid_dataset = tf.data.Dataset.from_generator(generator_valid, output_types=(tf.float32, {'rec_out_img': tf.float32, 'seg_out_img': tf.float32}))
elif(TYPE_NET == 'serene'):
    train_generator = LightConeGenerator_SERENEt(path=PATH_TRAIN, data_temp=train_idx, data_shape=conf.IM_SHAPE, batch_size=BATCH_SIZE, shuffle=True)
    valid_generator = LightConeGenerator_SERENEt(path=PATH_VALID, data_temp=valid_idx, data_shape=conf.IM_SHAPE, batch_size=BATCH_SIZE, shuffle=True)

    # Define generator functional
    def generator_train():
        multi_enqueuer = tf.keras.utils.OrderedEnqueuer(train_generator, use_multiprocessing=False)
        multi_enqueuer.start(workers=10, max_queue_size=10)
        while True:
            batch_xs, batch_ys1, batch_ys2 = next(multi_enqueuer.get()) 
            yield (batch_xs, {'out_imgSeg':batch_ys1, 'out_imgRec':batch_ys2})
            
    def generator_valid():
        multi_enqueuer = tf.keras.utils.OrderedEnqueuer(valid_generator, use_multiprocessing=False)
        multi_enqueuer.start(workers=10, max_queue_size=10)
        while True:
            batch_xs, batch_ys1, batch_ys2 = next(multi_enqueuer.get()) 
            yield (batch_xs, {'seg_out_img':batch_ys1, 'out_imgRec':batch_ys2})

    # Create dataset from data generator
    train_dataset = tf.data.Dataset.from_generator(generator_train, output_types=({'Image1': tf.float32, 'Image2': tf.float32}, {'out_img': tf.float32}))
    valid_dataset = tf.data.Dataset.from_generator(generator_valid, output_types=({'Image1': tf.float32, 'Image2': tf.float32}, {'out_img': tf.float32}))
elif(TYPE_NET == 'segunet' or TYPE_NET == 'recunet'):
    if(TYPE_NET == 'segunet'):
        DATA_TYPE = 'xH'
    elif(TYPE_NET == 'recunet'):
        DATA_TYPE = 'dT2'
    
    train_generator = LightConeGenerator(path=PATH_TRAIN, data_temp=train_idx, data_shape=conf.IM_SHAPE, zipf=ZIPFILE, batch_size=BATCH_SIZE, data_type=DATA_TYPE, shuffle=True)
    valid_generator = LightConeGenerator(path=PATH_VALID, data_temp=valid_idx, data_shape=conf.IM_SHAPE, zipf=ZIPFILE, batch_size=BATCH_SIZE, data_type=DATA_TYPE, shuffle=True)

    # Define generator functional
    def generator_train():
        multi_enqueuer = tf.keras.utils.OrderedEnqueuer(train_generator, use_multiprocessing=False)
        multi_enqueuer.start(workers=10, max_queue_size=10)
        while True:
            batch_xs, batch_ys = next(multi_enqueuer.get()) 
            yield batch_xs, batch_ys

    def generator_valid():
        multi_enqueuer = tf.keras.utils.OrderedEnqueuer(valid_generator, use_multiprocessing=False)
        multi_enqueuer.start(workers=10, max_queue_size=10)
        while True:
            batch_xs, batch_ys = next(multi_enqueuer.get()) 
            yield batch_xs, batch_ys

    # Create dataset from data generator
    train_dataset = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32, tf.float32), output_shapes=(tf.TensorShape([None]*(len(conf.IM_SHAPE)+2)), tf.TensorShape([None]*(len(conf.IM_SHAPE)+2))))
    valid_dataset = tf.data.Dataset.from_generator(generator_valid, output_types=(tf.float32, tf.float32), output_shapes=(tf.TensorShape([None]*(len(conf.IM_SHAPE)+2)), tf.TensorShape([None]*(len(conf.IM_SHAPE)+2))))

# Distribute the dataset to the devices
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
valid_dist_dataset = strategy.experimental_distribute_dataset(valid_dataset)

# Set the sharding policy to DATA
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
train_dataset.with_options(options)
valid_dataset.with_options(options)

# Define model or load model
with strategy.scope():
    if(os.path.exists(RESUME_MODEL)):
        print('Loaded existing model:\n %s' %RESUME_MODEL)
        ''' NOTE: load_model() is a compiled model ready to be used (unless the saved model was not compiled).
            Therefore re-compiling the model will reset the state of the loaded model. '''
        try:
            custom_metrics = {m:get_avail_metris(m) for m in np.append(conf.LOSS, conf.METRICS)}
            model = load_model(RESUME_MODEL, custom_objects=custom_metrics)
        except:
            model = load_model(RESUME_MODEL)

        if(FREEZE):
            for l in model.layers:
                l.trainable = False
        
        if(conf.RECOMPILE):
            #RESUME_LR = np.loadtxt('%soutputs/lr_ep-%d.txt' %(conf.RESUME_PATH, conf.RESUME_EPOCH))[conf.BEST_EPOCH-1]
            RESUME_LR = conf.LR
            model.compile(optimizer=Adam(lr=RESUME_LR), loss=LOSS, metrics=METRICS)
            resume_metrics = model.evaluate(valid_dist_dataset, steps=size_valid_dataset//BATCH_SIZE, verbose=1)
            RESUME_LOSS = resume_metrics[0]
            msg = ' Score resumed model:\n'
            for i, res_val in enumerate(resume_metrics):
                msg += ' %s = %.3f\n' %(model.metrics_names[i], res_val) 
            print(msg)
        else:
            RESUME_LR = np.loadtxt('%soutputs/lr_ep-%d.txt' %(conf.RESUME_PATH, conf.RESUME_EPOCH))[conf.RESUME_EPOCH-1]
            #tf.keras.backend.set_value(model.optimizer.lr, RESUME_LR)       # resume learning rate (this works on .h5 saved model but not .tf)
            RESUME_LOSS = np.loadtxt('%soutputs/val_loss_ep-%d.txt' %(conf.RESUME_PATH, conf.RESUME_EPOCH))[conf.BEST_EPOCH-1]
            print('\n Loss of resumed model: %.3e\t(%s)' %(RESUME_LOSS, conf.LOSS))
            model.compile(optimizer=Adam(lr=RESUME_LR), loss=LOSS, metrics=METRICS)

        print(' Resume Learning rate: %.3e' %(tf.keras.backend.get_value(model.optimizer.lr)))
    else: 
        print('\nModel on %d GPU\n' %NR_GPUS)
        RESUME_LOSS = None
        hyperpar = {'coarse_dim': conf.COARSE_DIM,
                    'dropout': conf.DROPOUT,
                    'kernel_size': conf.KERNEL_SIZE,
                    'activation': ACTIVATION,
                    'final_activation': None,
                    'depth': 4}

        # for Regression image + astropars
        if(TYPE_NET == 'full_serene'):
            model = FullSERENEt(img_shape=np.append(conf.IM_SHAPE, 1), params=hyperpar, path=PATH_OUT)
            model.compile(optimizer=OPTIMIZER, loss=[LOSS, LOSS], loss_weights=LOSS_WEIGHTS, metrics=[METRICS, METRICS])
        elif(TYPE_NET == 'serene'):
            model = SERENEt(img_shape1=np.append(conf.IM_SHAPE, 1), img_shape2=np.append(conf.IM_SHAPE, 1), params=hyperpar, path=PATH_OUT)
            model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
        elif(TYPE_NET == 'segunet' or TYPE_NET == 'recunet'):
            model = Unet(img_shape=np.append(conf.IM_SHAPE, 1), params=hyperpar, path=PATH_OUT)
            model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

# define callbacks
callbacks = [EarlyStopping(patience=30, verbose=1),
             ReduceLR(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7, verbose=1, wait=int(conf.RESUME_EPOCH-conf.BEST_EPOCH), best=RESUME_LOSS),
             SaveModelCheckpoint(PATH_OUT+'checkpoints/model-sem21cm_ep{epoch:d}.tf', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, best=RESUME_LOSS),
             HistoryCheckpoint(filepath=PATH_OUT+'outputs/', verbose=0, save_freq=1, in_epoch=conf.RESUME_EPOCH)]


# model fit
results = model.fit(x=train_dist_dataset,
                    batch_size=BATCH_SIZE, 
                    epochs=conf.EPOCHS,
                    steps_per_epoch=size_train_dataset//BATCH_SIZE,
                    initial_epoch=conf.RESUME_EPOCH,
                    callbacks=callbacks, 
                    validation_data=valid_dist_dataset,
                    validation_steps=size_valid_dataset//BATCH_SIZE,
                    shuffle=True)

# Plot Loss
#plot_loss(output=results, path=PATH_OUT+'outputs/')
os.system('python utils_plot/postpros_plot.py %s' %PATH_OUT)
