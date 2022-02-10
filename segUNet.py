import os, random, numpy as np, sys, random, pickle
import tensorflow as tf

from sklearn.model_selection import train_test_split
from datetime import datetime
from glob import glob

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.metrics import matthews_corrcoef
from tensorflow.keras.utils import multi_gpu_model

from config.net_config import NetworkConfig
from utils_network.networks import Unet, LSTM_Unet
from utils_network.metrics import r2score, precision, recall, iou, iou_loss, dice_coef, dice_coef_loss, phi_coef, balanced_cross_entropy
from utils_network.callbacks import HistoryCheckpoint, SaveModelCheckpoint, ReduceLR
from utils_network.data_generator import RotateGenerator, DataGenerator, LightConeGenerator
from utils.other_utils import get_data, get_data_lc, get_batch, save_cbin
from utils_plot.plotting import plot_loss

# title
print('  _____              _    _ _   _      _   \n / ____|            | |  | | \ | |    | |  \n| (___   ___  __ _  | |  | |  \| | ___| |_ \n \___ \ / _ \/ _` | | |  | | . ` |/ _ \ __|\n ____) |  __/ (_| | | |__| | |\  |  __/ |_ \n|_____/ \___|\__, |  \____/|_| \_|\___|\__|\n              __/ |                        \n             |___/                         \n')

config_file = sys.argv[1]
conf = NetworkConfig(config_file)

#avail_metrics = {'r2score':r2score, 'binary_accuracy':'binary_accuracy', 'iou':iou, 'dice_coef':dice_coef, 'iou_loss':iou_loss, 'dice_coef_loss':dice_coef_loss, 'phi_coef':phi_coef, 'mse':'mse', 'mae':'mae', 'binary_crossentropy':'binary_crossentropy', 'balanced_cross_entropy':balanced_cross_entropy, 'precision':precision, 'recall':recall}
with open('utils_network/avail_metrics.pkl', 'rb') as data:
    avail_metrics = pickle.loads(data.read())

# --------------------- NETWORK & RESUME OPTIONS ---------------------
RANDOM_SEED = 2021
BATCH_SIZE = conf.batch_size
DATA_AUGMENTATION = conf.augment
IM_SHAPE = conf.img_shape
COARSE_DIM = conf.chansize
DROPOUT = conf.dropout
KS = conf.kernel_size
EPOCHS = conf.epochs
LOSS = avail_metrics[conf.loss]
OPTIMIZER = Adam(lr=conf.learn_rate)
METRICS = [avail_metrics[m] for m in conf.metrics]
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
    RESUME_LR = np.loadtxt('%soutputs/lr_ep-%d.txt' %(RESUME_PATH, RESUME_EPOCH))[RESUME_EPOCH-1]
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
        #X, y = get_data(PATH_TRAIN+'data/', IM_SHAPE, shuffle=True)
        #X, y = get_batch(path=PATH_TRAIN, img_shape=IM_SHAPE, size=30000, dataset_size=30000)
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

# Define model or load model
if(os.path.exists(RESUME_MODEL)):
    print('\nLoaded existing model\n')
    ''' NOTE:
        load_model() is a compiled model ready to be used (unless the saved model was not compiled).
        Therefore re-compiling the model will reset the state of the loaded model. '''
    try:
        model = load_model(RESUME_MODEL)
    except:
        cb = {} 
        for func in np.append(METRICS, LOSS):
            if not isinstance(func, str): 
                cb[func.__name__] = func 
        model = load_model(RESUME_MODEL, custom_objects=cb)

    """ TODO: not sure if loaded model is distributed on GPU
    try:
        if(GPU == None or GPU == 0):
            model = load_model(RESUME_MODEL)
        else:
            with tf.device("/cpu:0"):
                model = load_model(RESUME_MODEL)
            model = multi_gpu_model(model, gpus=GPU)
            print('\nModel on %d GPU\n' %GPU)
    except:
        cb = {} 
        for func in np.append(METRICS, LOSS): 
             if not isinstance(func, str): 
                cb[func.__name__] = func 
        if(GPU == None or GPU == 0):
            model = load_model(RESUME_MODEL)
        else:
            with tf.device("/cpu:0"):
                model = load_model(RESUME_MODEL, custom_objects=cb)
            model = multi_gpu_model(model, gpus=GPU)
            print('\nModel on %d GPU\n' %GPU)
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
    """
    if(RECOMPILE):
        model.compile(optimizer=Adam(lr=RESUME_LR), loss=LOSS, metrics=METRICS)
        resume_metrics = model.evaluate(X_valid, y_valid, verbose=1)
        resume_loss = resume_metrics[0]
        msg = '\nScore resumed model:\n'
        for i, res_val in enumerate(resume_metrics):
            msg += ' %s: %.3f   ' %(model.metrics_names[i], res_val) 
        print(msg)
        print("Resume Learning rate: %.3e\n" %(tf.keras.backend.get_value(model.optimizer.lr)))
    else:
        tf.keras.backend.set_value(model.optimizer.lr, RESUME_LR)       # resume learning rate
        resume_loss = np.loadtxt('%soutputs/val_loss_ep-%d.txt' %(RESUME_PATH, RESUME_EPOCH))[BEST_EPOCH-1]
        print('\nScore resumed model:\n loss: %.3f' %resume_loss)
else: 
    print('\nModel created')
    resume_loss = None
    if(GPU == None or GPU == 0):
        model = Unet(img_shape=np.append(IM_SHAPE, 1), coarse_dim=COARSE_DIM, ks=KS, dropout=DROPOUT, path=PATH_OUT)
        #model = LSTM_Unet(img_shape=np.append(IM_SHAPE, 1), coarse_dim=COARSE_DIM, ks=KS, dropout=DROPOUT, path=PATH_OUT)
    else:
        print('\nModel on %d GPU\n' %GPU)
        with tf.device("/cpu:0"):
            model = Unet(img_shape=np.append(IM_SHAPE, 1), coarse_dim=COARSE_DIM, ks=KS, dropout=DROPOUT, path=PATH_OUT)
        model = multi_gpu_model(model, gpus=GPU)
    
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

# define callbacks
callbacks = [EarlyStopping(patience=15, verbose=1),
             ReduceLR(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1, wait=int(RESUME_EPOCH-BEST_EPOCH), best=resume_loss),
             SaveModelCheckpoint(PATH_OUT+'checkpoints/model-sem21cm_ep{epoch:d}.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, best=resume_loss),
             HistoryCheckpoint(filepath=PATH_OUT+'/outputs/', verbose=0, save_freq=1, in_epoch=RESUME_EPOCH)]

# model fit
if not isinstance(DATA_AUGMENTATION, str):
    results = model.fit(x=X_train, y=y_train,
                        batch_size=BATCH_SIZE, 
                        epochs=EPOCHS,
                        initial_epoch=RESUME_EPOCH,
                        callbacks=callbacks, 
                        validation_data=(X_valid, y_valid), 
                        shuffle=True)
else:
    if(DATA_AUGMENTATION == 'ROT'):
        print('\nData augmentation: random rotation of 90, 180, 270 or 360 deg for x,y or z-axis...\n')
        train_generator = RotateGenerator(data=X_train, label=y_train, batch_size=BATCH_SIZE, rotate_axis='random', rotate_angle='random', shuffle=True)
        valid_generator = RotateGenerator(data=X_valid, label=y_valid, batch_size=BATCH_SIZE, rotate_axis='random', rotate_angle='random', shuffle=True)
    elif(DATA_AUGMENTATION == 'NOISESMT'):
        print('\nData augmentation: Add noise cube and smooth 21cm cube...\n')
        train_generator = DataGenerator(path=PATH_TRAIN, data_temp=train_idx, data_shape=IM_SHAPE, zipf=ZIPFILE, batch_size=BATCH_SIZE, tobs=1000, shuffle=True)
        valid_generator = DataGenerator(path=PATH_VALID, data_temp=valid_idx, data_shape=IM_SHAPE, zipf=ZIPFILE, batch_size=BATCH_SIZE, tobs=1000, shuffle=True)
    elif(DATA_AUGMENTATION == 'LC'):
        print('\nData augmentation: Create LC data with noise cone and smooth...\n')
        train_generator = LightConeGenerator(path=PATH_TRAIN, data_temp=train_idx, data_shape=IM_SHAPE, zipf=ZIPFILE, batch_size=BATCH_SIZE, tobs=1000, shuffle=True)
        valid_generator = LightConeGenerator(path=PATH_VALID, data_temp=valid_idx, data_shape=IM_SHAPE, zipf=ZIPFILE, batch_size=BATCH_SIZE, tobs=1000, shuffle=True)

    if(GPU == None or GPU == 0):
        results = model.fit(x=train_generator, #y=y_train,
                            batch_size=BATCH_SIZE, 
                            epochs=EPOCHS,
                            initial_epoch=RESUME_EPOCH,
                            callbacks=callbacks, 
                            validation_data=valid_generator, 
                            shuffle=True)
        """
        results = model.fit_generator(generator=train_generator, 
                                    steps_per_epoch=(size_train_dataset//BATCH_SIZE),
                                    epochs=EPOCHS,
                                    initial_epoch=RESUME_EPOCH,
                                    callbacks=callbacks,
                                    validation_data=valid_generator,
                                    shuffle=True)
        """
    else:
        results = model.fit(x=train_generator, #y=y_train,
                    batch_size=BATCH_SIZE*GPU, 
                    epochs=EPOCHS,
                    initial_epoch=RESUME_EPOCH,
                    callbacks=callbacks, 
                    validation_data=valid_generator, 
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
