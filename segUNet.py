import os, random, numpy as np, sys
import tensorflow as tf

#from tensorflow import keras
from sklearn.model_selection import train_test_split
from datetime import datetime
from glob import glob

from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.metrics import matthews_corrcoef
from keras.utils import multi_gpu_model

from config.net_config import NetworkConfig
from utils_network.networks import Unet
from utils_network.metrics import iou, iou_loss, dice_coef, dice_coef_loss, phi_coef, balanced_cross_entropy
from utils_network.callbacks import HistoryCheckpoint, SaveModelCheckpoint, ReduceLR
from utils_network.data_generator import RotateGenerator
from utils.other_utils import get_data, save_cbin
from utils_plot.plotting import plot_loss

# title
print('  _____              _    _ _   _      _   \n / ____|            | |  | | \ | |    | |  \n| (___   ___  __ _  | |  | |  \| | ___| |_ \n \___ \ / _ \/ _` | | |  | | . ` |/ _ \ __|\n ____) |  __/ (_| | | |__| | |\  |  __/ |_ \n|_____/ \___|\__, |  \____/|_| \_|\___|\__|\n              __/ |                        \n             |___/                         \n')

config_file = sys.argv[1]
conf = NetworkConfig(config_file)

avail_metrics = {'binary_accuracy':'binary_accuracy', 'iou':iou, 'dice_coef':dice_coef, 'iou_loss':iou_loss, 'dice_coef_loss':dice_coef_loss, 'phi_coef':phi_coef, 'mse':'mse', 'mae':'mae', 'binary_crossentropy':'binary_crossentropy', 'balanced_cross_entropy':balanced_cross_entropy}                                                                                  

# --------------------- NETWORK & RESUME OPTIONS ---------------------
RANDOM_SEED = 2020
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
PATH_TRAIN = conf.path
BEST_EPOCH = conf.best_epoch
# if you want to restart from the previous best model set RESUME_EPOCH = BEST_EPOCH
RESUME_EPOCH = conf.resume_epoch
RESUME_PATH = conf.resume_path
# -------------------------------------------------------------------

if(BEST_EPOCH != 0 and RESUME_EPOCH !=0):
    PATH_OUT = RESUME_PATH
    RESUME_MODEL = '%smodel-sem21cm_ep%d.h5' %(RESUME_PATH+'checkpoints/', BEST_EPOCH)
    RESUME_LR = np.loadtxt(glob(RESUME_PATH+'outputs/lr_ep-*.txt')[0])[RESUME_EPOCH-1]
else:
    RESUME_MODEL = './dummy'
    if(len(IM_SHAPE) == 3):
        PATH_OUT = '/ichec/work/subgridEoRevol/michele/output_SegNet/'+ datetime.now().strftime('%d-%mT%H-%M-%S') + '_%dcube/' %IM_SHAPE[0]    
    elif(len(IM_SHAPE) == 2):
        PATH_OUT = '/ichec/work/subgridEoRevol/michele/output_SegNet/'+ datetime.now().strftime('%d-%mT%H-%M-%S') + '_%dslice/' %IM_SHAPE[0]
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
if isinstance(PATH_TRAIN, (list, np.ndarray)):
    print('Load images ...') 
    X_train, y_train = get_data(PATH_TRAIN[0]+'data/', IM_SHAPE, shuffle=True)
    size_train_dataset = X_train.shape[0]
    print('Load masks ...') 
    X_valid, y_valid = get_data(PATH_TRAIN[1]+'data/', IM_SHAPE, shuffle=True)
    size_valid_dataset = X_valid.shape[0]
else:
    print('Load dataset ...') 
    X, y = get_data(PATH_TRAIN+'data/', IM_SHAPE, shuffle=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=RANDOM_SEED)
    size_train_dataset = X_train.shape[0]

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
        #for func in [iou, dice_coef, 'binary_accuracy']:
        for func in np.append(METRICS, LOSS): 
             if not isinstance(func, str): 
                cb[func.__name__] = func 

        model = load_model(RESUME_MODEL, custom_objects=cb)
    
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
        tf.keras.backend.set_value(model.optimizer.lr, RESUME_LR)       # change learning rate
        resume_loss = np.loadtxt(glob(RESUME_PATH+'outputs/val_loss_ep-*.txt')[0])[BEST_EPOCH-1]
        print('\nScore resumed model:\n loss: %.3f' %resume_loss)
else: 
    print('\nModel created')
    if(GPU == None):
        model = Unet(img_shape=np.append(IM_SHAPE, 1), coarse_dim=COARSE_DIM, ks=KS, dropout=DROPOUT, path=PATH_OUT)
    else:
        print('\nModel on GPU\n')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        with tf.device("/cpu:0"):
            model = Unet(img_shape=np.append(IM_SHAPE, 1), coarse_dim=COARSE_DIM, ks=KS, dropout=DROPOUT, path=PATH_OUT)
        model = multi_gpu_model(model, gpus=GPU)
    
    resume_loss = None
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)


callbacks = [EarlyStopping(patience=21, verbose=1),
             ReduceLR(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1, wait=int(RESUME_EPOCH-BEST_EPOCH), best=resume_loss),
             SaveModelCheckpoint(PATH_OUT+'checkpoints/model-sem21cm_ep{epoch:d}.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, best=resume_loss),
             HistoryCheckpoint(filepath=PATH_OUT+'/outputs/', verbose=0, save_freq=1, in_epoch=RESUME_EPOCH)]


if not (DATA_AUGMENTATION):
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
        train_generator = RotateGenerator(data=X_train, label=y_train, batch_size=BATCH_SIZE,
                                        rotate_axis='random', rotate_angle='random', shuffle=True)
        valid_generator = RotateGenerator(data=X_valid, label=y_valid, batch_size=BATCH_SIZE,
                                        rotate_axis='random', rotate_angle='random', shuffle=True)
    elif(DATA_AUGMENTATION == 'NOISESMT'):
        print('\nData augmentation: Add noise cube and smooth 21cm cube...\n')
        train_generator = DataGenerator(data=X_train, label=y_train, batch_size=BATCH_SIZE,
                                        tobs=1000, path=PATH_TRAIN, shuffle=True)
        valid_generator = DataGenerator(data=X_valid, label=y_valid, batch_size=BATCH_SIZE,
                                        tobs=1000, path=PATH_TRAIN, shuffle=True)

    results = model.fit_generator(generator=train_generator, 
                                  steps_per_epoch=(size_train_dataset//BATCH_SIZE),
                                  epochs=EPOCHS,
                                  initial_epoch=RESUME_EPOCH,
                                  callbacks=callbacks,
                                  validation_data=valid_generator,
                                  shuffle=True)


# Plot Loss
plot_loss(output=results, path=PATH_OUT+'outputs/')


# copy weight to output directory
os.system('mv %s %s_acc%d' %(PATH_OUT[:-1], PATH_OUT[:-1], 100*np.max(results.history["val_binary_accuracy"])))
