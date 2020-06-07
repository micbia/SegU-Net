import os, sys, numpy as np, matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

import tools21cm as t2c

from matplotlib import gridspec
from tqdm import tqdm

from keras.optimizers import Adam
from keras.models import load_model

from config.net_config import PredictionConfig
from utils_network.networks import Unet
from utils_network.metrics import iou, iou_loss, dice_coef, dice_coef_loss, balanced_cross_entropy, phi_coef
from utils_network.data_generator import TTA_ModelWrapper
from utils.other_utils import get_data, save_cbin
from utils_plot.plotting import plot_sample, plot_sample3D, plot_phicoef
from utils_network.data_generator import DataGenerator

avail_metrics = {'binary_accuracy':'binary_accuracy', 'iou':iou, 'dice_coef':dice_coef, 'iou_loss':iou_loss, 'dice_coef_loss':dice_coef_loss, 'phi_coef':phi_coef, 'mse':'mse', 'mae':'mae', 'binary_crossentropy':'binary_crossentropy', 'balanced_cross_entropy':balanced_cross_entropy}                                                                                  

title_a = '\t\t _    _ _   _      _   \n\t\t| |  | | \ | |    | |  \n\t\t| |  | |  \| | ___| |_ \n\t\t| |  | | . ` |/ _ \ __|\n\t\t| |__| | |\  |  __/ |_ \n\t\t \____/|_| \_|\___|\__|\n'
title_b = ' _____              _ _      _         ___  __                \n|  __ \            | (_)    | |       |__ \/_ |               \n| |__) | __ ___  __| |_  ___| |_ ___     ) || | ___ _ __ ___  \n|  ___/ `__/ _ \/ _` | |/ __| __/ __|   / / | |/ __| `_ ` _ \ \n| |   | | |  __/ (_| | | (__| |_\__ \  / /_ | | (__| | | | | |\n|_|   |_|  \___|\__,_|_|\___|\__|___/ |____||_|\___|_| |_| |_|\n'
print(title_a+'\n'+title_b)

config_file = sys.argv[1]
conf = PredictionConfig(config_file)

MODEL_EPOCH = conf.model_epoch
IM_SHAPE = conf.img_shape
METRICS = [avail_metrics[m] for m in conf.metrics]
TTA_WRAP = conf.tta_wrap
AUGMENTATION = conf.augmentation
EVAL = conf.val
PATH_PRED = conf.path_pred
PATH_OUT = conf.path_out
IDX = conf.indexes

# Load data
print('Load dataset ...') 
# get parameters of dataset
idxs, redshift, eff_fact, Rmfp, Tvir, xn = np.loadtxt(PATH_PRED+'astro_params.txt', unpack=True)

# Get data
X, y = get_data(PATH_PRED+'data/', IM_SHAPE)
size_pred_dataset = X.shape[0]            

# Data augmentation by stucking togheter prediction datas
if(AUGMENTATION != False and AUGMENTATION > 1):
    for i in range(AUGMENTATION-1):
        generator = DataGenerator(data=X, label=y, batch_size=size_pred_dataset, rotate_axis='random', rotate_angle='random', shuffle=True)
        gen = generator.__getitem__(index=0)
        X, y = np.vstack((X, gen[0])), np.vstack((y, gen[1]))
        
        idxs_next, redshift_next, eff_fact_next, Rmfp_next, Tvir_next, xn_next = np.loadtxt(PATH_PRED+'astro_params.txt', unpack=True)
        idxs = np.hstack((idxs, idxs_next))
        redshift = np.hstack((redshift, redshift_next))
        eff_fact = np.hstack((eff_fact, eff_fact_next))
        Rmfp = np.hstack((Rmfp, Rmfp_next))
        Tvir = np.hstack((Tvir, Tvir_next))
        xn = np.hstack((xn, xn_next))

# Load model
try:
    model = load_model('%smodel-sem21cm_ep%d.h5' %(PATH_OUT+'checkpoints/', MODEL_EPOCH))
except:
    cb = {} 
    for func in METRICS: 
        if not isinstance(func, str): 
            cb[func.__name__] = func 
    model = load_model('%smodel-sem21cm_ep%d.h5' %(PATH_OUT+'checkpoints/', MODEL_EPOCH), custom_objects=cb)


# Create predictions output directory
try:
    os.makedirs(PATH_OUT+'predictions')
    PATH_OUT += 'predictions/'
except:
    PATH_OUT += 'predictions/'


# Get prediction and accuracy score
if(TTA_WRAP):
    tta_model = TTA_ModelWrapper(model)
    predictions = tta_model.predict(X)
else:
    predictions = model.predict(X, verbose=1).squeeze()

if(EVAL):
    resume_metrics = model.evaluate(X, y, verbose=1)
    msg = '\nAccuracy Score:\n'
    for i, metric in enumerate(resume_metrics):
        msg += ' %s: %.3f   ' %(model.metrics_names[i], metric) 
    print(msg+'\n')


# Plot matthews_corrcoef
plot_phicoef(y, predictions, idxs, redshift, xn, path=PATH_OUT)


for i in IDX:
    true, pred = y[i].squeeze(), predictions[i]
    """
    mfp_pred = t2c.bubble_stats.mfp(pred, xth=0.5, boxsize=IM_SHAPE[0], iterations=2000000, verbose=False,
                                    upper_lim=False, bins=None, r_min=None, r_max=None)

    mfp_true = t2c.bubble_stats.mfp(true, xth=0.5, boxsize=IM_SHAPE[0], iterations=2000000, verbose=False,
                                    upper_lim=False, bins=None, r_min=None, r_max=None)  

    compare = 100*(mfp_pred[1]/mfp_true[1] - 1)

    fig = plt.figure(figsize=[12,8])
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1.8]) # set height ratios for sublots
    # THE FIRST SUBPLOT
    ax0 = plt.subplot(gs[0])
    ax0.set_title('$z=%.3f$\t$x_n=%.3f$\t$r_{\phi}=%.3f$' %(redshift[i], xn[i], phi_coef[i]))
    ax0.semilogx(mfp_pred[0], mfp_pred[1], '-', label='recovered', lw=2)
    ax0.semilogx(mfp_true[0], mfp_true[1], 'k--', label='true', lw=2)
    ax0.legend(loc=0, borderpad=0.5)
    ax0.tick_params(axis='both', length=10, width=2, labelsize=15)
    ax0.set_ylabel('RdP/dR', size=15)
    # THE SECOND SUBPLOT
    ax1 = plt.subplot(gs[1], sharex = ax0)
    ax1.semilogx(mfp_true[0], compare, '-', lw=2)
    ax1.semilogx(mfp_true[0], np.zeros_like(mfp_true[0]), 'k--', lw=2)
    ax1.tick_params(axis='both', length=10, width=2, labelsize=15)
    ax1.set_ylabel('difference (%)', size=15)
    ax1.set_xlabel('R (Mpc)', size=12)
    # PLOT SETUP
    plt.setp(ax0.get_xticklabels(), visible=False) # remove x label from fist plot
    plt.subplots_adjust(hspace=0.05)   # remove vertical gap between subplots
    plt.savefig(PATH_OUT+'bs_comparison_%d.png' %i, bbox_inches='tight')
    plt.close('all')


    # Slice plot
    plot_sample3D(X, y, predictions, idx=i, path=PATH_OUT)


    # Power spectra plot
    ps_true, ks_true, n_modes_true = t2c.power_spectrum_1d(true, kbins=20, box_dims=256,
                                            return_n_modes=True, binning='log')

    ps_pred, ks_pred, n_modes_pred = t2c.power_spectrum_1d(pred, kbins=20, box_dims=256,
                                            return_n_modes=True, binning='log')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('z=%.3f   $x_n$=%.2f   $\zeta$=%.3f   $R_{mfp}$=%.3f cMpc   $log(T_{vir}^{min})$=%.3f' 
                 %(redshift[i], xn[i], eff_fact[i], Rmfp[i], Tvir[i]), fontsize=18)
    ax.loglog(ks_pred, ps_pred*ks_pred**3/2/np.pi**2, 'r-')
    ax.loglog(ks_true, ps_true*ks_true**3/2/np.pi**2, 'b-')

    ax.set_xlabel('k (Mpc$^{-1}$)', fontsize=16), plt.ylabel('$\Delta^2_{21}$', fontsize=16)
    ax.tick_params(axis='both', length=10, width=1.2, labelsize=14)
    ax.tick_params(which='minor', axis='both', length=5, width=1.2, labelsize=14)
    plt.savefig(PATH_OUT+'Pk_i%d.png' %i, bbox_inches='tight')
    plt.close('all')
    """
    save_cbin(PATH_OUT+'pred_dT_i%d.bin' %i, pred)
    save_cbin(PATH_OUT+'true_dT_i%d.bin' %i, true)

