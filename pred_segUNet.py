import os, sys, numpy as np, matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

import tools21cm as t2c

from matplotlib import gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from datetime import datetime
from tqdm import tqdm

from keras.optimizers import Adam
from keras.models import load_model

from utils_network.networks import Unet
from utils_network.metrics import iou, iou_loss, dice_coef, dice_coef_loss
from utils_network.data_generator import TTA_ModelWrapper
from utils.other_utils import get_data, save_cbin
from utils_plot.plotting import plot_sample, plot_sample3D

IM_SHAPE = (128, 128, 128)

#PATH_PRED = '/ichec/work/subgridEoRevol/michele/data3D_64_130320_valid/'
#PATH_PRED = '/ichec/work/subgridEoRevol/michele/data3D_128_180320_valid2/'
#PATH_PRED = '/ichec/work/subgridEoRevol/michele/data3D_128_050520_tobs900/'
PATH_PRED = '/ichec/work/subgridEoRevol/michele/data3D_128_050520_tobs1200/'
#PATH_PRED =['/ichec/work/subgridEoRevol/michele/data3D_128_180320_valid/', 
#            '/ichec/work/subgridEoRevol/michele/data3D_128_180320_valid2/']

PATH_OUT = '/ichec/work/subgridEoRevol/michele/output_SegNet/20-04T21-21-37_128cube/'
#PATH_OUT = '/ichec/work/subgridEoRevol/michele/output_SegNet/19-03T20-09-24_128cube_acc95/'
#PATH_OUT = '/ichec/work/subgridEoRevol/michele/output_SegNet/26-03T17-58-08_128cube_acc98/'

# Load data
if isinstance(PATH_PRED, (list, np.ndarray)):
    for j, path in enumerate(PATH_PRED):
        if(j == 0):
            print('Load dataset path %d ...' %(j+1)) 
            X, y = get_data(path+'data/', IM_SHAPE)

            # get parameters of dataset
            idxs, redshift, eff_fact, Rmfp, Tvir, xn = np.loadtxt(path+'astro_params.txt', unpack=True)
        else:
            print('Load images path %d ...' %(j+1))
            X_next, y_next = get_data(path+'data/', IM_SHAPE)
            X, y = np.vstack((X, X_next)), np.vstack((y,y_next))

            # get parameters of dataset
            idxs_next, redshift_next, eff_fact_next, Rmfp_next, Tvir_next, xn_next = np.loadtxt(path+'astro_params.txt', unpack=True)
            idxs = np.hstack((idxs, idxs_next))
            redshift = np.hstack((redshift, redshift_next))
            eff_fact = np.hstack((eff_fact, eff_fact_next))
            Rmfp = np.hstack((Rmfp, Rmfp_next))
            Tvir = np.hstack((Tvir, Tvir_next))
            xn = np.hstack((xn, xn_next))
else:
    print('Load dataset ...') 
    # get parameters of dataset
    idxs, redshift, eff_fact, Rmfp, Tvir, xn = np.loadtxt(PATH_PRED+'astro_params.txt', unpack=True)

    # Get data
    X, y = get_data(PATH_PRED+'data/', IM_SHAPE)

# Define model
model = load_model(PATH_OUT+'checkpoints/model-sem21cm_ep3.h5', custom_objects={'iou_loss': iou_loss, 'iou': iou,'dice_coef': dice_coef, 'dice_coef_loss':dice_coef_loss})
#model = load_model(PATH_OUT+'model-sem21cm_ep86.h5')

# Create predictions output directory
try:
    os.makedirs(PATH_OUT+'predictions')
    PATH_OUT += 'predictions/'
except:
    PATH_OUT += 'predictions/'

# Get prediction and accuracy score
#tta_model = TTA_ModelWrapper(model)
#predictions = tta_model.predict(X)
predictions = model.predict(X, verbose=1).squeeze()
#score = model.evaluate(X, y, verbose=1)
#print("Accuracy score %s: %.2f%%" % (model.metrics_names[1], score[1]*100))


# Plot matthews_corrcoef
phi_coef = np.zeros_like(idxs)
for i in tqdm(range(idxs.size)):
    phi_coef[i] = matthews_corrcoef(y[i].flatten(), predictions[i].flatten().round())
plt.xlabel('$x_n$'), plt.ylabel('$r_{\phi}$')
plt.plot(xn, phi_coef, 'r.')
np.savetxt(PATH_OUT+'phi_coef.txt', np.array([idxs, redshift, xn, phi_coef]).T, fmt='%d\t%.3f\t%.3f\t%.3f', header='z\tx_n\tphi_coef')
plt.savefig(PATH_OUT+'phi_coef.png', bbox_inches='tight')
plt.clf()


#idx = [114, 45, 210, 178]
#idx = [114, 178, 298, 45]
#idx = [10, 4, 6]
#idx = [3,5,2]
idx = range(18)
# Bubble size distribution Plot
for i in idx:
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

# Slice plot
#plot_sample3D(X, y, predictions, idx=idx[1], path=PATH_OUT)
#plt.close()
