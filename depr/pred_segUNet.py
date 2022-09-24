import numpy as np, matplotlib.pyplot as plt, os, json
import tools21cm as t2c
import random, zipfile
from tqdm import tqdm
import matplotlib.gridspec as gridspec

from sklearn.metrics import matthews_corrcoef, r2_score

from tensorflow.keras.models import load_model
from utils_network.metrics import get_avail_metris
from config.net_config import NetworkConfig
from utils_pred.prediction import SegUnet21cmPredict
from utils_network.data_generator import LightConeGenerator
from utils_plot.other_utils import adjust_axis

def LoadSegUnetModel(cfile):
    conf = NetworkConfig(cfile)
    
    path_out = conf.RESUME_PATH
    MODEL_EPOCH = conf.BEST_EPOCH
    METRICS = {m:get_avail_metris(m) for m in np.append(conf.LOSS, conf.METRICS)}
    model_loaded = load_model('%smodel-sem21cm_ep%d.tf' %(path_out+'checkpoints/', MODEL_EPOCH), custom_objects=METRICS)
    
    print(' Loaded model:\n %smodel-sem21cm_ep%d.tf' %(conf.RESUME_PATH, MODEL_EPOCH))
    return model_loaded

title_a = '\t\t _    _ _   _      _   \n\t\t| |  | | \ | |    | |  \n\t\t| |  | |  \| | ___| |_ \n\t\t| |  | | . ` |/ _ \ __|\n\t\t| |__| | |\  |  __/ |_ \n\t\t \____/|_| \_|\___|\__|\n'
title_b = ' _____              _ _      _         ___  __                \n|  __ \            | (_)    | |       |__ \/_ |               \n| |__) | __ ___  __| |_  ___| |_ ___     ) || | ___ _ __ ___  \n|  ___/ `__/ _ \/ _` | |/ __| __/ __|   / / | |/ __| `_ ` _ \ \n| |   | | |  __/ (_| | | (__| |_\__ \  / /_ | | (__| | | | | |\n|_|   |_|  \___|\__,_|_|\___|\__|___/ |____||_|\___|_| |_| |_|\n'
print(title_a+'\n'+title_b)

MAKE_PLOT = False
tobs = 1000

path_pred = '/store/ska/sk09/segunet/inputs/dataLC_128_pred_310822/'
path_in = '/scratch/snx3000/mibianco/output_segunet/outputs/16-09T14-15-20_128slice/'
#path_in = '/scratch/snx3000/mibianco/output_segunet/outputs/14-09T13-23-19_128slice/'
#path_in = '/scratch/snx3000/mibianco/output_segunet/outputs/dT4pca_12-09T16-07-57_128slice/'
#path_in = '/scratch/snx3000/mibianco/output_segunet/outputs/dT3_12-09T15-23-31_128slice/'
config_file = path_in+'net_Unet_lc.ini'
#path_in = '/jmain02/home/J2AD005/jck02/mxb47-jck02/data/outputs/24-02T09-32-28_10cube/'
#path_in = '/jmain02/home/J2AD005/jck02/mxb47-jck02/data/outputs/28-03T10-58-31_10cube/'
#config_file = path_in+'net_LSTMUnet_lc.ini'

path_out = path_in+'prediction/'
try:
    os.makedirs(path_out)
except:
    pass
    
#dT3 = t2c.read_cbin('%sdata/dT3_21cm_i0.bin' %path_pred)
dT3 = t2c.read_cbin('%sdata/dT4pca_21cm_i100.bin' %path_pred)
xH = t2c.read_cbin('%sdata/xH_21cm_i100.bin' %path_pred)
idx, zeta, Rmfp, Tvir, rseed = np.loadtxt('%sparameters/astro_params.txt' %path_pred, unpack=True)
redshift = np.loadtxt('%slc_redshifts.txt' %path_pred)

#dg = LightConeGenerator(path=path_pred, data_temp=np.arange(1), batch_size=1, data_shape=(10, 128, 128))
#X, Y = dg.__getitem__(0)

X = np.moveaxis(dT3, -1, 0)
Y = np.moveaxis(xH, -1, 0)

# Load & predict with model
model = LoadSegUnetModel(config_file)
X_seg = model.predict(X, verbose=0)

print(X_seg.min(), X_seg.max())
X_seg = X_seg.squeeze()
X_seg = np.round(np.clip(X_seg, 0, 1))

############## for LSTM
#for i in range(X_seg.shape[0]):
#    for j in range(X_seg.shape[1]):
#        if(i == 0 and j == 0):
#            tmp_X_seg = X_seg[i,j][np.newaxis, ...]
#        else:
#            tmp_X_seg = np.vstack((tmp_X_seg, X_seg[i,j][np.newaxis,...]))

#for k in range(0,redshift.size - tmp_X_seg.shape[0]):
#    tmp_X_seg = np.vstack((tmp_X_seg, np.zeros(tmp_X_seg.shape[1:])[np.newaxis,...]))

#X_seg = tmp_X_seg
#X = np.moveaxis(dT3, -1, 0)
#######################
print(X_seg.shape)

# Cosmology and Astrophysical parameters
with open(path_pred+'parameters/user_params.txt', 'r') as file:
    params = eval(file.read())
my_ext1 = [redshift.min(), redshift.max(), 0, params['BOX_LEN']]
my_ext2 = [0, params['BOX_LEN'], 0, params['BOX_LEN']]

a_params = {'HII_EFF_FACTOR':zeta, 'R_BUBBLE_MAX':Rmfp, 'ION_Tvir_MIN':Tvir}

phicoef_seg = np.zeros_like(redshift)
phicoef_err = np.zeros_like(phicoef_seg)
phicoef_sp = np.zeros_like(phicoef_seg)
r2score_seg = np.zeros_like(redshift)
xn_mask = np.zeros_like(phicoef_seg)
xn_seg = np.zeros_like(phicoef_seg)
xn_err = np.zeros_like(phicoef_seg)
xn_sp = np.zeros_like(phicoef_sp)
b0_true = np.zeros_like(phicoef_sp)
b1_true = np.zeros_like(phicoef_sp)
b2_true = np.zeros_like(phicoef_sp)
b0_sp = np.zeros_like(phicoef_sp)
b1_sp = np.zeros_like(phicoef_sp)
b2_sp = np.zeros_like(phicoef_sp)
b0_seg = np.zeros_like(phicoef_sp)
b1_seg = np.zeros_like(phicoef_sp)
b2_seg = np.zeros_like(phicoef_sp)

acc = np.zeros_like(phicoef_seg)
prec = np.zeros_like(phicoef_seg)
rec = np.zeros_like(phicoef_seg)
iou = np.zeros_like(phicoef_seg)

for i_z in tqdm(range(X_seg.shape[0])):    
    x = np.array(X_seg[i_z].flatten(), dtype=bool)
    m = np.array(Y[i_z].flatten(), dtype=bool)

    TP = np.sum((1-x)*(1-m))
    TN = np.sum(x*m)
    FP = np.sum((1-x)*m)
    FN = np.sum(x*(1-m))

    acc[i_z] = (TP+TN)/(TP+TN+FP+FN)
    prec[i_z] = TP/(TP+FP)
    rec[i_z] = TP/(TP+FN)
    iou[i_z] = TP/(TP+FP+FN)
    xn_seg[i_z] = np.mean(X_seg[i_z])
    xn_mask[i_z] = np.mean(Y[i_z])
    phicoef_seg[i_z] = matthews_corrcoef(m, x)

# PLOT MATTHEWS CORRELATION COEF
fig = plt.figure(figsize=(10, 8))
plt.plot(redshift, phicoef_seg, '-', label='PhiCoef')
plt.xlabel('z'), plt.ylabel(r'$r_{\phi}$')
plt.legend()
plt.savefig('%sphi_coef.png' %(path_out), bbox_inches='tight'), plt.clf()

# PLOT STATS
fig = plt.figure(figsize=(10, 8))
plt.plot(redshift, acc, label='Accuracy', color='tab:blue')
plt.plot(redshift, prec, label='Precision', color='tab:orange')
plt.plot(redshift, rec, label='Recall', color='tab:green')
plt.plot(redshift, iou, label='IoU', color='tab:red')
plt.xlabel('z'), plt.ylabel('%')
plt.legend()
plt.savefig('%sstats.png' %(path_out), bbox_inches='tight'), plt.clf()

# PLOTS AVERGE MASK HI
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1.8])

# Main plot
ax0 = plt.subplot(gs[0])
ax0.plot(redshift, xn_seg, ls='-', color='tab:orange', label='Prediction', lw=1.5)
ax0.plot(redshift, xn_mask, ls='-', color='tab:blue', label='True', lw=1.5)
#ax0.fill_between(z_mean, avrgR_mean_under/b*a, avrgR_mean_over/b*a, color='lightcoral', alpha=0.2)
ax0.legend()
ax0.set_ylabel(r'$x_{HI}$')

# THE SECORD SUBPLOT
ax1 = plt.subplot(gs[1], sharex = ax0)
perc_diff = xn_mask/xn_seg-1
ax1.plot(redshift, perc_diff, 'k-', lw=1.5)
ax1.set_ylabel('difference (%)')
ax1.set_xlabel('$z$')
#ax1.fill_between(z_quad, diff_s_avrgR_under, diff_s_avrgR_over, color='lightgreen', alpha=0.1)
ax1.axhline(y=0,  color='black', ls='dashed')
plt.setp(ax0.get_xticklabels(), visible=False)
plt.subplots_adjust(hspace=.0)
plt.savefig('%spred_xn.png' %(path_out), bbox_inches='tight'), plt.clf()

# Visual Plot
i_slice = np.argmin(abs(xn_mask - 0.5))
i_lc = params['HII_DIM']//2

plt.rcParams['font.size'] = 20
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.linewidth'] = 1.2

fig = plt.figure(figsize=(35, 15))
gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[2,1], height_ratios=[1,1])

# FIRST LC PLOT
ax0 = fig.add_subplot(gs[0,0])
ax0.set_title('$r_{\phi}=%.3f$ $t_{obs}=%d\,h$' %(phicoef_seg[i_slice], 1000), fontsize=20)
im = ax0.imshow(X[:,i_lc,:].T, cmap='jet', aspect='auto', origin='lower', extent=my_ext1)
ax0.contour(Y[:,i_lc,:].T, extent=my_ext1)
#adjust_axis(varr=redshift, xy='x', axis=ax0, to_round=10, step=0.25)
#adjust_axis(varr=np.linspace(0, 1.6402513488058277, 100), xy='y', axis=ax0, to_round=10, step=0.5)
ax0.set_ylabel('y [Mpc]', size=20)
ax0.set_xlabel('z', size=20)

# FIRST SLICE PLOT
ax01 = fig.add_subplot(gs[0,1])
ax01.set_title(r'$z$ = %.3f   $x_{HI}=%.2f$' %(redshift[i_slice], xn_mask[i_slice]), fontsize=20)
ax01.imshow(X[i_slice,...], cmap='jet', extent=my_ext2, origin='lower')
ax01.contour(Y[i_slice,...], extent=my_ext2)
#fig.colorbar(im, label=r'$\delta T_b$ [mK]', ax=ax01, pad=0.01, fraction=0.048)

# SECOND LC PLOT
ax1 = fig.add_subplot(gs[1,0])
ax1.imshow(X_seg[:,i_lc,:].T, cmap='jet', aspect='auto', origin='lower', extent=my_ext1, vmin=X_seg.min(), vmax=X_seg.max())
ax1.contour(Y[:,i_lc,:].T, extent=my_ext1)

# SECOND SLICE PLOT
ax11 = fig.add_subplot(gs[1,1])
ax11.set_title(r'$r_{\phi}$ = %.3f' %(phicoef_seg[i_slice]), fontsize=20)
im = ax11.imshow(X_seg[i_slice,...], cmap='jet', extent=my_ext2, origin='lower', vmin=X_seg.min(), vmax=X_seg.max())
ax11.contour(Y[i_slice,...], extent=my_ext2)
ax1.set_ylabel('y [Mpc]', size=20)
ax1.set_xlabel('z', size=20)

for ax in [ax01, ax11]:
    ax.set_ylabel('y [Mpc]', size=20)
    ax.set_xlabel('x [Mpc]', size=20)

plt.subplots_adjust(hspace=0.3, wspace=0.01)
plt.savefig('%svisual_comparison_lc.png' %path_out, bbox_inches='tight')

"""
#for i_z in tqdm(range(redshift.size)):
for i_z in tqdm(range(X.shape[0]):
    z = redshift[i_z]

    if(MAKE_PLOT):
        plt.rcParams['font.size'] = 20
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        fig, axs = plt.subplots(2, 2, figsize=(12,12))
        for ax in axs.flat: ax.label_outer()
        fig.suptitle('z = %.3f\t\t$x^v_{HI}$ = %.2f\n$\zeta$ = %.3f        $R_{mfp}$ = %.3f Mpc        $log_{10}(T_{vir}^{min})$ = %.3f' %(z, np.mean(xH), zeta, Rmfp, Tvir), fontsize=18)
        plt.rcParams['font.size'] = 16
        axs[0,0].set_title('$x_{HI}$', size=18)
        axs[0,0].contour(mask_xn[:,:,i_lc], colors='lime', levels=[0.5], extent=my_ext)
        im = axs[0,0].imshow(xH[:,:,i_lc], origin='lower', cmap='jet', extent=my_ext)
        fig.colorbar(im, ax=axs[0,0], pad=0.01, fraction=0.048)
        axs[0,0].set_ylabel('y [Mpc]')
        axs[0,1].set_title('$\delta T^{noise}_b(t_{obs}=%d\,h)$' %tobs, size=18)
        im = axs[0,1].imshow(dT2[:,:,i_lc], origin='lower', cmap='jet', extent=my_ext)
        fig.colorbar(im, ax=axs[0,1], pad=0.01, fraction=0.048)
        axs[1,0].set_title('$\delta T^{sim}_b$', size=18)
        im = axs[1,0].imshow(dT3[:,:,i_lc], origin='lower', cmap='jet', extent=my_ext)
        axs[1,0].set_xlabel('x [Mpc]'), axs[1,0].set_ylabel('y [Mpc]')
        fig.colorbar(im, ax=axs[1,0], pad=0.01, fraction=0.048)
        axs[1,1].set_title('$\delta T^{obs}_b(B=2\,km)$', size=18)
        im = axs[1,1].imshow(dT3[:,:,i_lc], origin='lower', cmap='jet', extent=my_ext)
        axs[1,1].contour(mask_xn[:,:,i_lc], colors='k', levels=[0.5], extent=my_ext)
        axs[1,1].set_xlabel('x [Mpc]')
        fig.colorbar(im, ax=axs[1,1], pad=0.02, fraction=0.048)
        plt.subplots_adjust(hspace=0.01, wspace=0.12)
        plt.savefig('%scube21cm_i%d.png' %(path_out+'plots/', i_z), bbox_inches='tight'), plt.clf()

    # SegU-net
    #X_tta = SegUnet21cmPredict(unet=model, x=dT3, TTA=False)
    #X_seg = np.round(np.mean(X_tta, axis=0))

    x = np.array(1-X_seg.flatten(), dtype=bool)
    m = np.array(1-mask_xn.flatten(), dtype=bool)
    TP = np.sum(x*m)
    TN = np.sum((1-x)*(1-m))
    FP = np.sum(x*(1-m))
    FN = np.sum((1-x)*m)
    acc[i_z] = (TP+TN)/(TP+TN+FP+FN)
    prec[i_z] = TP/(TP+FP)
    rec[i_z] = TP/(TP+FN)
    iou[i_z] = TP/(TP+FP+FN)
    xn_seg[i_z] = np.mean(X_seg)

    #new_astr_data = np.vstack((redshift, acc))
    new_astr_data = np.vstack((acc, prec))
    new_astr_data = np.vstack((new_astr_data, rec))
    new_astr_data = np.vstack((new_astr_data, iou))
    new_astr_data = np.vstack((new_astr_data, xn_mask))
    new_astr_data = np.vstack((new_astr_data, xn_seg))
    np.savetxt('%sastro_data_FM_tobs%d_stats.txt' %(path_out, tobs), new_astr_data.T, fmt='%.4f\t'*7, header='tobs = %d, eff_f = %.3f, Rmfp = %.3f, Tvir = %.3f\nz\tacc\tprec\trec\tiou\txn_mask xn_seg' %(tobs, zeta, Rmfp, Tvir))
    

    '''
    X_seg_err = np.std(X_tta, axis=0)

    phicoef_tta = np.zeros(X_tta.shape[0])
    xn_tta = np.zeros(X_tta.shape[0])
    for i in tqdm(range(len(X_tta))):
        x = X_tta[i]
        xn_tta[i] = np.mean(np.round(x))
        phicoef_tta[i] = matthews_corrcoef(mask_xn.flatten(), np.round(x).flatten())

    # super-pixel
    labels = t2c.slic_cube(dT3.astype(dtype='float64'), n_segments=7000, compactness=0.1, max_iter=20, sigma=0, min_size_factor=0.5, max_size_factor=3, cmap=None)
    superpixel_map = t2c.superpixel_map(dT3, labels)
    X_sp = 1-t2c.stitch_superpixels(dT3, labels, bins='knuth', binary=True, on_superpixel_map=True)
    
    # calculate data
    phicoef_seg[i_z] = matthews_corrcoef(mask_xn.flatten(), X_seg.flatten())
    phicoef_err[i_z] = np.std(phicoef_tta)
    phicoef_sp[i_z] = matthews_corrcoef(mask_xn.flatten(), X_sp.flatten())
    xn_seg[i_z] = np.mean(X_seg)
    xn_err[i_z] = np.std(xn_tta)
    xn_sp[i_z] = np.mean(X_sp)
    b0_true[i_z] = t2c.betti0(data=mask_xn)
    b1_true[i_z] = t2c.betti1(data=mask_xn)
    b2_true[i_z] = t2c.betti2(data=mask_xn)
    b0_seg[i_z] = t2c.betti0(data=X_seg)
    b1_seg[i_z] = t2c.betti1(data=X_seg)
    b2_seg[i_z] = t2c.betti2(data=X_seg)
    b0_sp[i_z] = t2c.betti0(data=X_sp)
    b1_sp[i_z] = t2c.betti1(data=X_sp)
    b2_sp[i_z] = t2c.betti2(data=X_sp)

    if(MAKE_PLOT):
        plt.rcParams['font.size'] = 20
        plt.rcParams['xtick.top'] = False
        plt.rcParams['ytick.right'] = False
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['xtick.direction'] = 'out'
        plt.rcParams['ytick.direction'] = 'out'
        plt.rcParams['figure.figsize'] = [20, 10]
        ls = 22

        idx = i_lc

        # Plot visual comparison
        fig, axs = plt.subplots(figsize=(20,10), ncols=3, sharey=True, sharex=True)
        (ax0, ax1, ax2) = axs
        ax0.set_title('Super-Pixel ($r_{\phi}=%.3f$)' %phicoef_sp[i_z], size=ls)
        ax0.imshow(X_sp[:,:,idx], origin='lower', cmap='jet', extent=my_ext)
        ax0.contour(mask_xn[:,:,idx], colors='lime', levels=[0.5], extent=my_ext)
        ax0.set_xlabel('x [Mpc]'), ax0.set_ylabel('y [Mpc]')
        ax1.set_title('SegU-Net ($r_{\phi}=%.3f$)' %phicoef_seg[i_z], size=ls)
        ax1.imshow(X_seg[:,:,idx], origin='lower', cmap='jet', extent=my_ext)
        ax1.contour(mask_xn[:,:,idx], colors='lime', levels=[0.5], extent=my_ext)
        ax1.set_xlabel('x [Mpc]')
        ax2.set_title('SegUNet Pixel-Error', size=ls)
        im = plt.imshow(X_seg_err[:,:,idx], origin='lower', cmap='jet', extent=my_ext)
        fig.colorbar(im, label=r'$\sigma_{std}$', ax=ax2, pad=0.02, cax=fig.add_axes([0.905, 0.25, 0.02, 0.51]))
        ax2.set_xlabel('x [Mpc]')
        plt.subplots_adjust(hspace=0.1, wspace=0.01)
        for ax in axs.flat: ax.label_outer()
        plt.savefig('%svisual_comparison_i%d.png' %(path_out+'plots/', i_z), bbox_inches='tight'), plt.clf()

        # Plot BSD-MFP of the prediction
        mfp_pred_ml = t2c.bubble_stats.mfp(X_seg, xth=0.5, boxsize=params['BOX_LEN'], iterations=2000000, verbose=False, upper_lim=False, bins=None, r_min=None, r_max=None)
        mfp_pred_sp = t2c.bubble_stats.mfp(X_sp, xth=0.5, boxsize=params['BOX_LEN'], iterations=2000000, verbose=False, upper_lim=False, bins=None, r_min=None, r_max=None)
        mfp_true = t2c.bubble_stats.mfp(mask_xn, xth=0.5, boxsize=params['BOX_LEN'], iterations=2000000, verbose=False, upper_lim=False, bins=None, r_min=None, r_max=None)  

        mfp_tta = np.zeros((X_tta.shape[0], 2, 128))
        for j in tqdm(range(0, X_tta.shape[0])):
            mfp_pred_ml1, mfp_pred_ml2 = t2c.bubble_stats.mfp(np.round(X_tta[j]), xth=0.5, boxsize=params['BOX_LEN'], iterations=2000000, verbose=False, upper_lim=False, bins=None, r_min=None, r_max=None)
            mfp_tta[j,0] = mfp_pred_ml1
            mfp_tta[j,1] = mfp_pred_ml2

        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['font.size'] = 20

        compare_ml = (mfp_pred_ml[1]/mfp_true[1])
        compare_ml_tta = (mfp_tta[:,1,:]/mfp_true[1])
        compare_sp = (mfp_pred_sp[1]/mfp_true[1])

        fig, ax0 = plt.subplots(figsize=(12, 9))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1.8]) # set height ratios for sublots
        ax0 = plt.subplot(gs[0])
        ax0.set_title('$z=%.3f$\t$x_n=%.3f$\t$r_{\phi}=%.3f$' %(z, xn_mask[i_z], phicoef_seg[i_z]), fontsize=ls)
        ax0.fill_between(mfp_pred_ml[0], np.min(mfp_tta[:,1,:], axis=0), np.max(mfp_tta[:,1,:], axis=0), color='tab:blue', alpha=0.2)
        ax0.loglog(mfp_pred_ml[0], mfp_pred_ml[1], '-', color='tab:blue', label='SegUNet', lw=2)
        ax0.loglog(mfp_pred_sp[0], mfp_pred_sp[1], '-', color='tab:orange', label='Super-Pixel', lw=2)
        ax0.loglog(mfp_true[0], mfp_true[1], 'k--', label='Ground true', lw=2)
        ax0.legend(loc=0, borderpad=0.5)
        ax0.tick_params(axis='both', length=7, width=1.2)
        ax0.tick_params(axis='both', which='minor', length=5, width=1.2)
        ax0.set_ylabel('RdP/dR', size=18), ax0.set_xlabel('R (Mpc)')
        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.loglog(mfp_true[0], compare_ml, '-', lw=2)
        ax1.loglog(mfp_true[0], compare_sp, '-', lw=2)
        ax1.loglog(mfp_true[0], np.ones_like(mfp_true[0]), 'k--', lw=2)
        ax1.fill_between(mfp_true[0], np.min(compare_ml_tta, axis=0), np.max(compare_ml_tta, axis=0), color='tab:blue', alpha=0.2)
        ax1.tick_params(axis='both', length=7, width=1.2, labelsize=15)
        ax1.set_ylabel('difference (%)', size=15)
        ax1.set_xlabel('R (Mpc)', size=18)
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=0.0)
        ax1.tick_params(which='minor', axis='both', length=5, width=1.2)
        plt.savefig('%sbs_comparison_i%d.png' %(path_out+'plots/', i_z), bbox_inches='tight'), plt.clf()

        # Plot dimensioneless power spectra of the x field
        plt.rcParams['font.size'] = 20

        ps_true, ks_true = t2c.power_spectrum_1d(mask_xn, kbins=20, box_dims=256, binning='log')
        ps_pred_sp, ks_pred_sp = t2c.power_spectrum_1d(X_sp, kbins=20, box_dims=256, binning='log')
        ps_pred_ml, ks_pred_ml = t2c.power_spectrum_1d(X_seg, kbins=20, box_dims=256, binning='log')

        ps_tta = np.zeros((X_tta.shape[0],20))
        for k in range(0,X_tta.shape[0]):
            ps_tta[k], ks_pred_ml = t2c.power_spectrum_1d(np.round(X_tta[k]), kbins=20, box_dims=256, binning='log')

        compare_ml = 100*(ps_pred_ml/ps_true - 1.)
        compare_ml_tta = 100*(ps_tta/ps_true - 1.)
        compare_sp = 100*(ps_pred_sp/ps_true - 1.)
        
        fig, ax = plt.subplots(figsize=(12, 9))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1.8])
        ax0 = plt.subplot(gs[0])
        ax0.set_title('$z=%.3f$\t$x_n=%.3f$\t$r_{\phi}=%.3f$' %(z, xn_mask[i_z], phicoef_seg[i_z]), fontsize=ls)
        ax0.fill_between(ks_pred_ml, np.min(ps_tta*ks_pred_ml**3/2/np.pi**2, axis=0), np.max(ps_tta*ks_pred_ml**3/2/np.pi**2, axis=0), color='tab:blue', alpha=0.2)
        ax0.loglog(ks_pred_ml, ps_pred_ml*ks_pred_ml**3/2/np.pi**2, '-', color='tab:blue', label='SegUNet', lw=2)
        ax0.loglog(ks_pred_sp, ps_pred_sp*ks_pred_sp**3/2/np.pi**2, '-', color='tab:orange', label='Super-Pixel', lw=2)
        ax0.loglog(ks_true, ps_true*ks_true**3/2/np.pi**2, 'k--', label='Ground true', lw=2)
        ax0.set_yscale('log')
        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.semilogx(ks_true, compare_ml, '-', lw=2)
        ax1.semilogx(ks_true, compare_sp, '-', lw=2)
        ax1.semilogx(ks_true, np.zeros_like(ks_true), 'k--', lw=2)
        ax1.fill_between(ks_true, np.min(compare_ml_tta, axis=0), np.max(compare_ml_tta, axis=0), color='tab:blue', alpha=0.2)
        ax1.tick_params(axis='both', length=7, width=1.2, labelsize=15)
        ax1.set_xlabel('k (Mpc$^{-1}$)'), ax0.set_ylabel('$\Delta^2_{xx}$')
        ax1.set_ylabel('difference (%)', size=15)
        ax0.tick_params(axis='both', length=10, width=1.2)
        ax0.tick_params(which='minor', axis='both', length=5, width=1.2)
        ax1.tick_params(which='minor', axis='both', length=5, width=1.2)
        ax0.legend(loc=0, borderpad=0.5)
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=0.0)
        plt.savefig('%sPk_comparison_i%d.png' %(path_out+'plots/', i_z), bbox_inches='tight'), plt.clf()

        ds_data = np.vstack((ks_true, np.vstack((ps_true*ks_true**3/2/np.pi**2, np.vstack((np.vstack((ps_pred_ml*ks_pred_ml**3/2/np.pi**2, np.vstack((np.min(ps_tta*ks_pred_ml**3/2/np.pi**2, axis=0), np.max(ps_tta*ks_pred_ml**3/2/np.pi**2, axis=0))))), ps_pred_sp*ks_pred_sp**3/2/np.pi**2))))))
        bsd_data = np.vstack((mfp_true[0], np.vstack((mfp_true[1], np.vstack((np.vstack((mfp_pred_ml[1], np.vstack((np.min(mfp_tta[:,1,:], axis=0), np.max(mfp_tta[:,1,:], axis=0))))), mfp_pred_sp[1]))))))

        np.savetxt('%sds_data_i%d.txt' %(path_out+'data/', i_z), ds_data.T, fmt='%.6e', delimiter='\t', header='k [Mpc^-1]\tds_true\tds_seg_mean\tds_err_min\tds_err_max\tds_sp')
        np.savetxt('%sbsd_data_i%d.txt' %(path_out+'data/', i_z), bsd_data.T, fmt='%.6e', delimiter='\t', header='R [Mpc]\tbs_true\tbs_seg_mean\tb_err_min\tbs_err_max\tbs_sp')

    # Save data
    new_astr_data = np.vstack((redshift, phicoef_seg))
    new_astr_data = np.vstack((new_astr_data, phicoef_err))
    new_astr_data = np.vstack((new_astr_data, phicoef_sp))
    new_astr_data = np.vstack((new_astr_data, xn_mask))
    new_astr_data = np.vstack((new_astr_data, xn_seg))
    new_astr_data = np.vstack((new_astr_data, xn_err))
    new_astr_data = np.vstack((new_astr_data, xn_sp))
    new_astr_data = np.vstack((new_astr_data, b0_true))
    new_astr_data = np.vstack((new_astr_data, b1_true))
    new_astr_data = np.vstack((new_astr_data, b2_true))
    new_astr_data = np.vstack((new_astr_data, b0_seg))
    new_astr_data = np.vstack((new_astr_data, b1_seg))
    new_astr_data = np.vstack((new_astr_data, b2_seg))
    new_astr_data = np.vstack((new_astr_data, b0_sp))
    new_astr_data = np.vstack((new_astr_data, b1_sp))
    new_astr_data = np.vstack((new_astr_data, b2_sp))
    np.savetxt('%sastro_data_FM_tobs%d.txt' %(path_out, tobs), new_astr_data.T, fmt='%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d', header='tobs = %d, eff_f = %.3f, Rmfp = %.3f, Tvir = %.3f\nz\tphi_ML\tphi_err phi_SP\txn_mask xn_seg\txn_err\txn_sp\tb0 true b1\tb2\tb0 ML\tb1\tb2\tb0 SP\tb1\tb2' %(tobs, zeta, Rmfp, Tvir))
    '''
"""
print('... done.')