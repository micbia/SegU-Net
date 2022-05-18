import numpy as np, matplotlib.pyplot as plt, os, json
import tools21cm as t2c
import random, zipfile
from tqdm import tqdm
import matplotlib.gridspec as gridspec

from sklearn.metrics import matthews_corrcoef

from tensorflow.keras.models import load_model
from utils_network.metrics import get_avail_metris
from config.net_config import NetworkConfig
from utils_network.prediction import SegUnet21cmPredict
from utils_network.data_generator import LightConeGenerator
from utils_plot.other_utils import adjust_axis

def LoadSegUnetModel(cfile):
    conf = NetworkConfig(cfile)
    
    path_out = conf.resume_path
    MODEL_EPOCH = conf.best_epoch
    METRICS = {m:get_avail_metris(m) for m in np.append(conf.loss, conf.metrics)}
    model_loaded = load_model('%smodel-sem21cm_ep%d.h5' %(path_out+'checkpoints/', MODEL_EPOCH), custom_objects=METRICS)
    
    print(' Loaded model:\n %smodel-sem21cm_ep%d.h5' %(conf.resume_path, MODEL_EPOCH))
    return model_loaded

title_a = '\t\t _    _ _   _      _   \n\t\t| |  | | \ | |    | |  \n\t\t| |  | |  \| | ___| |_ \n\t\t| |  | | . ` |/ _ \ __|\n\t\t| |__| | |\  |  __/ |_ \n\t\t \____/|_| \_|\___|\__|\n'
title_b = ' _____              _ _      _         ___  __                \n|  __ \            | (_)    | |       |__ \/_ |               \n| |__) | __ ___  __| |_  ___| |_ ___     ) || | ___ _ __ ___  \n|  ___/ `__/ _ \/ _` | |/ __| __/ __|   / / | |/ __| `_ ` _ \ \n| |   | | |  __/ (_| | | (__| |_\__ \  / /_ | | (__| | | | | |\n|_|   |_|  \___|\__,_|_|\___|\__|___/ |____||_|\___|_| |_| |_|\n'
print(title_a+'\n'+title_b)

MAKE_PLOT = False
tobs = 1000

path_pred = '/jmain02/home/J2AD005/jck02/mxb47-jck02/data/inputs/prediction_set/'
path_in = '/jmain02/home/J2AD005/jck02/mxb47-jck02/data/outputs/24-02T09-32-28_10cube/'
config_file = path_in+'net_LSTMUnet_lc.ini'

path_out = path_in+'prediction/'
try:
    os.makedirs(path_out)
except:
    pass
    
dT3 = t2c.read_cbin('%sdata/dT3_21cm_i0.bin' %path_pred)
xH = t2c.read_cbin('%sdata/xH_21cm_i0.bin' %path_pred)
idx, zeta, Rmfp, Tvir, rseed = np.loadtxt('%sparameters/astro_params.txt' %path_pred)
redshift = np.loadtxt('%slc_redshifts.txt' %path_pred)

#dg = LightConeGenerator(path=path_pred, data_temp=np.arange(1), batch_size=1, data_shape=(10, 128, 128))
#X, Y = dg.__getitem__(0)
# Cosmology and Astrophysical parameters
with open(path_pred+'parameters/user_params.txt', 'r') as file:
    params = eval(file.read())

a_params = {'HII_EFF_FACTOR':zeta, 'R_BUBBLE_MAX':Rmfp, 'ION_Tvir_MIN':Tvir}

# Load model
model = LoadSegUnetModel(config_file)

X = np.moveaxis(dT3, -1, 0)
Y = np.moveaxis(xH, -1, 0)
freq_size = 10

idx = [100, 200, 250]
long_X = np.zeros((np.append(np.append(len(idx), freq_size), X.shape[1:])))
long_Y = np.zeros((np.append(np.append(len(idx), freq_size), Y.shape[1:])))
short_redshift = np.zeros(np.append(len(idx), freq_size))
for i_b, rseed in enumerate(idx):
    long_X[i_b] = np.array([X[i,...].astype(np.float32) for i in range(rseed-freq_size//2, rseed+freq_size//2)])
    long_Y[i_b] = np.array([Y[i,...].astype(np.float32) for i in range(rseed-freq_size//2, rseed+freq_size//2)])
    short_redshift[i_b] = np.array([redshift[i] for i in range(rseed-freq_size//2, rseed+freq_size//2)])

X = long_X[..., np.newaxis]
Y = long_Y
print(X.shape, Y.shape, short_redshift.shape)

X_seg = model.predict(X, verbose=0)
X_seg = np.round(X_seg).squeeze()
X = X.squeeze()
print(X_seg.shape)

for i_b in range(X_seg.shape[0]):
    rseed = idx[i_b]
    redshift = short_redshift[i_b]
    my_ext1 = [redshift.min(), redshift.max(), 0, params['BOX_LEN']]
    my_ext2 = [0, params['BOX_LEN'], 0, params['BOX_LEN']]

    phicoef_seg = np.zeros(freq_size)
    phicoef_err = np.zeros(freq_size)
    phicoef_sp = np.zeros(freq_size)
    xn_mask = np.zeros(freq_size)
    xn_seg = np.zeros(freq_size)
    xn_err = np.zeros(freq_size)
    xn_sp = np.zeros(freq_size)
    b0_true = np.zeros(freq_size)
    b1_true = np.zeros(freq_size)
    b2_true = np.zeros(freq_size)
    b0_sp = np.zeros(freq_size)
    b1_sp = np.zeros(freq_size)
    b2_sp = np.zeros(freq_size)
    b0_seg = np.zeros(freq_size)
    b1_seg = np.zeros(freq_size)
    b2_seg = np.zeros(freq_size)

    acc = np.zeros(freq_size)
    prec = np.zeros(freq_size)
    rec = np.zeros(freq_size)
    iou = np.zeros(freq_size)

    for i_z in tqdm(range(X_seg.shape[1])):
        x = np.array(X_seg[i_b, i_z].flatten(), dtype=bool)
        m = np.array(Y[i_b, i_z].flatten(), dtype=bool)

        TP = np.sum((1-x)*(1-m))
        TN = np.sum(x*m)
        FP = np.sum((1-x)*m)
        FN = np.sum(x*(1-m))

        acc[i_z] = (TP+TN)/(TP+TN+FP+FN)
        prec[i_z] = TP/(TP+FP)
        rec[i_z] = TP/(TP+FN)
        iou[i_z] = TP/(TP+FP+FN)
        xn_seg[i_z] = np.mean(X_seg[i_b, i_z])
        xn_mask[i_z] = np.mean(Y[i_b, i_z])
        phicoef_seg[i_z] = matthews_corrcoef(m, x)

    # PLOT MATTHEWS CORRELATION COEF
    fig = plt.figure(figsize=(10, 8))
    plt.plot(redshift, phicoef_seg, '-', label='PhiCoef')
    plt.xlabel('z'), plt.ylabel(r'$r_{\phi}$')
    plt.legend()
    plt.savefig('%sphi_coef_lc%d.png' %(path_out, rseed), bbox_inches='tight'), plt.clf()

    # PLOT STATS
    fig = plt.figure(figsize=(10, 8))
    plt.plot(redshift, acc, label='Accuracy', color='tab:blue')
    plt.plot(redshift, prec, label='Precision', color='tab:orange')
    plt.plot(redshift, rec, label='Recall', color='tab:green')
    plt.plot(redshift, iou, label='IoU', color='tab:red')
    plt.xlabel('z'), plt.ylabel('%')
    plt.legend()
    plt.savefig('%sstats_lc%d.png' %(path_out, rseed), bbox_inches='tight'), plt.clf()

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
    plt.savefig('%spred_xn%d.png' %(path_out, rseed), bbox_inches='tight'), plt.clf()


    # Visual Plot
    i_slice = np.argmin(abs(xn_mask - 0.5))
    i_lc = freq_size//2

    plt.rcParams['font.size'] = 20
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 1.2

    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[0.08,1], height_ratios=[1,1])

    # FIRST LC PLOT
    ax0 = fig.add_subplot(gs[0,0])
    #ax0.set_title('$r_{\phi}=%.3f$ $t_{obs}=%d\,h$' %(phicoef_seg[i_slice], 1000), fontsize=20)
    im = ax0.imshow(X[i_b, :, i_lc, :].T, cmap='jet', aspect='auto', origin='lower', extent=my_ext1)
    ax0.contour(Y[i_b, :, i_lc, :].T, extent=my_ext1)
    #adjust_axis(varr=redshift, xy='x', axis=ax0, to_round=10, step=0.25)
    #adjust_axis(varr=np.linspace(0, 1.6402513488058277, 100), xy='y', axis=ax0, to_round=10, step=0.5)
    ax0.set_ylabel('y [Mpc]', size=20)
    ax0.set_xlabel('z', size=20)

    # FIRST SLICE PLOT
    ax01 = fig.add_subplot(gs[0,1])
    #ax01.set_title(r'$z$ = %.3f   $x_{HI}=%.2f$' %(redshift[i_slice], xn_mask[i_slice]), fontsize=20)
    ax01.imshow(X[i_b, i_slice, ...], cmap='jet', extent=my_ext2, origin='lower')
    ax01.contour(Y[i_b, i_slice, ...], extent=my_ext2)
    #fig.colorbar(im, label=r'$\delta T_b$ [mK]', ax=ax01, pad=0.01, fraction=0.048)

    # SECOND LC PLOT
    ax1 = fig.add_subplot(gs[1,0])
    ax1.imshow(X_seg[i_b,:,i_lc,:].T, cmap='jet', aspect='auto', origin='lower', extent=my_ext1)
    ax1.contour(Y[i_b,:,i_lc,:].T, extent=my_ext1)

    # SECOND SLICE PLOT
    ax11 = fig.add_subplot(gs[1,1])
    ax11.set_title(r'$r_{\phi}$ = %.3f' %(phicoef_seg[i_slice]), fontsize=20)
    im = ax11.imshow(X_seg[i_b, i_slice, ...], cmap='jet', extent=my_ext2, origin='lower')
    ax11.contour(Y[i_b, i_slice, ...], extent=my_ext2)
    ax1.set_ylabel('y [Mpc]', size=20)
    ax1.set_xlabel('z', size=20)

    for ax in [ax01, ax11]:
        ax.set_ylabel('y [Mpc]', size=20)
        ax.set_xlabel('x [Mpc]', size=20)

    plt.subplots_adjust(hspace=0.3, wspace=0.01)
    plt.savefig('%svisual_comparison_lc%d.png' %(path_out, rseed), bbox_inches='tight')

print('... done.')