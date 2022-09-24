import numpy as np, matplotlib.pyplot as plt, os, json
import tools21cm as t2c
import random, zipfile
from tqdm import tqdm
import matplotlib.gridspec as gridspec
import tools21cm as t2c

from sklearn.metrics import matthews_corrcoef, r2_score

from tensorflow.keras.models import load_model
from utils_network.metrics import get_avail_metris
from config.net_config import NetworkConfig
from utils_network.prediction import SegUnet21cmPredict
from utils_network.data_generator import LightConeGenerator
from utils_plot.other_utils import adjust_axis, MidpointNormalize

def LoadSegUnetModel(cfile):
    conf = NetworkConfig(cfile)
    
    path_out = conf.RESUME_PATH
    MODEL_EPOCH = conf.BEST_EPOCH
    METRICS = {m:get_avail_metris(m) for m in np.append(conf.LOSS, conf.METRICS)}
    model_loaded = load_model('%smodel-sem21cm_ep%d.h5' %(path_out+'checkpoints/', MODEL_EPOCH), custom_objects=METRICS)
    
    print(' Loaded model:\n %smodel-sem21cm_ep%d.h5' %(path_out, MODEL_EPOCH))
    return model_loaded

title_a = '\t\t _    _ _   _      _   \n\t\t| |  | | \ | |    | |  \n\t\t| |  | |  \| | ___| |_ \n\t\t| |  | | . ` |/ _ \ __|\n\t\t| |__| | |\  |  __/ |_ \n\t\t \____/|_| \_|\___|\__|\n'
title_b = ' _____              _ _      _         ___  __                \n|  __ \            | (_)    | |       |__ \/_ |               \n| |__) | __ ___  __| |_  ___| |_ ___     ) || | ___ _ __ ___  \n|  ___/ `__/ _ \/ _` | |/ __| __/ __|   / / | |/ __| `_ ` _ \ \n| |   | | |  __/ (_| | | (__| |_\__ \  / /_ | | (__| | | | | |\n|_|   |_|  \___|\__,_|_|\___|\__|___/ |____||_|\___|_| |_| |_|\n'
print(title_a+'\n'+title_b)

MAKE_PLOT = False
tobs = 1000

path_pred = '/jmain02/home/J2AD005/jck02/mxb47-jck02/data/inputs/prediction_set/'
path_in = '/jmain02/home/J2AD005/jck02/mxb47-jck02/data/outputs/16-06T13-25-30_128slice/'
config_file = path_in+'net_Unet_lc.ini'
path_out = path_in+'prediction/'
try:
    os.makedirs(path_out)
except:
    pass
    
dT3 = t2c.read_cbin('%sdata/dT3_21cm_i0.bin' %path_pred)
dT2 = t2c.read_cbin('%sdata/dT2_21cm_i0.bin' %path_pred)
xH = t2c.read_cbin('%sdata/xH_21cm_i0.bin' %path_pred)
idx, zeta, Rmfp, Tvir, rseed = np.loadtxt('%sparameters/astro_params.txt' %path_pred)
redshift = np.loadtxt('%slc_redshifts.txt' %path_pred)

#dg = LightConeGenerator(path=path_pred, data_temp=np.arange(1), batch_size=1, data_shape=(10, 128, 128))
#X, Y = dg.__getitem__(0)

X = np.moveaxis(dT3, -1, 0)
Y = np.moveaxis(dT2, -1, 0)
xH = np.moveaxis(xH, -1, 0)

# Load & predict with model
model = LoadSegUnetModel(config_file)
X_seg = model.predict(X, verbose=0)


#X_seg = model.predict(X, verbose=0)
astro_params = X_seg[1]
print(X_seg[0].shape, X_seg[1].shape)

np.save(path_out+'dT_pred', X_seg[0])
np.save(path_out+'astroparams_pred', X_seg[1])

X_seg = X_seg[0]
X_seg = X_seg.squeeze()

nu = t2c.z_to_nu(redshift)


# Cosmology and Astrophysical parameters
with open(path_pred+'parameters/user_params.txt', 'r') as file:
    params = eval(file.read())
my_ext1 = [redshift.min(), redshift.max(), 0, params['BOX_LEN']]
my_ext2 = [0, params['BOX_LEN'], 0, params['BOX_LEN']]

a_params = {'HII_EFF_FACTOR':zeta, 'R_BUBBLE_MAX':Rmfp, 'ION_Tvir_MIN':Tvir}

for i in range(300):
    dT3 = t2c.read_cbin('%sdata/dT3_21cm_i%d.bin' %(path_pred, i))
    dT2 = t2c.read_cbin('%sdata/dT2_21cm_i%d.bin' %(path_pred, i))

    Y = np.moveaxis(dT2, -1, 0)
    X = np.moveaxis(dT3, -1, 0)
    X_seg = model.predict(X, verbose=0)

    pred_nu = astro_params[:,-1]
    res = (pred_nu - nu)/nu
    plt.plot(redshift, res)
    plt.ylabel(r'$(\nu_{pred} - \nu_{true})/\nu_{true}$'), plt.xlabel('z')
    plt.savefig('%sres_nu_i%d.png' %(path_pred, i), bbox_inches='tight'), plt.clf()

    plt.scatter(nu, pred_nu, label=r'$\nu_{obs}$')
    plt.plot(nu, nu, 'k--')
    plt.xlabel(r'$\nu_{true}$'), plt.ylabel(r'$\nu_{pred}$')
    plt.legend()
    plt.savefig('%scorr_nu_i%d.png' %(path_pred, i), bbox_inches='tight'), plt.clf()

    r2score_seg = np.zeros_like(redshift)
    xn_mask = np.zeros_like(redshift)

    ps_seg = np.zeros((redshift.size,20))
    ps_true = np.zeros((redshift.size, 20))

    for i_z in tqdm(range(X_seg.shape[0])):
        ps_seg[i_z], ks = t2c.power_spectrum_1d(X_seg[i_z], kbins=20, box_dims=params['BOX_LEN'])
        ps_true[i_z], ks = t2c.power_spectrum_1d(Y[i_z], kbins=20, box_dims=params['BOX_LEN'])

        x = X_seg[i_z].flatten()
        m = Y[i_z].flatten()
        r2score_seg[i_z] = r2_score(m, x)

        xn_mask[i_z] = np.mean(xH[i_z])

    np.savetxt('%sr2score_pred_i%d.txt' %(path_pred, i), r2score_seg)

    np.save('%sPk_seg_i%d' %(path_pred, i), ps_seg)
    np.save('%sPk_true_i%d' %(path_pred, i), ps_true)
    np.save('%sks_i%d' %(path_pred, i), ks)

    # PLOT R2 SCORE
    fig = plt.figure(figsize=(10, 8))
    plt.plot(redshift, r2score_seg, '-')
    plt.xlabel('z'), plt.ylabel(r'$R^2$')
    plt.savefig('%sr2score_i%d.png' %(path_out, i), bbox_inches='tight'), plt.clf()

    # PLOTS POWER SPECTRA
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 2])
    # Main plot
    i_slice = np.argmin(abs(xn_mask - 0.5))
    ax0 = plt.subplot(gs[0])
    ax0.set_title(r'$z$ = %.3f   $x_{HI}=%.2f$' %(redshift[i_slice], xn_mask[i_slice]), fontsize=16)
    ax0.loglog(ks, ps_seg[i_slice]*ks**3/2/np.pi**2, ls='-', color='tab:orange', label='Prediction', lw=1.5)
    ax0.loglog(ks, ps_true[i_slice]*ks**3/2/np.pi**2, ls='-', color='tab:blue', label='True', lw=1.5)
    #ax0.fill_between(z_mean, avrgR_mean_under/b*a, avrgR_mean_over/b*a, color='lightcoral', alpha=0.2)
    ax0.legend()
    ax0.set_xlabel(r'k [Mpc$^{-1}$]'), ax0.set_ylabel(r'$\Delta_{21cm}$')

    # second plot
    ax1 = plt.subplot(gs[1])
    i_k = np.argmin(abs(ks - 0.1))
    ax1.semilogy(redshift, ps_seg[:, i_k]*ks[i_k]**3/2/np.pi**2, ls='--', color='tab:orange', label=r'k = %.1f Mpc$^{-1}$' %ks[i_k], lw=1.5)
    ax1.semilogy(redshift, ps_true[:, i_k]*ks[i_k]**3/2/np.pi**2, ls='--', color='tab:blue', label=r'k = %.1f Mpc$^{-1}$' %ks[i_k], lw=1.5)
    #ax1.plot(redshift, ps_seg[:, 12]*ks[12]**3/2/np.pi**2, ls=':', color='tab:orange', label=r'k = %.1f Mpc${-1}$' %ks[12], lw=1.5)
    #ax1.plot(redshift, ps_true[:, 12]*ks[12]**3/2/np.pi**2, ls=':', color='tab:blue', label=r'k = %.1f Mpc${-1}$' %ks[12], lw=1.5)
    ax1.set_xlabel('$z$'), ax1.set_ylabel(r'$\Delta_{21cm}$')
    #ax1.fill_between(z_quad, diff_s_avrgR_under, diff_s_avrgR_over, color='lightgreen', alpha=0.1)
    #ax1.axhline(y=0,  color='black', ls='dashed')
    #plt.setp(ax0.get_xticklabels(), visible=False)
    ax1.legend()
    plt.subplots_adjust(hspace=0.3)
    plt.savefig('%sPk_comparison_i%d.png' %(path_out, i), bbox_inches='tight'), plt.clf()

    # VISUAL COMPARIOSON PLOT
    i_slice = np.argmin(abs(xn_mask - 0.5))
    i_lc = params['HII_DIM']//2

    plt.rcParams['font.size'] = 20
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 1.2

    fig = plt.figure(figsize=(28, 18))
    gs = gridspec.GridSpec(nrows=3, ncols=2, width_ratios=[3,1], height_ratios=[1, 1, 1])
    plot_min, plot_max = np.min([Y.min(), X_seg.min()]), np.max([Y.max(), X_seg.max()])
    print(Y.min(), Y.max(), X_seg.min(), X_seg.max())

    # FIRST LC PLOT
    ax0 = fig.add_subplot(gs[0,0])
    ax0.set_title(r'$\zeta$ = %.3f        $R_{mfp}$ = %.3f Mpc        $log_{10}(T_{vir}^{min})$ = %.3f' %(zeta, Rmfp, Tvir  ), fontsize=18)
    im = ax0.imshow(X[:,i_lc,:].T, cmap='jet', aspect='auto', origin='lower', extent=my_ext1, norm=MidpointNormalize(midpoint=0., vmin=plot_min, vmax=plot_max))
    ax0.contour(xH[:,i_lc,:].T, extent=my_ext1)
    #adjust_axis(varr=redshift, xy='x', axis=ax0, to_round=10, step=0.25)
    #adjust_axis(varr=np.linspace(0, 1.6402513488058277, 100), xy='y', axis=ax0, to_round=10, step=0.5)
    ax0.set_xlabel('z', size=20), ax0.set_ylabel('y [Mpc]', size=20)

    ax01 = fig.add_subplot(gs[0,1])
    ax01.set_title(r'$z$ = %.3f   $t_{obs}=%d\,h$' %(redshift[i_slice], tobs), fontsize=20)
    im = ax01.imshow(X[i_slice,...], cmap='jet', extent=my_ext2, origin='lower', norm=MidpointNormalize(midpoint=0., vmin=plot_min, vmax=plot_max))
    ax01.contour(xH[i_slice,...], extent=my_ext2)
    fig.colorbar(im, label=r'$\delta T_b$ [mK]', ax=ax01, pad=0.01, fraction=0.048)

    # SECOND LC PLOT
    ax1 = fig.add_subplot(gs[1,0])
    ax1.imshow(Y[:,i_lc,:].T, cmap='jet', aspect='auto', origin='lower', extent=my_ext1, norm=MidpointNormalize(midpoint=0., vmin=plot_min, vmax=plot_max))
    #ax1.contour(xH[:,i_lc,:].T, extent=my_ext1)

    ax11 = fig.add_subplot(gs[1,1])
    ax11.set_title(r'$z$ = %.3f   $x_{HI}=%.2f$' %(redshift[i_slice], xn_mask[i_slice]), fontsize=20)
    im = ax11.imshow(Y[i_slice,...], cmap='jet', extent=my_ext2, origin='lower', norm=MidpointNormalize(midpoint=0., vmin=plot_min, vmax=plot_max))
    #ax11.contour(xH[i_slice,...], extent=my_ext2)
    ax1.set_xlabel('z', size=20), ax1.set_ylabel('y [Mpc]', size=20)
    fig.colorbar(im, label=r'$\delta T_b$ [mK]', ax=ax11, pad=0.01, fraction=0.048)

    # THIRD LC PLOT
    ax2 = fig.add_subplot(gs[2,0])
    ax2.imshow(X_seg[:,i_lc,:].T, cmap='jet', aspect='auto', origin='lower', extent=my_ext1, norm=MidpointNormalize(midpoint=0., vmin=plot_min, vmax=plot_max))
    #ax2.contour(xH[:,i_lc,:].T, extent=my_ext1)

    ax21 = fig.add_subplot(gs[2,1])
    ax21.set_title(r'$R^2$ = %.3f' %(r2score_seg[i_slice]), fontsize=20)
    im = ax21.imshow(X_seg[i_slice,...], cmap='jet', extent=my_ext2, origin='lower', norm=MidpointNormalize(midpoint=0., vmin=plot_min, vmax=plot_max))
    #ax21.contour(xH[i_slice,...], extent=my_ext2)
    ax2.set_xlabel('z', size=20), ax2.set_ylabel('y [Mpc]', size=20)
    fig.colorbar(im, label=r'$\delta T_b$ [mK]', ax=ax21, pad=0.01, fraction=0.048)


    for ax in [ax01, ax11]:
        ax.set_ylabel('y [Mpc]', size=20)
        ax.set_xlabel('x [Mpc]', size=20)

    plt.subplots_adjust(hspace=0.3, wspace=0.01)
    plt.savefig('%svisual_comparison_lc.png' %path_out, bbox_inches='tight')


print('... done.')