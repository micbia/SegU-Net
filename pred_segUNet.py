import numpy as np, matplotlib.pyplot as plt, os
import tools21cm as t2c, py21cmfast as p21c
from py21cmfast import plotting
import random, zipfile
from astropy.cosmology import FlatLambdaCDM
from tqdm import tqdm
import pickle
import matplotlib.gridspec as gridspec

from sklearn.metrics import matthews_corrcoef

from tensorflow.keras.models import load_model
from utils_network.metrics import iou, iou_loss, dice_coef, dice_coef_loss, balanced_cross_entropy, phi_coef
from config.net_config import NetworkConfig
from utils_network.prediction import SegUnet21cmPredict


def LoadSegUnetModel(cfile):
    avail_metrics = {'binary_accuracy':'binary_accuracy', 'iou':iou, 'dice_coef':dice_coef, 'iou_loss':iou_loss, 'dice_coef_loss':dice_coef_loss, 'phi_coef':phi_coef, 'mse':'mse', 'mae':'mae', 'binary_crossentropy':'binary_crossentropy', 'balanced_cross_entropy':balanced_cross_entropy}
    conf = NetworkConfig(cfile)

    outpath = conf.resume_path
    MODEL_EPOCH = conf.best_epoch
    METRICS = [avail_metrics[m] for m in np.append(conf.loss, conf.metrics)]
    cb = {func.__name__:func for func in METRICS if not isinstance(func, str)}
    model_loaded = load_model('%smodel-sem21cm_ep%d.h5' %(outpath+'checkpoints/', MODEL_EPOCH), custom_objects=cb)
    
    print(' Loaded model:\n %smodel-sem21cm_ep%d.h5' %(conf.resume_path, MODEL_EPOCH))
    return model_loaded


avail_metrics = {'binary_accuracy':'binary_accuracy', 'iou':iou, 'dice_coef':dice_coef, 'iou_loss':iou_loss, 'dice_coef_loss':dice_coef_loss, 'phi_coef':phi_coef, 'mse':'mse', 'mae':'mae', 'binary_crossentropy':'binary_crossentropy', 'balanced_cross_entropy':balanced_cross_entropy}                                                                                  

title_a = '\t\t _    _ _   _      _   \n\t\t| |  | | \ | |    | |  \n\t\t| |  | |  \| | ___| |_ \n\t\t| |  | | . ` |/ _ \ __|\n\t\t| |__| | |\  |  __/ |_ \n\t\t \____/|_| \_|\___|\__|\n'
title_b = ' _____              _ _      _         ___  __                \n|  __ \            | (_)    | |       |__ \/_ |               \n| |__) | __ ___  __| |_  ___| |_ ___     ) || | ___ _ __ ___  \n|  ___/ `__/ _ \/ _` | |/ __| __/ __|   / / | |/ __| `_ ` _ \ \n| |   | | |  __/ (_| | | (__| |_\__ \  / /_ | | (__| | | | | |\n|_|   |_|  \___|\__,_|_|\___|\__|___/ |____||_|\___|_| |_| |_|\n'
print(title_a+'\n'+title_b)

MAKE_PLOT = False
tobs = 1000
redshift = np.linspace(7, 9, 20)
redshift = np.append(redshift, 7.310)
redshift = np.append(redshift, 8.032)
redshift = np.sort(np.append(redshift, 8.720))
zeta = 39.204   #65.204
Rmfp = 12.861   #11.861
Tvir = 4.539    #4.539

outpath = '/home/michele/Documents/PhD_Sussex/output/ML/dataset/outputs/new/02-10T23-52-36_128slice/predictions/fiducial_model/'

# Cosmology and Astrophysical parameters
params = {'HII_DIM':128, 'DIM':384, 'BOX_LEN':256}
my_ext = [0, params['BOX_LEN'], 0, params['BOX_LEN']]
c_params = {'OMm':0.27, 'OMb':0.046, 'SIGMA_8':0.82, 'POWER_INDEX':0.96}
a_params = {'HII_EFF_FACTOR':zeta, 'R_BUBBLE_MAX':Rmfp, 'ION_Tvir_MIN':Tvir}

# Load model
model = LoadSegUnetModel('/home/michele/Documents/PhD_Sussex/output/ML/SegNet/tests/runs/net2D_021020.ini')

# Load uv-distribution 
uvfile = '/home/michele/Documents/PhD_Sussex/output/ML/dataset/inputs/uvmap_128_z7-20.pkl'
if not (os.path.exists(uvfile)):
    print('uv-map pickle not found')
else:
    uvs = pickle.load(open(uvfile, 'rb'))

# Initial condition
ic = p21c.initial_conditions(user_params=params, cosmo_params=c_params, random_seed=2021)

phicoef_seg = np.zeros_like(redshift)
phicoef_err = np.zeros_like(phicoef_seg)
phicoef_sp = np.zeros_like(phicoef_seg)
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

acc = np.zeros_like(phicoef_sp)
prec = np.zeros_like(phicoef_sp)
rec = np.zeros_like(phicoef_sp)
iou = np.zeros_like(phicoef_sp)



#for i_z in tqdm(range(redshift.size)):
for i_z in tqdm(range(redshift.size)):
    z = redshift[i_z]
    cube = p21c.run_coeval(redshift=z, init_box=ic, astro_params=a_params, zprime_step_factor=1.05)

    dT = cube.brightness_temp
    xH = cube.xH_box

    uv = uvs['%.3f' %z]
    Nant = uvs['Nant']

    noise_cube = t2c.noise_cube_coeval(params['HII_DIM'], z, depth_mhz=None, obs_time=tobs, filename=None, boxsize=params['BOX_LEN'], total_int_time=6.0, int_time=10.0, declination=-30.0, uv_map=uv, N_ant=Nant, verbose=True, fft_wrap=False)
    dT1 = t2c.subtract_mean_signal(dT, los_axis=2)
    dT2 = dT1 + noise_cube
    dT3 = t2c.smooth_coeval(dT2, z, box_size_mpc=params['HII_DIM'], max_baseline=2.0, ratio=1.0, nu_axis=2)
    smt_xn = t2c.smooth_coeval(xH, z, box_size_mpc=params['HII_DIM'], max_baseline=2.0, ratio=1.0, nu_axis=2)
    mask_xn = smt_xn>0.5
    xn_mask[i_z] = np.mean(mask_xn)

    if(MAKE_PLOT):
        plt.rcParams['font.size'] = 20
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        fig, axs = plt.subplots(2, 2, figsize=(12,12))
        for ax in axs.flat: ax.label_outer()
        fig.suptitle('z = %.3f\t\t$x^v_{HI}$ = %.2f\n$\zeta$ = %.3f        $R_{mfp}$ = %.3f Mpc        $log_{10}(T_{vir}^{min})$ = %.3f' %(z, np.mean(xH), zeta, Rmfp, Tvir), fontsize=18)
        plt.rcParams['font.size'] = 16
        axs[0,0].set_title('$x_{HI}$', size=18)
        axs[0,0].contour(mask_xn[:,:,params['HII_DIM']//2], colors='lime', levels=[0.5], extent=my_ext)
        im = axs[0,0].imshow(xH[:,:,params['HII_DIM']//2], origin='lower', cmap='jet', extent=my_ext)
        fig.colorbar(im, ax=axs[0,0], pad=0.01, fraction=0.048)
        axs[0,0].set_ylabel('y [Mpc]')
        axs[0,1].set_title('$\delta T^{noise}_b(t_{obs}=%d\,h)$' %tobs, size=18)
        im = axs[0,1].imshow(dT2[:,:,params['HII_DIM']//2], origin='lower', cmap='jet', extent=my_ext)
        fig.colorbar(im, ax=axs[0,1], pad=0.01, fraction=0.048)
        axs[1,0].set_title('$\delta T^{sim}_b$', size=18)
        im = axs[1,0].imshow(dT[:,:,params['HII_DIM']//2], origin='lower', cmap='jet', extent=my_ext)
        axs[1,0].set_xlabel('x [Mpc]'), axs[1,0].set_ylabel('y [Mpc]')
        fig.colorbar(im, ax=axs[1,0], pad=0.01, fraction=0.048)
        axs[1,1].set_title('$\delta T^{obs}_b(B=2\,km)$', size=18)
        im = axs[1,1].imshow(dT3[:,:,params['HII_DIM']//2], origin='lower', cmap='jet', extent=my_ext)
        axs[1,1].contour(mask_xn[:,:,params['HII_DIM']//2], colors='k', levels=[0.5], extent=my_ext)
        axs[1,1].set_xlabel('x [Mpc]')
        fig.colorbar(im, ax=axs[1,1], pad=0.02, fraction=0.048)
        plt.subplots_adjust(hspace=0.01, wspace=0.12)
        plt.savefig('%scube21cm_i%d.png' %(outpath+'plots/', i_z), bbox_inches='tight'), plt.clf()

    # SegU-net
    X_tta = SegUnet21cmPredict(unet=model, x=dT3, TTA=False)

    X_seg = np.round(np.mean(X_tta, axis=0))
    
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

    new_astr_data = np.vstack((redshift, acc))
    new_astr_data = np.vstack((new_astr_data, prec))
    new_astr_data = np.vstack((new_astr_data, rec))
    new_astr_data = np.vstack((new_astr_data, iou))
    new_astr_data = np.vstack((new_astr_data, xn_mask))
    new_astr_data = np.vstack((new_astr_data, xn_seg))
    np.savetxt('%sastro_data_FM_tobs%d_stats.txt' %(outpath, tobs), new_astr_data.T, fmt='%.4f\t'*7, header='tobs = %d, eff_f = %.3f, Rmfp = %.3f, Tvir = %.3f\nz\tacc\tprec\trec\tiou\txn_mask xn_seg' %(tobs, zeta, Rmfp, Tvir))
    

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

        idx = params['HII_DIM']//2

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
        plt.savefig('%svisual_comparison_i%d.png' %(outpath+'plots/', i_z), bbox_inches='tight'), plt.clf()

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
        plt.savefig('%sbs_comparison_i%d.png' %(outpath+'plots/', i_z), bbox_inches='tight'), plt.clf()

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
        plt.savefig('%sPk_comparison_i%d.png' %(outpath+'plots/', i_z), bbox_inches='tight'), plt.clf()

        ds_data = np.vstack((ks_true, np.vstack((ps_true*ks_true**3/2/np.pi**2, np.vstack((np.vstack((ps_pred_ml*ks_pred_ml**3/2/np.pi**2, np.vstack((np.min(ps_tta*ks_pred_ml**3/2/np.pi**2, axis=0), np.max(ps_tta*ks_pred_ml**3/2/np.pi**2, axis=0))))), ps_pred_sp*ks_pred_sp**3/2/np.pi**2))))))
        bsd_data = np.vstack((mfp_true[0], np.vstack((mfp_true[1], np.vstack((np.vstack((mfp_pred_ml[1], np.vstack((np.min(mfp_tta[:,1,:], axis=0), np.max(mfp_tta[:,1,:], axis=0))))), mfp_pred_sp[1]))))))

        np.savetxt('%sds_data_i%d.txt' %(outpath+'data/', i_z), ds_data.T, fmt='%.6e', delimiter='\t', header='k [Mpc^-1]\tds_true\tds_seg_mean\tds_err_min\tds_err_max\tds_sp')
        np.savetxt('%sbsd_data_i%d.txt' %(outpath+'data/', i_z), bsd_data.T, fmt='%.6e', delimiter='\t', header='R [Mpc]\tbs_true\tbs_seg_mean\tb_err_min\tbs_err_max\tbs_sp')

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
    np.savetxt('%sastro_data_FM_tobs%d.txt' %(outpath, tobs), new_astr_data.T, fmt='%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d', header='tobs = %d, eff_f = %.3f, Rmfp = %.3f, Tvir = %.3f\nz\tphi_ML\tphi_err phi_SP\txn_mask xn_seg\txn_err\txn_sp\tb0 true b1\tb2\tb0 ML\tb1\tb2\tb0 SP\tb1\tb2' %(tobs, zeta, Rmfp, Tvir))
    '''

    os.system('rm /home/michele/21CMMC_Boxes/*h5')
print('... done.')