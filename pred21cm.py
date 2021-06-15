import matplotlib, zipfile
matplotlib.use('agg')
import sys, numpy as np, matplotlib.pyplot as plt, os, tools21cm as t2c, matplotlib.gridspec as gridspec
from sklearn.metrics import matthews_corrcoef
from glob import glob
from tensorflow.keras.models import load_model
from tqdm import tqdm

from config.net_config import NetworkConfig
from utils.other_utils import RotateCube
from utils_network.metrics import iou, iou_loss, dice_coef, dice_coef_loss, balanced_cross_entropy, phi_coef
from utils_network.prediction import SegUnet21cmPredict

from myutils.utils import OrderNdimArray

title_a = '\t\t _    _ _   _      _   \n\t\t| |  | | \ | |    | |  \n\t\t| |  | |  \| | ___| |_ \n\t\t| |  | | . ` |/ _ \ __|\n\t\t| |__| | |\  |  __/ |_ \n\t\t \____/|_| \_|\___|\__|\n'
title_b = ' _____              _ _      _         ___  __                \n|  __ \            | (_)    | |       |__ \/_ |               \n| |__) | __ ___  __| |_  ___| |_ ___     ) || | ___ _ __ ___  \n|  ___/ `__/ _ \/ _` | |/ __| __/ __|   / / | |/ __| `_ ` _ \ \n| |   | | |  __/ (_| | | (__| |_\__ \  / /_ | | (__| | | | | |\n|_|   |_|  \___|\__,_|_|\___|\__|___/ |____||_|\___|_| |_| |_|\n'
print(title_a+'\n'+title_b)

config_file = sys.argv[1]
conf = PredictionConfig(config_file)
PATH_OUT = conf.path_out
PATH_INPUT = conf.path+conf.pred_data

print(' PATH_INPUT = %s' %PATH_INPUT)
if(PATH_INPUT[-3:] == 'zip'):
    ZIPFILE = True
    PATH_IN_ZIP = PATH_INPUT[PATH_INPUT.rfind('/')+1:-4]+'/'
    PATH_UNZIP = PATH_INPUT[:PATH_INPUT.rfind('/')+1]
MAKE_PLOTS = True

# load model
avail_metrics = {'binary_accuracy':'binary_accuracy', 'iou':iou, 'dice_coef':dice_coef, 'iou_loss':iou_loss, 'dice_coef_loss':dice_coef_loss, 'phi_coef':phi_coef, 'mse':'mse', 'mae':'mae', 'binary_crossentropy':'binary_crossentropy', 'balanced_cross_entropy':balanced_cross_entropy} 
MODEL_EPOCH = conf.best_epoch
METRICS = [avail_metrics[m] for m in np.append(conf.loss, conf.metrics)]
cb = {func.__name__:func for func in METRICS if not isinstance(func, str)}
model = load_model('%smodel-sem21cm_ep%d.h5' %(PATH_OUT+'checkpoints/', MODEL_EPOCH), custom_objects=cb)


try:
    os.makedirs(PATH_OUT+'predictions')
except:
    pass
PATH_OUT += 'predictions/pred_tobs1200/'
print(' PATH_OUTPUT = %s' %PATH_OUT)

try:
    os.makedirs(PATH_OUT+'data')
    os.makedirs(PATH_OUT+'plots')
except:
    pass

if(os.path.exists('%sastro_data.txt' %PATH_OUT)):
    astr_data = np.loadtxt('%sastro_data.txt' %PATH_OUT, unpack=True)
    restarts = astr_data[6:].argmin(axis=1)

    if(all(int(np.mean(restarts)) == restarts)):
        restart = int(np.mean(restarts))
        print(' Restart from idx=%d' %restart)
    else:
        ValueError(' Restart points does not match.')

    phicoef_seg, phicoef_err, phicoef_sp, xn_mask, xn_seg, xn_err, xn_sp, b0_true, b1_true, b2_true, b0_seg, b1_seg, b2_seg, b0_sp, b1_sp, b2_sp = astr_data[6:]
    astr_data = astr_data[:6]
else:
    if(ZIPFILE):
        with zipfile.ZipFile(PATH_INPUT, 'r') as myzip:
            astr_data = np.loadtxt(myzip.open('%sastro_params.txt' %PATH_IN_ZIP), unpack=True)
    else:
        astr_data = np.loadtxt('%sastro_params.txt' %PATH_INPUT, unpack=True)

    restart = 0

    phicoef_seg = np.zeros(astr_data.shape[1])
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


params = {'HII_DIM':128, 'DIM':384, 'BOX_LEN':256}
my_ext = [0, params['BOX_LEN'], 0, params['BOX_LEN']]

zc = (astr_data[1,:] < 7.5) + (astr_data[1,:] > 8.3)
c1 = (astr_data[5,:]<=0.25)*(astr_data[5,:]>=0.15)*zc
c2 = (astr_data[5,:]<=0.55)*(astr_data[5,:]>=0.45)*zc
c3 = (astr_data[5,:]<=0.75)*(astr_data[5,:]>=0.85)*zc
indexes = astr_data[0,:]
new_idx = indexes[c1+c2+c3].astype(int)

#for i in tqdm(range(restart, astr_data.shape[1])):
print(new_idx)
for new_i in tqdm(range(3, new_idx.size)):
    i = new_idx[new_i]
    z = astr_data[1,i]
    zeta = astr_data[2,i]
    Rmfp = astr_data[3,i]
    Tvir = astr_data[4,i]
    xn = astr_data[5,i]

    #print('z = %.3f  x_n =%.3f  zeta = %.3f  R_mfp = %.3f  T_vir = %.3f' %(z, xn, zeta, Rmfp, Tvir))
    if(ZIPFILE):
        with zipfile.ZipFile(PATH_INPUT, 'r') as myzip:
            f = myzip.extract(member='%simage_21cm_i%d.bin' %(PATH_IN_ZIP+'data/', i), path=PATH_UNZIP)
            dT3 = t2c.read_cbin(f) 
            f = myzip.extract(member='%smask_21cm_i%d.bin' %(PATH_IN_ZIP+'data/', i), path=PATH_UNZIP)
            mask_xn = t2c.read_cbin(f) 
            os.system('rm -r %s/' %(PATH_UNZIP+PATH_IN_ZIP))
    else:
        dT3 = t2c.read_cbin('%simage_21cm_i%d.bin' %(PATH_INPUT+'data/', i))
        mask_xn = t2c.read_cbin('%smask_21cm_i%d.bin' %(PATH_INPUT+'data/', i))

    # Calculate Betti number
    b0_true[i] = t2c.betti0(data=mask_xn)
    b1_true[i] = t2c.betti1(data=mask_xn)
    b2_true[i] = t2c.betti2(data=mask_xn)

    xn_mask[i] = np.mean(mask_xn)

    plt.rcParams['font.size'] = 20
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = False
    plt.rcParams['ytick.right'] = False
    plt.rcParams['axes.linewidth'] = 1.2
    ls = 22

    # -------- predict with SegUnet 3D  -------- 
    print(' calculating predictioon for data i = %d...' %i)
    X_tta = SegUnet21cmPredict(unet=model, x=dT3, TTA=True)
    X_seg = np.round(np.mean(X_tta, axis=0))
    X_seg_err = np.std(X_tta, axis=0)
    
    # calculate Matthew coef and mean neutral fraction
    phicoef_seg[i] = matthews_corrcoef(mask_xn.flatten(), X_seg.flatten())
    xn_seg[i] = np.mean(X_seg)

    
    # calculate errors
    phicoef_tta = np.zeros(X_tta.shape[0])
    xn_tta = np.zeros(X_tta.shape[0])
    for k in tqdm(range(len(X_tta))):
        xn_tta[k] = np.mean(np.round(X_tta[k]))
        phicoef_tta[k] = matthews_corrcoef(mask_xn.flatten(), np.round(X_tta[k]).flatten())

    xn_err[i] = np.std(xn_tta)
    phicoef_err[i] = np.std(phicoef_tta)
    
    # Calculate Betti number
    b0_seg[i] = t2c.betti0(data=X_seg)
    b1_seg[i] = t2c.betti1(data=X_seg)
    b2_seg[i] = t2c.betti2(data=X_seg)

    # --------------------------------------------
    #  -------- predict with Super-Pixel  --------
    labels = t2c.slic_cube(dT3.astype(dtype='float64'), n_segments=5000, compactness=0.1, max_iter=20, sigma=0, min_size_factor=0.5, max_size_factor=3, cmap=None)
    superpixel_map = t2c.superpixel_map(dT3, labels)
    X_sp = 1-t2c.stitch_superpixels(dT3, labels, bins='knuth', binary=True, on_superpixel_map=True)
    
    # calculate Matthew coef and mean neutral fraction
    phicoef_sp[i] = matthews_corrcoef(mask_xn.flatten(), X_sp.flatten())
    xn_sp[i] = np.mean(X_sp)
    
    # Calculate Betti number
    b0_sp[i] = t2c.betti0(data=X_sp)
    b1_sp[i] = t2c.betti1(data=X_sp)
    b2_sp[i] = t2c.betti2(data=X_sp)

    
    # --------------------------------------------
    if(i in new_idx and MAKE_PLOTS):
        plt.rcParams['xtick.direction'] = 'out'
        plt.rcParams['ytick.direction'] = 'out'
        plt.rcParams['figure.figsize'] = [20, 10]

        idx = params['HII_DIM']//2

        # Plot visual comparison
        fig, axs = plt.subplots(figsize=(20,10), ncols=3, sharey=True, sharex=True)
        (ax0, ax1, ax2) = axs
        ax0.set_title('Super-Pixel ($r_{\phi}=%.3f$)' %phicoef_sp[i], size=ls)
        ax0.imshow(X_sp[:,:,idx], origin='lower', cmap='jet', extent=my_ext)
        ax0.contour(mask_xn[:,:,idx], colors='lime', levels=[0.5], extent=my_ext)
        ax0.set_xlabel('x [Mpc]'), ax0.set_ylabel('y [Mpc]')
        ax1.set_title('SegU-Net ($r_{\phi}=%.3f$)' %phicoef_seg[i], size=ls)
        ax1.imshow(X_seg[:,:,idx], origin='lower', cmap='jet', extent=my_ext)
        ax1.contour(mask_xn[:,:,idx], colors='lime', levels=[0.5], extent=my_ext)
        ax1.set_xlabel('x [Mpc]')
        ax2.set_title('SegUNet Pixel-Error', size=ls)
        im = plt.imshow(X_seg_err[:,:,idx], origin='lower', cmap='jet', extent=my_ext)
        fig.colorbar(im, label=r'$\sigma_{std}$', ax=ax2, pad=0.02, cax=fig.add_axes([0.905, 0.25, 0.02, 0.51]))
        ax2.set_xlabel('x [Mpc]')
        plt.subplots_adjust(hspace=0.1, wspace=0.01)
        for ax in axs.flat: ax.label_outer()
        plt.savefig('%svisual_comparison_i%d.png' %(PATH_OUT+'plots/', i), bbox_inches='tight'), plt.clf()

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

        compare_ml = (mfp_pred_ml[1]/mfp_true[1])
        compare_ml_tta = (mfp_tta[:,1,:]/mfp_true[1])
        compare_sp = (mfp_pred_sp[1]/mfp_true[1])

        fig, ax0 = plt.subplots(figsize=(12, 9))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1.8]) # set height ratios for sublots
        ax0 = plt.subplot(gs[0])
        ax0.set_title('$z=%.3f$\t$x_n=%.3f$\t$r_{\phi}=%.3f$' %(z, xn_mask[i], phicoef_seg[i]), fontsize=ls)
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
        plt.savefig('%sbs_comparison_i%d.png' %(PATH_OUT+'plots/', i), bbox_inches='tight'), plt.clf()

        # Plot dimensioneless power spectra of the x field
        ps_true, ks_true = t2c.power_spectrum_1d(mask_xn, kbins=20, box_dims=256, binning='log')
        ps_pred_sp, ks_pred_sp = t2c.power_spectrum_1d(X_sp, kbins=20, box_dims=256, binning='log')
        ps_pred_ml, ks_pred_ml = t2c.power_spectrum_1d(X_seg, kbins=20, box_dims=256, binning='log')

        ps_tta = np.zeros((X_tta.shape[0],20))
        for k in range(0,X_tta.shape[0]):
            ps_tta[k], ks_pred_ml = t2c.power_spectrum_1d(np.round(X_tta[k]), kbins=20, box_dims=256, binning='log')

        compare_ml = 100*(ps_pred_ml/ps_true - 1.)
        compare_ml_tta = 100*(ps_tta/ps_true - 1.)
        compare_sp = 100*(ps_pred_sp/ps_true - 1.)
        
        fig, ax = plt.subplots(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1.8])
        ax0 = plt.subplot(gs[0])
        ax0.set_title('$z=%.3f$\t$x_n=%.3f$\t$r_{\phi}=%.3f$' %(z, xn_mask[i], phicoef_seg[i]), fontsize=ls)
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
        plt.savefig('%sPk_comparison_i%d.png' %(PATH_OUT+'plots/', i), bbox_inches='tight'), plt.clf()

        ds_data = np.vstack((ks_true, np.vstack((ps_true*ks_true**3/2/np.pi**2, np.vstack((np.vstack((ps_pred_ml*ks_pred_ml**3/2/np.pi**2, np.vstack((np.min(ps_tta*ks_pred_ml**3/2/np.pi**2, axis=0), np.max(ps_tta*ks_pred_ml**3/2/np.pi**2, axis=0))))), ps_pred_sp*ks_pred_sp**3/2/np.pi**2))))))
        bsd_data = np.vstack((mfp_true[0], np.vstack((mfp_true[1], np.vstack((np.vstack((mfp_pred_ml[1], np.vstack((np.min(mfp_tta[:,1,:], axis=0), np.max(mfp_tta[:,1,:], axis=0))))), mfp_pred_sp[1]))))))

        np.savetxt('%sds_data_i%d.txt' %(PATH_OUT+'data/', i), ds_data.T, fmt='%.6e', delimiter='\t', header='k [Mpc^-1]\tds_true\tds_seg_mean\tds_err_min\tds_err_max\tds_sp')
        np.savetxt('%sbsd_data_i%d.txt' %(PATH_OUT+'data/', i), bsd_data.T, fmt='%.6e', delimiter='\t', header='R [Mpc]\tbs_true\tbs_seg_mean\tb_err_min\tbs_err_max\tbs_sp')
    
    new_astr_data = np.vstack((astr_data, phicoef_seg))
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
    np.savetxt('%sastro_data.txt' %(PATH_OUT), new_astr_data.T, fmt='%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d', header='i\tz\teff_f\tRmfp\tTvir\tx_n\tphi_ML\tphi_err phi_SP\txn_mask xn_seg\txn_err\txn_sp\tb0 true b1\tb2\tb0 ML\tb1\tb2\tb0 SP\tb1\tb2')
    np.savetxt('%sastro_data_sample.txt' %(PATH_OUT+'data/'), new_astr_data[:,new_idx].T, fmt='%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d', header='i\tz\teff_f\tRmfp\tTvir\tx_n\tphi_ML\tphi_err phi_SP\txn_mask xn_seg\txn_err\txn_sp\tb0 true b1\tb2\tb0 ML\tb1\tb2\tb0 SP\tb1\tb2')

# Plot phi coeff
plt.rcParams['font.size'] = 16 
redshift, xfrac, phicoef_seg, phicoef_seg_err, phicoef_sp, xn_mask_true, xn_seg, xn_seg_err, xn_sp = OrderNdimArray(np.loadtxt(PATH_OUT+'astro_data.txt', unpack=True, usecols=(1,5,6,7,8,9,10,11,12)), 1)

print('phi_coef = %.3f +/- %.3f\t(SegUnet)' %(np.mean(phicoef_seg), np.std(phicoef_seg)))
print('phi_coef = %.3f\t\t(Superpixel)' %(np.mean(phicoef_sp)))

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(20,8))
#ax0.hlines(y=np.mean(phicoef_seg), xmin=0, xmax=1, ls='--', alpha=0.5)
#ax0.fill_between(x=np.linspace(0, 1, 100), y1=np.mean(phicoef_seg)+np.std(phicoef_seg), y2=np.mean(phicoef_seg)-np.std(phicoef_seg), alpha=0.5, color='lightgray')
# MCC SegUnet
cm = matplotlib.cm.plasma
sc = ax0.scatter(xfrac, phicoef_seg, c=redshift, vmin=7, vmax=9, s=25, cmap=cm, marker='.')
norm = matplotlib.colors.Normalize(vmin=7, vmax=9, clip=True)
mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm)
redshift_color = np.array([(mapper.to_rgba(v)) for v in redshift])
for x, y, e, clr in zip(xfrac, phicoef_seg, phicoef_seg_err, redshift_color):
    ax0.errorbar(x, y, e, lw=1, marker='o', capsize=3, color=clr)
ax0.set_xlim(xfrac.min()-0.02, xfrac.max()+0.02), ax0.set_xlabel(r'$x_i$')
ax0.set_ylim(-0.02, 1.02), ax0.set_ylabel(r'$r_{\phi}$')
fig.colorbar(sc, ax=ax0, pad=0.01, label=r'$z$')
ax2 = ax0.twinx()
ax2.hist(xfrac, np.linspace(0.09, 0.81, 15), density=True, histtype='step', color='tab:blue', alpha=0.5)
ax2.axes.get_yaxis().set_visible(False)
# MCC comparison
ax1.hlines(y=np.mean(phicoef_seg), xmin=0, xmax=1, ls='--', alpha=0.5, color='tab:blue')
ax1.hlines(y=np.mean(phicoef_sp), xmin=0, xmax=1, ls='--', alpha=0.5, color='tab:orange')
new_x = np.linspace(xfrac.min(), xfrac.max(), 100)
f1 = np.poly1d(np.polyfit(xfrac, phicoef_seg, 10))
ax1.plot(new_x, f1(new_x), label='SegUnet', color='tab:blue')
f2 = np.poly1d(np.polyfit(xfrac, phicoef_sp, 10))
ax1.plot(new_x, f2(new_x), label='Super-Pixel', color='tab:orange')
ax1.set_xlim(xfrac.min()-0.02, xfrac.max()+0.02), ax1.set_xlabel(r'$x_i$')
ax1.set_ylim(-0.02, 1.02), ax1.set_ylabel(r'$r_{\phi}$')
ax1.legend(loc=4)
plt.savefig('%sphi_coef.png' %PATH_OUT, bbox_inches="tight"), plt.clf()

# Plot correlation
fig, (ax0, ax1) = plt.subplots(ncols=2)
ax0.plot(xn_mask_true, xn_mask_true, 'k--')
cm = matplotlib.cm.plasma
sc = ax0.scatter(xn_mask_true, xn_seg, c=redshift, vmin=7, vmax=9, s=25, cmap=cm, marker='.')
norm = matplotlib.colors.Normalize(vmin=7, vmax=9, clip=True)
mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap='plasma')
redshift_color = np.array([(mapper.to_rgba(v)) for v in redshift])
for x, y, e, clr in zip(xn_mask_true, xn_seg, xn_seg_err, redshift_color):
    ax0.errorbar(x, y, e, lw=1, marker='o', capsize=3, color=clr)
ax0.set_xlim(xn_mask_true.min()-0.02, xn_mask_true.max()+0.02), ax0.set_xlabel(r'$\rm x_{n,\,true}$')
ax0.set_ylim(xn_mask_true.min()-0.02, xn_mask_true.max()+0.02), ax0.set_ylabel(r'$\rm x_{n,\,predict}$')
fig.colorbar(sc, ax=ax0, pad=0.01, label=r'$z$')

ax1.plot(xn_mask_true, xn_mask_true, 'k--', label='Ground True')
ax1.scatter(xn_mask_true, xn_seg, color='tab:blue', marker='o', label='SegUnet')
ax1.scatter(xn_mask_true, xn_sp, color='tab:orange', marker='o', label='Super-Pixel')
ax1.set_xlim(xn_mask_true.min()-0.02, xn_mask_true.max()+0.02), ax1.set_xlabel(r'$\rm x_{n,\,true}$')
ax1.set_ylim(xn_mask_true.min()-0.02, xn_mask_true.max()+0.02), ax1.set_ylabel(r'$\rm x_{n,\,predict}$')
plt.legend(loc=4)
plt.savefig('%scorr.png' %PATH_OUT, bbox_inches="tight"), plt.clf()


# Betti numbers plot
fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(23,5), sharex=True)
h = np.histogram(xn_mask_true, np.linspace(1e-5, 1., 20), density=True)
new_x = h[1][:-1]+0.5*(h[1][1:]-h[1][:-1])
# Betti 0
f_b0_true = np.array([np.mean(b0_true[(xn_mask_true < h[1][i+1]) * (xn_mask_true >= h[1][i])]) for i in range(h[1].size-1)])
ax0.plot(new_x, f_b0_true, 'k--', label='Ground True')
f_b0_seg = np.array([np.mean(b0_seg[(xn_mask_true < h[1][i+1]) * (xn_mask_true >= h[1][i])]) for i in range(h[1].size-1)])
ax0.plot(new_x, f_b0_seg, label='SegUnet', color='tab:blue', marker='o')
f_b0_sp = np.array([np.mean(b0_sp[(xn_mask_true < h[1][i+1]) * (xn_mask_true >= h[1][i])]) for i in range(h[1].size-1)])
ax0.plot(new_x, f_b0_sp, label='Super-Pixel', color='tab:orange', marker='o')
ax0.legend(loc=1)
ax0.set_xlabel(r'$\rm x^v_{HI}$', size=20), ax0.set_ylabel(r'$\rm\beta_0$', size=20)
# Betti 1
f_b1_true = np.array([np.mean(b1_true[(xn_mask_true < h[1][i+1]) * (xn_mask_true >= h[1][i])]) for i in range(h[1].size-1)])
ax1.plot(new_x, f_b1_true, 'k--', label='Ground True')
f_b1_seg = np.array([np.mean(b1_seg[(xn_mask_true < h[1][i+1]) * (xn_mask_true >= h[1][i])]) for i in range(h[1].size-1)])
ax1.plot(new_x, f_b1_seg, label='SegUnet', color='tab:blue', marker='o')
f_b1_sp = np.array([np.mean(b1_sp[(xn_mask_true < h[1][i+1]) * (xn_mask_true >= h[1][i])]) for i in range(h[1].size-1)])
ax1.plot(new_x, f_b1_sp, label='Super-Pixel', color='tab:orange', marker='o')
ax1.set_xlabel(r'$\rm x^v_{HI}$', size=20), ax1.set_ylabel(r'$\rm\beta_1$', size=20)
# Betti 2
f_b2_true = np.array([np.mean(b2_true[(xn_mask_true < h[1][i+1]) * (xn_mask_true >= h[1][i])]) for i in range(h[1].size-1)])
ax2.plot(new_x, f_b2_true, 'k--', label='Ground True')
f_b2_seg = np.array([np.mean(b2_seg[(xn_mask_true < h[1][i+1]) * (xn_mask_true >= h[1][i])]) for i in range(h[1].size-1)])
ax2.plot(new_x, f_b2_seg, label='SegUnet', color='tab:blue', marker='o')
f_b2_sp = np.array([np.mean(b2_sp[(xn_mask_true < h[1][i+1]) * (xn_mask_true >= h[1][i])]) for i in range(h[1].size-1)])
ax2.plot(new_x, f_b2_sp, label='Super-Pixel', color='tab:orange', marker='o')
ax2.set_xlabel(r'$\rm x^v_{HI}$', size=20), ax2.set_ylabel(r'$\rm\beta_2$', size=20)

plt.subplots_adjust(hspace=0.0)
plt.savefig('%sbetti.png' %PATH_OUT, bbox_inches="tight"), plt.clf()

print('... done.')