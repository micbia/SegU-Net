import numpy as np, matplotlib.pyplot as plt, os
import matplotlib
import tools21cm as t2c
import tensorflow as tf

from tqdm import tqdm
from glob import glob
import matplotlib.gridspec as gridspec

from sklearn.metrics import matthews_corrcoef, r2_score
from sklearn.metrics import confusion_matrix

from utils_pred.prediction import SegUnet2Predict, LoadSegUnetModel
from utils_plot.other_utils import adjust_axis


title_a = '\t\t _    _ _   _      _   \n\t\t| |  | | \ | |    | |  \n\t\t| |  | |  \| | ___| |_ \n\t\t| |  | | . ` |/ _ \ __|\n\t\t| |__| | |\  |  __/ |_ \n\t\t \____/|_| \_|\___|\__|\n'
title_b = ' _____              _ _      _         ___  __                \n|  __ \            | (_)    | |       |__ \/_ |               \n| |__) | __ ___  __| |_  ___| |_ ___     ) || | ___ _ __ ___  \n|  ___/ `__/ _ \/ _` | |/ __| __/ __|   / / | |/ __| `_ ` _ \ \n| |   | | |  __/ (_| | | (__| |_\__ \  / /_ | | (__| | | | | |\n|_|   |_|  \___|\__,_|_|\___|\__|___/ |____||_|\___|_| |_| |_|\n'
print(title_a+'\n'+title_b)

PLOT_STATS, PLOT_MEAN, PLOT_VISUAL, PLOT_ERROR, PLOT_SCORE = True, True, True, False, True
#PLOT_STATS, PLOT_MEAN, PLOT_VISUAL, PLOT_ERROR, PLOT_SCORE = False, False, False, False, True

#path_pred = '/store/ska/sk09/segunet/inputs/dataLC_128_pred_310822/'
path_pred = '/store/ska/sk09/segunet/inputs/dataLC_128_valid_190922/'
#dataset_size = len(glob(path_pred+'data/dT2*'))
pred_idx = np.arange(11)
#pred_idx = np.loadtxt(path_pred+'good_data.txt', dtype=int)

#path_model = '/scratch/snx3000/mibianco/output_segunet/outputs/BT23-09T11-19-42_128slice/'
path_model = '/scratch/snx3000/mibianco/output_segunet/outputs/testall_23-09T21-05-03_128slice/'

#path_model = '/scratch/snx3000/mibianco/output_segunet/outputs/BCE_biastrain21-09T19-29-15_128slice/'
#path_model = '/scratch/snx3000/mibianco/output_segunet/outputs/16-09T14-15-20_128slice/'
#path_model = '/scratch/snx3000/mibianco/output_segunet/outputs/17-09T22-53-05_128slice/'
#path_model = '/scratch/snx3000/mibianco/output_segunet/outputs/19-09T18-59-33_128slice/'
#path_model = '/scratch/snx3000/mibianco/output_segunet/outputs/14-09T13-23-19_128slice/'
#path_model = '/scratch/snx3000/mibianco/output_segunet/outputs/dT4pca_12-09T16-07-57_128slice/'
#path_model = '/scratch/snx3000/mibianco/output_segunet/outputs/dT3_12-09T15-23-31_128slice/'
config_file = path_model+'net_Unet_lc.ini'
path_out = path_model+'prediction/'
try:
    os.makedirs(path_out)
except:
    pass

# load redshift
redshift = np.loadtxt('%slc_redshifts.txt' %path_pred)
    
# Cosmology and Astrophysical parameters
with open(path_pred+'parameters/user_params.txt', 'r') as file:
    params = eval(file.read())

with open(path_pred+'parameters/cosm_params.txt', 'r') as file:
    c_params = eval(file.read())

astro_params = np.loadtxt('%sparameters/astro_params.txt' %path_pred)

# Load best model
model = LoadSegUnetModel(config_file)

# Prediction loop
#for i_pred in tqdm(range(dataset_size)):
for ii in tqdm(range(pred_idx.size)):
    i_pred = pred_idx[ii]
    idx, zeta, Rmfp, Tvir, rseed = astro_params[i_pred]
    a_params = {'HII_EFF_FACTOR':zeta, 'R_BUBBLE_MAX':Rmfp, 'ION_Tvir_MIN':Tvir}

    # Load input and target
    x_input = t2c.read_cbin('%sdata/dT4pca_21cm_i%d.bin' %(path_pred, i_pred))
    y_true = t2c.read_cbin('%sdata/xH_21cm_i%d.bin' %(path_pred, i_pred))

    # Prediction on dataset
    #x_input, _ = t2c.smooth_lightcone(x_input, z_array=redshift, box_size_mpc=params['BOX_LEN'])    # additional smoothing
    y_tta = SegUnet2Predict(unet=model, lc=x_input, tta=PLOT_ERROR)
    if(PLOT_ERROR):
        y_pred = np.round(np.clip(np.mean(y_tta, axis=0), 0, 1))
        y_error = np.std(y_tta, axis=0)
        y_tta = np.round(np.clip(y_tta, 0, 1)) 
    else:
        y_pred = np.round(np.clip(y_tta.squeeze(), 0, 1))
    assert x_input.shape == y_pred.shape

    # Statistical quantities
    TP = np.sum(y_pred * y_true, axis=(0,1))
    TN = np.sum((1-y_pred) * (1-y_true), axis=(0,1))
    FP = np.sum(y_pred * (1-y_true), axis=(0,1))
    FN = np.sum((1-y_pred) * y_true, axis=(0,1))
    #TN, FP, FN, TP = confusion_matrix(y_true[..., 0], y_pred[..., 0]).ravel()
    #assert TP+TN+FP+FN == params['HII_DIM']*params['HII_DIM']
    #assert FP+TN == np.sum(1-y_true, axis=(0,1))
    #assert TP+FP == np.sum(y_true, axis=(0,1))
    
    FNR = FN/(FN+TP)
    FPR = FP/(FP+TN)
    TNR = TN/(FP+TN)   # a.k.a specificy
    TPR = TP/(TP+FP)   # a.k.a precision
    acc = (TP+TN)/(TP+TN+FP+FN)
    #rec = TP/(TP+FN)    # very similar to precision
    iou = TP/(TP+FP+FN)
    mcc = (TP*TN - FP*FN) / (np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) + tf.keras.backend.epsilon())
    mean_pred = np.mean(y_pred, axis=(0,1))
    mean_true = np.mean(y_true, axis=(0,1))
    np.savetxt('%sstats_i%d.txt' %(path_out, i_pred), np.array([redshift, acc, TPR, TNR, iou, mcc, mean_pred, mean_true]).T, fmt='%.3f\t'+('%.3e\t'*7)[:-1], header='eff_fact=%.4f\tRmfp=%.4f\tTvir=%.4e\t\nz\tacc\t\tprec\t\tspec\t\tiou\t\tmcc\t\tx_pred\t\ty_true' %(zeta, Rmfp, Tvir))

    xHI_plot = np.arange(0.1, 1., 0.1)
    redshift_plot = np.array([redshift[np.argmin(abs(mean_true - meanHI))] for meanHI in xHI_plot])

    if(PLOT_STATS):
        # PLOT MATTHEWS CORRELATION COEF
        fig = plt.figure(figsize=(10, 8))
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['font.size'] = 16
        plt.plot(redshift, mcc, '-', label='PhiCoef')
        plt.vlines(redshift_plot, ymin=0, ymax=1, color='black', ls='--')
        plt.xlabel('z'), plt.ylabel(r'$r_{\phi}$')
        plt.ylim(0, 1)
        plt.legend()
        for iplot in range(redshift_plot.size):
            plt.text(redshift_plot[iplot]+0.03, 0.95, round(xHI_plot[iplot],1), rotation=90)
        plt.savefig('%smcc_i%d.png' %(path_out, i_pred), bbox_inches='tight'), plt.clf()

        # PLOT STATS
        fig = plt.figure(figsize=(10, 8))
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['font.size'] = 16
        plt.plot(redshift, acc, label='Accuracy', color='tab:blue')
        plt.plot(redshift, TPR, label='Precision', color='tab:orange')
        plt.plot(redshift, TNR, label='Specificy', color='tab:green')
        plt.plot(redshift, iou, label='IoU', color='tab:red')
        plt.vlines(redshift_plot, ymin=0, ymax=1, color='black', ls='--')
        plt.xlabel('z'), plt.ylabel('%')
        plt.ylim(0, 1)
        for iplot in range(redshift_plot.size):
            plt.text(redshift_plot[iplot]+0.03, 0.95, round(xHI_plot[iplot],1), rotation=90)
        plt.legend()
        plt.savefig('%sstats_i%d.png' %(path_out, i_pred), bbox_inches='tight'), plt.clf()

        # PLOT RATES
        fig = plt.figure(figsize=(10, 8))
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['font.size'] = 16
        plt.plot(redshift, FNR, label='FNR', color='tab:blue')
        plt.plot(redshift, FPR, label='FPR', color='tab:red')
        plt.plot(redshift, TPR, label='TPR', color='tab:orange')
        plt.plot(redshift, TNR, label='TNR', color='tab:green')
        plt.vlines(redshift_plot, ymin=0, ymax=1, color='black', ls='--')
        plt.xlabel('z'), plt.ylabel('%')
        plt.ylim(0, 1)
        for iplot in range(redshift_plot.size):
            plt.text(redshift_plot[iplot]+0.03, 0.95, round(xHI_plot[iplot],1), rotation=90)
        plt.legend()
        plt.savefig('%srates_i%d.png' %(path_out, i_pred), bbox_inches='tight'), plt.clf()

    if(PLOT_MEAN):
        # PLOTS AVERGE MASK HI
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1.8])

        # Main plot
        ax0 = plt.subplot(gs[0])
        ax0.plot(redshift, mean_pred, ls='-', color='tab:orange', label='Prediction', lw=1.5)
        ax0.plot(redshift, mean_true, ls='-', color='tab:blue', label='True', lw=1.5)
        #ax0.fill_between(z_mean, avrgR_mean_under/b*a, avrgR_mean_over/b*a, color='lightcoral', alpha=0.2)
        ax0.legend()
        ax0.set_ylabel(r'$x_{HI}$')

        # plot relative difference
        ax1 = plt.subplot(gs[1], sharex = ax0)
        perc_diff = mean_true/mean_pred-1
        ax1.plot(redshift, perc_diff, 'k-', lw=1.5)
        ax1.set_ylabel('difference (%)')
        ax1.set_xlabel('$z$')
        #ax1.fill_between(z_quad, diff_s_avrgR_under, diff_s_avrgR_over, color='lightgreen', alpha=0.1)
        ax1.axhline(y=0,  color='black', ls='dashed')
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=.0)
        plt.savefig('%smean_i%d.png' %(path_out, i_pred), bbox_inches='tight'), plt.clf()
        
    if(PLOT_VISUAL):
        # Visual Plot
        i_slice = np.argmin(abs(mean_true - 0.5))
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
        #ax0.set_title('$r_{\phi}=%.3f$ $t_{obs}=%d\,h$' %(mcc[i_slice], 1000), fontsize=20)
        im = ax0.imshow(x_input[:,i_lc,:], cmap='jet', aspect='auto', origin='lower')
        ax0.contour(y_true[:,i_lc,:])
        adjust_axis(varr=redshift, xy='x', axis=ax0, to_round=1, step=0.25)
        adjust_axis(varr=np.linspace(0, params['BOX_LEN'], params['HII_DIM']), xy='y', axis=ax0, to_round=params['BOX_LEN'], step=50)
        ax0.set_ylabel('y [Mpc]', size=20)
        ax0.set_xlabel('z', size=20)

        # FIRST SLICE PLOT
        ax01 = fig.add_subplot(gs[0,1])
        ax01.set_title(r'$z$ = %.3f   $x_{HI}=%.2f$' %(redshift[i_slice], mean_true[i_slice]), fontsize=20)
        ax01.imshow(x_input[...,i_slice], cmap='jet', origin='lower')
        ax01.contour(y_true[...,i_slice])
        adjust_axis(varr=np.linspace(0, params['BOX_LEN'], params['HII_DIM']), xy='xy', axis=ax01, to_round=params['BOX_LEN'], step=50)
        #fig.colorbar(im, label=r'$\delta T_b$ [mK]', ax=ax01, pad=0.01, fraction=0.048)

        # SECOND LC PLOT
        ax1 = fig.add_subplot(gs[1,0])
        ax1.imshow(y_pred[:,i_lc,:]  , cmap='jet', aspect='auto', origin='lower', vmin=y_pred.min(), vmax=y_pred.max())
        ax1.contour(y_true[:,i_lc,:])
        adjust_axis(varr=redshift, xy='x', axis=ax1, to_round=1, step=0.25)
        adjust_axis(varr=np.linspace(0, params['BOX_LEN'], params['HII_DIM']), xy='y', axis=ax1, to_round=params['BOX_LEN'], step=50)
        ax1.set_ylabel('y [Mpc]', size=20)
        ax1.set_xlabel('z', size=20)

        # SECOND SLICE PLOT
        ax11 = fig.add_subplot(gs[1,1])
        ax11.set_title(r'$r_{\phi}$ = %.3f' %(mcc[i_slice]), fontsize=20)
        im = ax11.imshow(y_pred[...,i_slice], cmap='jet', origin='lower', vmin=y_pred.min(), vmax=y_pred.max())
        ax11.contour(y_true[...,i_slice])
        adjust_axis(varr=np.linspace(0, params['BOX_LEN'], params['HII_DIM']), xy='xy', axis=ax11, to_round=params['BOX_LEN'], step=50)

        for ax in [ax01, ax11]:
            ax.set_ylabel('y [Mpc]', size=20)
            ax.set_xlabel('x [Mpc]', size=20)

        plt.subplots_adjust(hspace=0.3, wspace=0.01)
        plt.savefig('%svisual_i%d.png' %(path_out, i_pred), bbox_inches='tight')

    if(PLOT_ERROR):
        # Visual Plot
        i_slice = np.argmin(abs(mean_true - 0.5))
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
        #ax0.set_title('$r_{\phi}=%.3f$ $t_{obs}=%d\,h$' %(mcc[i_slice], 1000), fontsize=20)
        ax0.imshow(y_pred[:,i_lc,:], cmap='jet', aspect='auto', origin='lower')
        ax0.contour(y_true[:,i_lc,:])
        adjust_axis(varr=redshift, xy='x', axis=ax0, to_round=1, step=0.25)
        adjust_axis(varr=np.linspace(0, params['BOX_LEN'], params['HII_DIM']), xy='y', axis=ax0, to_round=params['BOX_LEN'], step=50)
        ax0.set_ylabel('y [Mpc]', size=20)
        ax0.set_xlabel('z', size=20)

        # FIRST SLICE PLOT
        ax01 = fig.add_subplot(gs[0,1])
        ax01.set_title(r'$z$ = %.3f   $x_{HI}=%.2f$' %(redshift[i_slice], mean_true[i_slice]), fontsize=20)
        im = ax01.imshow(y_pred[...,i_slice], cmap='jet', origin='lower')
        ax01.contour(y_true[...,i_slice])
        adjust_axis(varr=np.linspace(0, params['BOX_LEN'], params['HII_DIM']), xy='xy', axis=ax01, to_round=params['BOX_LEN'], step=50)

        # SECOND LC PLOT
        ax1 = fig.add_subplot(gs[1,0])
        im = ax1.imshow(y_error[:,i_lc,:]  , cmap='jet', aspect='auto', origin='lower', vmin=y_error.min(), vmax=y_error.max())
        ax1.contour(y_true[:,i_lc,:])
        adjust_axis(varr=redshift, xy='x', axis=ax1, to_round=1, step=0.25)
        adjust_axis(varr=np.linspace(0, params['BOX_LEN'], params['HII_DIM']), xy='y', axis=ax1, to_round=params['BOX_LEN'], step=50)
        fig.colorbar(im, label=r'$\sigma_{std}$', ax=ax1, pad=0.01, fraction=0.048)
        ax1.set_ylabel('y [Mpc]', size=20)
        ax1.set_xlabel('z', size=20)

        # SECOND SLICE PLOT
        ax11 = fig.add_subplot(gs[1,1])
        ax11.set_title(r'$r_{\phi}$ = %.3f' %(mcc[i_slice]), fontsize=20)
        im = ax11.imshow(y_error[...,i_slice], cmap='jet', origin='lower', vmin=y_error.min(), vmax=y_error.max())
        ax11.contour(y_true[...,i_slice])
        adjust_axis(varr=np.linspace(0, params['BOX_LEN'], params['HII_DIM']), xy='xy', axis=ax11, to_round=params['BOX_LEN'], step=50)
        fig.colorbar(im, label=r'$\sigma_{std}$', ax=ax11, pad=0.01, fraction=0.048)

        for ax in [ax01, ax11]:
            ax.set_ylabel('y [Mpc]', size=20)
            ax.set_xlabel('x [Mpc]', size=20)

        plt.subplots_adjust(hspace=0.3, wspace=0.01)
        plt.savefig('%serror_i%d.png' %(path_out, i_pred), bbox_inches='tight')

    if(PLOT_SCORE):
        #if(i_pred == 0):
        if(ii == 0):
            fig1, ax_s = plt.subplots(figsize=(10,8), ncols=1)

        # get redshift color
        cm = matplotlib.cm.plasma
        sc = ax_s.scatter(mean_true, mcc, c=redshift, vmin=redshift.min(), vmax=redshift.max(), s=25, cmap=cm, marker='.')
        norm = matplotlib.colors.Normalize(vmin=7, vmax=9, clip=True)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm)
        redshift_color = np.array([(mapper.to_rgba(v)) for v in redshift])

        #for x, y, e, clr, red in zip(mean_true, mcc, mcc_err, redshift_color, redshift):
        #    ax0.errorbar(x, y, e, lw=1, marker='o', capsize=3, color=clr)
        
        for x, y, clr, red in zip(mean_true, mcc, redshift_color, redshift):
            ax_s.scatter(x, y, lw=1, marker='o', color=clr)
        
        ax_s.set_xlim(mean_true.min()-0.02, mean_true.max()+0.02), ax_s.set_xlabel(r'$\rm x^v_{HI}$', size=20)
        ax_s.set_ylim(-0.02, 1.02), ax_s.set_ylabel(r'$\rm r_{\phi}$', size=20)
        ax_s.set_yticks(np.arange(0, 1.1, 0.1))
        ax_s.set_xticks(np.arange(0, 1.1, 0.2))
        #ax_s.hlines(y=np.mean(mcc), xmin=-0.02, xmax=1.1, ls='--', alpha=0.8, color='tab:blue', zorder=3)
        #if(i_pred == 0):
        if(ii == 0):
            fig1.colorbar(sc, ax=ax_s, pad=0.01, label=r'$\rm z$')

if(PLOT_SCORE):
    fig1.savefig('%smcc_dataset.png' %path_out, bbox_inches='tight')

print('... done.')

