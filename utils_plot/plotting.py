import matplotlib
matplotlib.use('agg')
import numpy as np, tools21cm as t2c
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import matthews_corrcoef

import sys
sys.path.append("..")
from tqdm import tqdm

def get_axis_locs(varr, to_round=10, step=5, fmt=int):    
    v_max = int(round(varr.max()/to_round)*to_round) if int(round(varr.max()/to_round)*to_round) <= varr.max() else int(round(varr.max()/to_round)*to_round)-to_round
    v_min = int(round(varr.min()/to_round)*to_round) if int(round(varr.min()/to_round)*to_round) >= varr.min() else int(round(varr.min()/to_round)*to_round)+to_round
    v_plot = np.arange(v_min, v_max+step, step)
    loc_v = np.array([np.argmin(abs(varr-v_plot[i])) for i in range(v_plot.size)]).astype(fmt)
    return loc_v

# Plots Predictions
def plot_sample(X, y, preds, idx):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(X[idx,...].squeeze(), origin='lower', cmap='jet')
    ax[0].contour(y[idx].squeeze(), origin='lower', colors='k', levels=[0.5])
    ax[0].set_title('True')
    ax[0].grid(False)
    ax[1].imshow(preds[idx].squeeze(), origin='lower', vmin=0, vmax=1, cmap='jet')
    ax[1].contour(y[idx].squeeze(), origin='lower', colors='k', levels=[0.5])
    ax[1].set_title('Predicted')
    ax[1].grid(False)


# Plots Predictions
def plot_sample3D(X, y, preds, idx, path):
    for i in range(X.shape[1]):
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(X[idx,i,...].squeeze(), origin='lower', cmap='jet')
        ax[0].contour(y[idx,i, ...].squeeze(), origin='lower', colors='k', levels=[0.5])
        ax[0].set_title('True')
        ax[0].grid(False)
        ax[1].imshow(preds[idx,i, ...].squeeze(), origin='lower', vmin=0, vmax=1, cmap='jet')
        ax[1].contour(y[idx,i, ...].squeeze(), origin='lower', colors='k', levels=[0.5])
        ax[1].set_title('Predicted')
        ax[1].grid(False)
        plt.savefig(path+'train_prediction_idx%d_x%d.png' %(idx, i), bbox_inches='tight')
        plt.close()


# Plot Loss
def plot_loss(output, path='./'):
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_title('Learning curve')
    ax1.plot(output.history["loss"], color='tab:blue', label='train loss')
    ax1.plot(output.history["val_loss"], color='tab:green', label='validation loss')
    ax1.plot(np.argmin(output.history["val_loss"]), np.min(output.history["val_loss"]), marker="x", color="r", label="best model")
    ax1.set_xlabel('Epoch'), ax1.set_ylabel('Loss'), ax1.legend()

    ax2 = ax1.twinx()
    ax2.semilogy(output.history['lr'], color='gray', ls='--', alpha=0.5)
    ax2.set_ylabel('Learning Rate')

    plt.savefig(path+'loss.png', bbox_inches='tight')
    plt.close('all')


# Plot matthews_corrcoef
def plot_phicoef(true, predict, indexes, red, xfrac, path='./'):
    pcoef = np.zeros_like(indexes)
    for i in tqdm(range(indexes.size)):
        pcoef[i] = matthews_corrcoef(true[i].flatten(), predict[i].flatten().round())
    np.savetxt(path+'phi_coef.txt', np.array([indexes, red, xfrac, pcoef]).T, fmt='%d\t%.3f\t%.3f\t%.3f', header='z\tx_n\tphi_coef')
    fig = plt.figure(figsize=(12, 12))
    plt.plot(xfrac, pcoef, 'r.')
    plt.xlabel('$x_n$'), plt.ylabel('$r_{\phi}$')
    plt.savefig(path+'phi_coef.png', bbox_inches='tight')
    plt.close('all')


def plot_slice(i, BOX_LEN, path_out):
    """ e.g.
        
        cd ~/SegU-Net/utils_plot

        from plotting import plot_slice 
        path = '/cosma6/data/dp004/dc-bian1/inputs/data3D_128_100821/' 
        plot_slice(i=8083, BOX_LEN=256, path_out=path)
    """
    my_ext = [0, BOX_LEN, 0, BOX_LEN] 

    xH = t2c.read_cbin('%sdata/xH_21cm_i%d.bin' %(path_out, i)) 
    dT = t2c.read_cbin('%sdata/dT1_21cm_i%d.bin' %(path_out, i))  
    idx = xH.shape[0]//2

    # Plot outputs comparisons 
    fig, axs = plt.subplots(1, 2, figsize=(12,7)) 
    #fig.suptitle('z=%.3f\t\t$x_n$=%.2f\n$\zeta$=%.3f\t\t$R_{mfp}$=%.3f\t\t$T_{vir}^{min}$=%.3f' %(z, xn, eff_fact, Rmfp, Tvir), fontsize=18)
    #axs[0].set_title('$x_{HII}$', size=16) 
    im = axs[0].imshow(1-xH[:,:,idx], origin='lower', cmap='jet', extent=my_ext) 
    axs[0].set_xlabel('[Mpc]'), axs[0].set_ylabel('[Mpc]'); 
    fig.colorbar(im, ax=axs[0], pad=0.01, fraction=0.048, label='$x_{HII}$') 

    #axs[1].set_title('$\delta T_b$', size=16) 
    im = axs[1].imshow(dT[:,:,idx], origin='lower', cmap='jet', extent=my_ext) 
    axs[1].set_xlabel('[Mpc]'), axs[1].set_ylabel('[Mpc]'); 
    fig.colorbar(im, ax=axs[1], pad=0.01, fraction=0.048, label='$\delta T_b$') 
    plt.subplots_adjust(wspace=0.3)
    plt.savefig('%simages/slice_i%d.png' %(path_out, i), bbox_inches='tight')
    print(' Plot saved.')
    plt.close()


def plot_lc(i, BOX_LEN, path_out, tobs=1000):
    my_ext = [0, BOX_LEN, 0, BOX_LEN] 

    redshift = np.loadtxt(path_out+'lc_redshifts.txt')
    idx, eff_fact, Rmfp, Tvir, seed = np.loadtxt(path_out+'parameters/astro_params.txt', unpack=True)
    zeta, Rmfp, Tvir = eff_fact[i], Rmfp[i], Tvir[i]

    xH = t2c.read_cbin('%sdata/xH_21cm_i%d.bin' %(path_out, i)) 
    dT3 = t2c.read_cbin('%sdata/dT3_21cm_i%d.bin' %(path_out, i))  
    HII_DIM = dT3.shape[0]
    i_plot = dT3.shape[-1]//2

    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[3,1], height_ratios=[1])

    # FIRST LC PLOT
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title('$\zeta$ = %.3f   $R_{mfp}$ = %.3f Mpc   $log_{10}(T_{vir}^{min})$ = %.3f   $t_{obs}=%d\,h$' %(zeta, Rmfp, Tvir, tobs), fontsize=16)
    im = ax1.imshow(dT3[:,HII_DIM//2,:], cmap='jet', origin='lower')
    ax1.contour(xH[:,HII_DIM//2,:])

    idx_x = get_axis_locs(redshift, to_round=1, step=1)
    idx_y = np.linspace(0, dT3.shape[0]-1, 7, endpoint=True, dtype=int)

    ax1.set_xlabel('z', size=16)
    ax1.set_ylabel('x [Mpc]', size=16)
    ax1.set_xticks(idx_x), ax1.set_yticks(idx_y)
    ax1.set_xticklabels([int(round(redshift[i_n])) for i_n in idx_x])
    ax1.set_yticklabels(np.array(idx_y*BOX_LEN/HII_DIM, dtype=int))
    #ax1.label_outer()
    ax1.tick_params(axis='both', length=5, width=1.2)
    ax1.tick_params(which='minor', axis='both', length=5, width=1.2)

    # SECOND LC PLOT
    ax01 = fig.add_subplot(gs[0,1])
    ax01.set_title('$z$ = %.3f   $\delta T_b$=%.3f' %(redshift[i_plot], np.mean(dT3[:,:,i_plot])), fontsize=18)
    ax01.imshow(dT3[:,:,i_plot], cmap='jet', extent=my_ext, origin='lower', vmin=dT3.min(), vmax=dT3.max())
    ax01.contour(xH[:,:,HII_DIM//2], extent=my_ext)
    fig.colorbar(im, ax=ax01, pad=0.01, fraction=0.048)

    ax01.set_ylabel('y [Mpc]', size=16)
    ax01.set_xlabel('x [Mpc]', size=16)

    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.linewidth'] = 1.2
    plt.subplots_adjust(hspace=0.3, wspace=0.15)
    plt.savefig('%slc_%dMpc_%d_i%d.png' %(path_out+'images/', BOX_LEN, dT3.shape[0], i), bbox_inches='tight')