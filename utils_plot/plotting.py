import matplotlib, tools21cm as t2c
matplotlib.use('agg')
import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef

import sys
sys.path.append("..")
from tqdm import tqdm

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