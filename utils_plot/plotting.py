import matplotlib
matplotlib.use('agg')
import numpy as np, tools21cm as t2c
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .other_utils import PercentContours
from sklearn.metrics import matthews_corrcoef
from glob import glob
from tqdm import tqdm

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


def plot_lc(i, path_out):
    plt.rcParams['font.size'] = 20
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 1.2

    redshift = np.loadtxt(path_out+'lc_redshifts.txt')
    idx, eff_fact, Rmfp, Tvir, seed = np.loadtxt(path_out+'parameters/astro_params.txt', unpack=True)
    zeta, Rmfp, Tvir = eff_fact[i], Rmfp[i], Tvir[i]

    with open(path_out+'parameters/user_params.txt','r') as f:
        user_par = eval(f.read())
        BOX_LEN = user_par['BOX_LEN']
        HII_DIM = user_par['HII_DIM']
        angl_scale = np.linspace(0, BOX_LEN, HII_DIM)

    xH = t2c.read_cbin('%sdata/xH_21cm_i%d.bin' %(path_out, i)) 
    dT_input = t2c.read_cbin('%sdata/dT4pca4_21cm_i%d.bin' %(path_out, i))

    mean_xH = np.mean(xH, axis=(0,1))
    i_z, i_angl = np.argmin(np.abs(0.5-mean_xH)), HII_DIM//2

    fig = plt.figure(figsize=(35, 8))
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[3,1], height_ratios=[1])

    # FIRST LC PLOT
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title('$\zeta$ = %.3f   $R_{mfp}$ = %.3f Mpc   $log_{10}(T_{vir}^{min})$ = %.3f' %(zeta, Rmfp, Tvir))
    im = ax0.pcolormesh(redshift, angl_scale, dT_input[:,i_angl,:], cmap='jet')
    ax0.contour(redshift, angl_scale, xH[:,i_angl,:])

    ax0.set_xlabel('z'), ax0.set_ylabel('x [Mpc]')
    ax0.tick_params(axis='both', length=5, width=1.2)
    ax0.tick_params(which='minor', axis='both', length=5, width=1.2)

    # SECOND LC PLOT
    ax01 = fig.add_subplot(gs[0,1])
    ax01.set_title('$z$=%.3f   $\overline{x}_{HI}$=%.3f' %(redshift[i_z], mean_xH[i_z]))
    ax01.pcolormesh(angl_scale, angl_scale, dT_input[:,:,i_z], cmap='jet', vmin=dT_input.min(), vmax=dT_input.max())
    ax01.contour(angl_scale, angl_scale, xH[:,:,i_z])
    fig.colorbar(im, ax=ax01, pad=0.01, fraction=0.048)

    ax01.set_ylabel('y [Mpc]'), ax01.set_xlabel('x [Mpc]')

    plt.subplots_adjust(hspace=0.3, wspace=0.15)
    plt.savefig('%slc_%dMpc_%d_i%d.png' %(path_out+'images/', BOX_LEN, HII_DIM, i), bbox_inches='tight')


def plot_dataset(path_out):
    plt.rcParams['font.size'] = 18

    arr_size = glob('%sstats_i*.txt' %path_out)

    redshift, mean_true, mcc = [], [], []
    for i, fname in enumerate(arr_size):
        if(i % 15 == 0):
            try:
                red, one_mcc, one_mean = np.loadtxt(fname, usecols=(0, 5, 9), unpack=True)
            except:
                red, one_mcc, one_mean = np.loadtxt(fname, usecols=(0, 5, 7), unpack=True)
            if(i == 0):
                mcc = one_mcc
                redshift = red
                mean_true = one_mean
            else:
                mcc = np.hstack((mcc, one_mcc))
                redshift = np.hstack((redshift, red))
                mean_true = np.hstack((mean_true, one_mean))

    print(np.shape(redshift))
    redshift, mcc, mean_true = np.array(redshift), np.array(mcc), np.array(mean_true)

    fig, ax = plt.subplots(figsize=(10,8), ncols=1)
    sc = ax.scatter(mean_true, mcc, c=redshift, vmin=redshift.min(), vmax=redshift.max(), s=25, cmap='plasma', marker='.')
    PercentContours(x=mean_true, y=mcc, bins='lin', colour='lime', style=['--', '-'], perc_arr=[0.95, 0.68])
    ax.hlines(y=np.mean(mcc), xmin=0, xmax=1, ls='--', label=r'$r_{\phi}$ = %.3f' %(np.mean(mcc)), alpha=0.8, color='tab:blue', zorder=3)
    
    plt.legend(loc=1)
    ax.set_xlim(mean_true.min()-0.02, mean_true.max()+0.02), ax.set_xlabel(r'$\rm x^v_{HI}$', size=20)
    ax.set_ylim(0, 1), ax.set_ylabel(r'$\rm r_{\phi}$', size=20)
    ax.set_yticks(np.arange(0, 1.1, 0.1)), ax.set_xticks(np.arange(0, 1.1, 0.2))
    fig.colorbar(sc, ax=ax, pad=0.01, label=r'$\rm z$')
    fig.savefig('%smcc_dataset.png' %path_out, bbox_inches='tight')
    
