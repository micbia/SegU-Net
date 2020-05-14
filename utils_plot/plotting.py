import matplotlib
matplotlib.use('agg')
import numpy as np, matplotlib.pyplot as plt

from sklearn.metrics import matthews_corrcoef
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
    ax1.plot(output.history["loss"], c='tab:blue', label='train loss')
    ax1.plot(output.history["val_loss"], c='tab:green', label='validation loss')
    ax1.plot(np.argmin(output.history["val_loss"]), np.min(output.history["val_loss"]), marker="x", color="r", label="best model")
    ax1.set_xlabel('Epoch'), ax1.set_ylabel('Loss'), ax1.legend()

    ax2 = ax1.twinx()
    ax2.semilogy(output.history['lr'], c='gray', ls='--', alpha=0.5)
    ax2.set_ylabel('Learning Rate')

    plt.savefig(path+'loss.png', bbox_inches='tight')
    plt.close('all')


# Plot matthews_corrcoef
def plot_phicoef(true, predict, indexes, red, xfrac, path='./'):
    phi_coef = np.zeros_like(indexes)
    for i in tqdm(range(indexes.size)):
        phi_coef[i] = matthews_corrcoef(true[i].flatten(), predict[i].flatten().round())
    np.savetxt(path+'phi_coef.txt', np.array([indexes, red, xfrac, phi_coef]).T, fmt='%d\t%.3f\t%.3f\t%.3f', header='z\tx_n\tphi_coef')
    fig = plt.figure(figsize=(12, 12))
    plt.plot(xfrac, phi_coef, 'r.')
    plt.xlabel('$x_n$'), plt.ylabel('$r_{\phi}$')
    plt.savefig(path+'phi_coef.png', bbox_inches='tight')
    plt.close('all')