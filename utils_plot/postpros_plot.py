import numpy as np, matplotlib.pyplot as plt, os
import matplotlib.ticker as ticker

from sys import argv

plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.1
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = False
plt.rcParams['ytick.right'] = True

script, path, epoch = argv
epoch = int(epoch)

os.chdir(path)

# Load Data
loss, val_loss = np.loadtxt('loss_ep-%d.txt' %epoch), np.loadtxt('val_loss_ep-%d.txt' %epoch) 
#mse, val_mse = np.loadtxt('mean_squared_error_ep-%d.txt' %epoch), np.loadtxt('val_mean_squared_error_ep-%d.txt' %epoch) 

dice_coef, val_dice_coef = np.loadtxt('dice_coef_ep-%d.txt' %epoch), np.loadtxt('val_dice_coef_ep-%d.txt' %epoch) 
bin_acc, val_bin_acc = np.loadtxt('binary_accuracy_ep-%d.txt' %epoch), np.loadtxt('val_binary_accuracy_ep-%d.txt' %epoch) 

iou, val_iou = np.loadtxt('iou_ep-%d.txt' %epoch), np.loadtxt('val_iou_ep-%d.txt' %epoch) 

lr = np.loadtxt('lr_ep-%d.txt' %epoch) 

idx_best_mode = np.argmin(val_loss)

# Plot
fig1 = plt.figure(figsize=(16, 6), dpi = 96) 
fig1.subplots_adjust(hspace=0.3, wspace=0.25, top=0.99, bottom=0.08, right=0.955, left=0.045) 
  
ax1 = plt.subplot(1,2,1) 
ax1.set_ylabel('Loss functions'), ax1.set_xlabel('Epoch') 
ax1.scatter(idx_best_mode, val_loss[idx_best_mode], marker="x", color="r", label="Best Model")
ax1.semilogy(val_loss, color='cornflowerblue', label='Validation Dice Loss')  
ax1.semilogy(loss, color='navy', label='Training Dice Loss') 
#ax1.plot(val_mse, color='firebrick', label='Validation MSE')  
#ax1.plot(mse, color='lightcoral', label='Training MSE') 
ax1.set_xlim(-1, loss.size) 

ax3 = ax1.twinx() 
ax3.semilogy(lr, color='k', alpha=0.4, label='Learning Rate') 
ax3.set_ylabel('Learning Rate') 
lns, labs   = ax1.get_legend_handles_labels() 
lns2, labs2 = ax3.get_legend_handles_labels() 
ax1.legend(lns+lns2, labs+labs2, loc=1) 
  
ax2 = plt.subplot(1,2,2) 
ax2.set_ylabel('Accuracy'), ax2.set_xlabel('Epoch') 
#ax2.scatter(idx_best_mode, val_dice_coef[idx_best_mode], marker="x", color="r", label="Best Model")
ax2.plot(val_dice_coef, color='lightgreen', label='Validation Dice Coefficien') 
ax2.plot(dice_coef, color='forestgreen', label='Training Dice Coefficien')
ax2.scatter(idx_best_mode, val_bin_acc[idx_best_mode], marker="x", color="r", label="Best Model")
ax2.plot(val_bin_acc, color='violet', label='Validation Binary Accuracy') 
ax2.plot(bin_acc, color='purple', label='Training Binary Accuracy')
ax2.plot(val_iou, color='orange', label='Validation IoU') 
ax2.plot(iou, color='darkorange', label='Training IoU')
ax2.set_xlim(-1,loss.size) 

"""
ax4 = ax2.twinx() 
ax4.semilogy(lr, color='k', alpha=0.4, label='Learning Rate') 
ax4.set_ylabel('Learning Rate') 
lns2, labs2 = ax4.get_legend_handles_labels() 
"""
lns, labs   = ax2.get_legend_handles_labels()
ax2.legend(lns+lns2, labs+labs2, loc='best') 

ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))

ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

ax1.tick_params(axis='both', length=7, width=1.1)
ax1.tick_params(which='minor', axis='both', length=3, width=1.1)
ax2.tick_params(axis='both', length=7, width=1.1)
ax2.tick_params(which='minor', axis='both', length=3, width=1.1)
ax3.tick_params(axis='both', length=7, width=1.1)
ax3.tick_params(which='minor', axis='both', length=3, width=1.1)
"""
ax4.tick_params(axis='both', length=7, width=1.1)
ax4.tick_params(which='minor', axis='both', length=3, width=1.1)
"""

plt.savefig('loss.png', bbox_inches='tight')
plt.show()
