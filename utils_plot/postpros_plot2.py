import numpy as np, matplotlib.pyplot as plt, os
import matplotlib.ticker as ticker
from glob import glob

from sys import argv

plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.1
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = False
plt.rcParams['ytick.right'] = True

script, path = argv
os.chdir(path)

# Load Data
name_val_metric = np.sort(glob('val_*.txt')) 
name_metric = [sf[4:] for sf in name_val_metric] 
epoch = int(name_metric[0][name_metric[0].rfind('-')+1:name_metric[0].rfind('.')]) 

loss, val_loss = np.loadtxt('loss_ep-%d.txt' %epoch), np.loadtxt('val_loss_ep-%d.txt' %epoch)
zloss_img, val_loss_img = np.loadtxt('output_img_loss_ep-%d.txt' %epoch), np.loadtxt('val_output_img_loss_ep-%d.txt' %epoch)
loss_rec, val_loss_rec = np.loadtxt('output_rec_loss_ep-%d.txt' %epoch), np.loadtxt('val_output_rec_loss_ep-%d.txt' %epoch)

lr = np.loadtxt('lr_ep-%d.txt' %epoch)

idx_best_mode = np.argmin(val_loss)

# Plot
fig1 = plt.figure(figsize=(16, 6)) 
fig1.subplots_adjust(hspace=0.3, wspace=0.25, top=0.99, bottom=0.08, right=0.955, left=0.045) 

ax1 = plt.subplot(1,2,1)
ax1.set_ylabel('Loss functions'), ax1.set_xlabel('Epoch') 
ax1.semilogy(val_loss, color='cornflowerblue', label='Validation Loss', ls='--')  
ax1.semilogy(val_loss_img, color='cornflowerblue', label='val loss img', ls='--',  marker='o')  
ax1.semilogy(val_loss_rec, color='cornflowerblue', label='val loss params', ls='--',  marker='^')  
ax1.semilogy(loss, color='navy', label='Training Loss') 
ax1.semilogy(loss_img, color='navy', label='loss img', marker='o') 
ax1.semilogy(loss_rec, color='navy', label='loss params', marker='^') 
ax1.scatter(idx_best_mode, val_loss[idx_best_mode], marker="x", color="r", label="Best Model: %.3e" %(np.min(val_loss)))
plot_min, plot_max = np.min([loss.min(), val_loss.min()])*0.9, np.min([loss.max(), val_loss.max()])
ax1.set_xlim(-1, loss.size), #ax1.set_ylim(plot_min, plot_max)
ax1.set_ylim(5e-2, 0.5)

ax3 = ax1.twinx() 
ax3.semilogy(lr, color='k', alpha=0.4, label='Learning Rate') 
ax3.set_ylabel('Learning Rate') 
lns, labs   = ax1.get_legend_handles_labels() 
lns2, labs2 = ax3.get_legend_handles_labels() 
ax1.legend(lns+lns2, labs+labs2, loc='best') 
ax2 = plt.subplot(1,2,2) 
ax2.set_ylabel('Accuracy'), ax2.set_xlabel('Epoch')


#metric, val_metric = np.loadtxt('r2score_ep-%d.txt' %epoch), np.loadtxt('val_r2score_ep-%d.txt' %epoch)
metric_img, val_metric_img = np.loadtxt('output_img_r2score_ep-%d.txt' %epoch), np.loadtxt('val_output_img_r2score_ep-%d.txt' %epoch)
metric_rec, val_metric_rec = np.loadtxt('output_rec_r2score_ep-%d.txt' %epoch), np.loadtxt('val_output_rec_r2score_ep-%d.txt' %epoch)
metric, val_metric = 0.5*(metric_img + metric_rec), 0.5*(val_metric_img + val_metric_rec)
ax2.plot(metric, color='darkorange', label=r'$R^2$')
ax2.plot(metric_img, color='darkorange', label=r'$R^2$ img', marker='o')
ax2.plot(metric_rec, color='darkorange', label=r'$R^2$ params', marker='^')
ax2.plot(val_metric, color='tab:orange', ls='--')
ax2.plot(val_metric_img, color='tab:orange', ls='--',  marker='o')
ax2.plot(val_metric_rec, color='tab:orange', ls='--',  marker='^')
ax2.scatter(idx_best_mode, val_metric[idx_best_mode], marker="x", color="r")

ax2.set_xlim(-1,loss.size), ax2.set_ylim(0.5, 0.98)
#ax2.set_ylim(0., 1.0)
ax4 = ax2.twinx() 
ax4.semilogy(lr, color='k', alpha=0.4, label='Learning Rate') 
ax4.set_ylabel('Learning Rate') 
#lns2, labs2 = ax4.get_legend_handles_labels() 
#lns, labs = ax2.get_legend_handles_labels()
#ax2.legend(lns+lns2, labs+labs2, loc='best') 
ax2.legend(loc='best')

ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax2.xaxis.set_major_locator(ticker.MultipleLocator(20))
#ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
#ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

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
print('\nBest loss (epoch=%d):\t%.3e' %(idx_best_mode+1, np.min(val_loss)))
print('Validation\tAccuracy\tLoss\n images:\t%.2f%%\t\t%.4e\n params:\t%.2f%%\t\t%.4e\n' %(100*val_metric_img[idx_best_mode], val_loss_img[idx_best_mode], 100*val_metric_rec[idx_best_mode], val_loss_rec[idx_best_mode]))
#%(100*np.max(val_metric_img), 100*np.max(val_metric_rec)))

plt.savefig('loss.png', bbox_inches='tight')
#plt.show()
