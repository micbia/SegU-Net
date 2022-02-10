import talos, pandas as pd, numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from other_utils import MergeImages

#path = '/jmain02/home/J2AD005/jck02/mxb47-jck02/SegU-Net/tests/test1/'
#r = talos.Reporting(path+'120521192455.csv')
path = '/jmain02/home/J2AD005/jck02/mxb47-jck02/SegU-Net/tests/test2/'
r = talos.Reporting(path+'120621000150.csv')
print(r.data)

best_params = r.best_params(metric='loss', exclude=['round_epochs', 'val_matthews_coef', 'val_precision', 'val_recall', 'val_iou', 'val_loss', 'matthews_coef', 'precision', 'recall', 'iou'])
print(best_params)

p = {'coarse_dim': [128, 256, 512],
      'dropout':[0.05, 0.1, 0.15],
      'kernel_size':[3, 5],
      #'activation': [ReLU(), LeakyReLU()],
      #'final_activation': ['sigmoid'], 
      #'optimizer': [Adam], 
      'depth': [3, 4]
     }

# create corrlation matrix
corr_matrix = {}
corr_matrix['loss'] = r.correlate('loss', exclude=['round_epochs', 'val_matthews_coef', 'val_precision', 'val_recall', 'val_iou', 'val_loss', 'matthews_coef', 'precision', 'recall', 'iou'])
idx_arr = ['loss']
for par in p:
    idx_arr = np.append(idx_arr, par)
    corr_matrix[par] = r.correlate(par, exclude=['round_epochs', 'val_matthews_coef', 'val_precision', 'val_recall', 'val_iou', 'val_loss', 'matthews_coef', 'precision', 'recall', 'iou'])
corr_matrix = pd.DataFrame(corr_matrix).fillna(1).reindex(idx_arr)
corr_matrix.to_csv(path+'corr_matrix.csv')

print(corr_matrix)
corr = corr_matrix.corr()
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200),square=True, annot=True, fmt=".2f")
ax.set_xticklabels(ax.get_xticklabels(),rotation=20,horizontalalignment='right')
ax.set_title('Correlation Matrix')
plt.savefig('%scorr.png' %path)

# define metrics (double check cvs file)
avail_metric = ['loss', 'matthews_coef', 'precision', 'recall', 'iou']
bar_legend = ['dropout', 'coarse_dim']

for mtr in avail_metric:
    #r.plot_hist(metric=mtr, bins=50)
    #plt.xlabel(mtr), plt.ylabel('N')
    #plt.savefig('%shist_%s.png' %(path, mtr)), plt.clf()

    r.plot_line(metric=mtr)
    plt.xlabel('hyperparameters space'), plt.ylabel(mtr)
    plt.savefig('%sline_%s.png' %(path,mtr)), plt.clf()

    r.plot_bars(x='depth', y=mtr, hue=bar_legend[0], col=bar_legend[1])
    plt.savefig('%sbars_%s-%s_%s.png' %(path, bar_legend[0], bar_legend[1], mtr))

# Plot kde for loss
hyp_par = ['coarse_dim', 'dropout', 'kernel_size', 'depth']
for hp in hyp_par:
    r.plot_kde(x='loss', y=hp)
    plt.xlabel('loss'), plt.ylabel(hp)
    plt.savefig('%skde_%s.png' %(path, hp))

arr_img = ['kde_coarse_dim.png', 'kde_dropout.png', 'kde_kernel_size.png', 'kde_depth.png']
MergeImages(new_image_name='kde', old_image_name=arr_img, output_path=path, form='v', delete_old=True)
#r.plot_corr('loss', exclude=['kernel_size', 'depth', 'round_epochs', 'val_r2score', 'val_precision', 'val_recall', 'val_iou', 'val_loss'], color_grades=10)
#r.plot_corr('loss', exclude=['round_epochs', 'val_matthews_coef', 'val_precision', 'val_recall', 'val_iou', 'val_loss', 'matthews_coef', 'precision', 'recall', 'iou'], color_grades=10)
#plt.savefig('%scorr.png' %path)

# Merge lines plot
arr_img = ['line_%s.png' %am for am in avail_metric[1:]]
MergeImages(new_image_name='lines_metric', old_image_name=arr_img, output_path=path, form=(2,2), delete_old=True)
MergeImages(new_image_name='lines', old_image_name=['lines_metric.png', 'line_loss.png'], output_path=path, form='v', delete_old=True)

# Merge bars plot
arr_img = ['bars_%s-%s_%s.png' %(bar_legend[0], bar_legend[1], am) for am in avail_metric]
MergeImages(new_image_name='bars_%s-%s' %(bar_legend[0], bar_legend[1]), old_image_name=arr_img, output_path=path, form='v', delete_old=True)

