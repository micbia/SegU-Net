import numpy as np, matplotlib.pyplot as plt
from glob import glob
import matplotlib

plt.rcParams['font.size'] = 18
path_out = '/scratch/snx3000/mibianco/output_segunet/outputs/all24-09T23-36-45_128slice/prediction_preprocess/'

depth_mhz = 20
ftype = 'poly'

j = 0
arr_size = glob('%sstats_i*_from_dT4%s_z*_%dMHz_i*.bin.txt' %(path_out, ftype, depth_mhz))

redshift, mean_true, mcc = [], [], []
for i, fname in enumerate(arr_size):
    red, one_mcc, one_err, one_mean = np.loadtxt(fname, usecols=(0, 5, 6, 9), unpack=True)
    if(i == 0):
        mcc = one_mcc
        mcc_err = one_err
        redshift = red
        mean_true = one_mean
    else:
        mcc = np.hstack((mcc, one_mcc))
        mcc_err = np.hstack((mcc_err, one_err)) 
        redshift = np.hstack((redshift, red))
        mean_true = np.hstack((mean_true, one_mean))

print(np.shape(redshift))
redshift, mcc, mean_true = np.array(redshift), np.array(mcc), np.array(mean_true)

fig, ax = plt.subplots(figsize=(10,8), ncols=1)
cm = matplotlib.cm.plasma
sc = ax.scatter(mean_true, mcc, c=redshift, vmin=redshift.min(), vmax=redshift.max(), s=25, cmap=cm, marker='.')
norm = matplotlib.colors.Normalize(vmin=7, vmax=9, clip=True)
mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm)
redshift_color = np.array([(mapper.to_rgba(v)) for v in redshift])
#for x, y, e, clr, red in zip(mean_true, mcc, mcc_err, redshift_color, redshift):
#    ax.errorbar(x=x, y=y, yerr=e, markersize=3.5, lw=1, marker='o', capsize=1, color=clr)

#PercentContours(x=mean_true, y=mcc, bins='lin', colour='lime', style=['--', '-'], perc_arr=[0.95, 0.68])
ax.hlines(y=np.mean(mcc), xmin=0, xmax=1, ls='--', label=r'$r_{\phi}$ = %.3f' %(np.mean(mcc)), alpha=0.8, color='tab:blue', zorder=3)

plt.legend(loc=1)
ax.set_xlim(mean_true.min()-0.02, mean_true.max()+0.02), ax.set_xlabel(r'$\rm x^v_{HI}$', size=20)
ax.set_ylim(0, 1), ax.set_ylabel(r'$\rm r_{\phi}$', size=20)
ax.set_yticks(np.arange(0, 1.1, 0.1)), ax.set_xticks(np.arange(0, 1.1, 0.2))
fig.colorbar(sc, ax=ax, pad=0.01, label=r'$\rm z$')
fig.savefig('%smcc_dataset_%s_%dMHz.png' %(path_out, ftype, depth_mhz), bbox_inches='tight')
