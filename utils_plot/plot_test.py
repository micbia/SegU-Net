import numpy as np, matplotlib.pyplot as plt, os
import matplotlib.gridspec as gridspec
import tools21cm as t2c
from matplotlib import colors

from other_utils import MidpointNormalize

import sys
sys.path.append('../')
from utils_network.data_generator import LightConeGenerator, LightConeGenerator_SERENEt, LightConeGenerator_FullSERENEt

path_input = '/jmain02/home/J2AD005/jck02/mxb47-jck02/data/inputs/dataLC_128_train_060921_untar/'
redshifts = np.loadtxt(path_input+'lc_redshifts.txt')

dg = LightConeGenerator(path=path_input, data_temp=np.arange(64), data_shape=(128, 128), batch_size=2, shuffle=True)
data = dg.__getitem__(0)
dT2 = data[0]
mask_xn = data[1]
dT3 = data[2]

dT2, dT3, mask_xn = dT2[0].squeeze(), dT3[0].squeeze(), mask_xn[0].squeeze()

print(dT2.shape, mask_xn.shape, dT3.shape)
#mask_xn = t2c.read_cbin(path_input+'data/xH_21cm_i0.bin')
#dT2 = t2c.read_cbin(path_input+'data/dT2_21cm_i0.bin')
#dT3 = t2c.read_cbin(path_input+'data/dT3_21cm_i0.bin')
#dT3_wdg = t2c.read_cbin(path_input+'data/dT3wdg_21cm_i0.bin')

#i_plot = lightcone.brightness_temp.shape[-1]//2
i_plot=64
my_ext1 = [redshifts.min(), redshifts.max(), 0, 256.]
my_ext2 = [0, 256., 0, 256.]
#my_ext1 =[t2c.z_to_nu(redshifts).max(), t2c.z_to_nu(redshifts).min(), -1.6402513488058277/2, 1.6402513488058277/2]
#my_ext2 = [-1.6402513488058277/2, 1.6402513488058277/2, -1.6402513488058277/2, 1.6402513488058277/2]

fig = plt.figure(figsize=(28, 18))
gs = gridspec.GridSpec(nrows=3, ncols=2, width_ratios=[3,1], height_ratios=[1, 1, 1])

# FIRST LC PLOT
ax0 = fig.add_subplot(gs[0,0])
im = ax0.imshow(dT3[:,params['HII_DIM']//2,:], cmap='jet', aspect='auto', origin='lower', extent=my_ext1, norm=MidpointNormalize(vmin=dT3[:,:,i_plot].min(), vmax=dT3[:,:,i_plot].max(), midpoint=0))
ax0.contour(mask_xn[:,params['HII_DIM']//2,:], extent=my_ext1)

ax01 = fig.add_subplot(gs[0,1])
ax01.set_title('$z$ = %.3f   $x^v_{HI}$=%.3f' %(redshifts[i_plot], np.mean(mask_xn[:,:,i_plot])), fontsize=18)
#ax01.set_title(r'$\nu_{obs}$ = %d MHz   $SNR^{noise}$=%.3f' %(t2c.z_to_nu(redshifts[i_plot]), (np.std(dT2[:,:,i_plot])/np.std(dT3[:,:,i_plot]))**2), fontsize=18)
ax01.imshow(dT3[:,:,i_plot], cmap='jet', origin='lower', extent=my_ext2, norm=MidpointNormalize(vmin=dT3[:,:,i_plot].min(), vmax=dT3[:,:,i_plot].max(), midpoint=0))
ax01.contour(mask_xn[:,:,i_plot], extent=my_ext2)
fig.colorbar(im, ax=ax01, pad=0.01, fraction=0.048)

# SECOND LC PLOT
ax1 = fig.add_subplot(gs[1,0])
#ax1.imshow(dT3_wdg[:,params['HII_DIM']//2,:], cmap='jet', aspect='auto', origin='lower', extent=my_ext1, norm=MidpointNormalize(vmin=dT2_smt.min(), vmax=dT2_smt.max(), midpoint=0))
ax1.imshow(dT3_wdg[:,params['HII_DIM']//2,:], cmap='jet', aspect='auto', origin='lower', extent=my_ext1, norm=MidpointNormalize(vmin=dT3_wdg[:,:,i_plot].min(), vmax=dT3_wdg[:,:,i_plot].max(), midpoint=0))
ax1.contour(mask_xn[:,params['HII_DIM']//2,:], extent=my_ext1)

ax11 = fig.add_subplot(gs[1,1])
#ax11.set_title(r'$\nu_{obs}$ = %d MHz   $SNR^{wedge}$=%.3f' %(t2c.z_to_nu(redshifts[i_plot]), (np.std(dT2[:,:,i_plot])/np.std(dT3_wdg[:,:,i_plot]))**2), fontsize=18)
#ax11.imshow(mask_xn[:,:,i_plot], cmap='jet', extent=my_ext, origin='lower', vmin=mask_xn.min(), vmax=mask_xn.max())
#im = ax11.imshow(dT3_wdg[:,:,i_plot], cmap='jet', origin='lower', extent=my_ext2, norm=MidpointNormalize(vmin=dT2_smt.min(), vmax=dT2_smt.max(), midpoint=0))
im = ax11.imshow(dT3_wdg[:,:,i_plot], cmap='jet', origin='lower', extent=my_ext2, norm=MidpointNormalize(vmin=dT3_wdg[:,:,i_plot].min(), vmax=dT3_wdg[:,:,i_plot].max(), midpoint=0))
ax11.contour(mask_xn[:,:,i_plot], extent=my_ext2)
fig.colorbar(im, ax=ax11, pad=0.01, fraction=0.048)


# THIRD LC PLOT
ax2 = fig.add_subplot(gs[2,0])
ax2.imshow(dT2[:,params['HII_DIM']//2,:], cmap='jet', origin='lower', aspect='auto', extent=my_ext1, norm=MidpointNormalize(vmin=dT2.min(), vmax=dT2.max(), midpoint=0))
#ax2.contour(mask_xn[:,:,i_plot], extent=my_ext2)
#im = ax2.imshow(lightcone.xH_box[:,params['HII_DIM']//2,:], cmap='jet', origin='lower')
ax21 = fig.add_subplot(gs[2,1])

#ax21.set_title(r'$\nu_{obs}$ = %d MHz  $<\delta T_b>^{1/2}$ = %.3f mK' %(t2c.z_to_nu(redshifts[i_plot]), np.std(dT2[:,:,i_plot])), fontsize=18)
#ax21.set_title('$z$ = %.3f   $x^v_{HI}$=%.3f' %(redshifts[i_plot], np.mean(lightcone.xH_box[:,:,i_plot])), fontsize=18)
im = ax21.imshow(dT2[:,:,i_plot], cmap='jet', extent=my_ext2, origin='lower', norm=MidpointNormalize(vmin=dT2[:,:,i_plot].min(), vmax=dT2[:,:,i_plot].max(), midpoint=0))
#ax21.contour(mask_xn[:,:,i_plot], extent=my_ext2)
#ax21.imshow(lightcone.xH_box[:,:,i_plot], cmap='jet', extent=my_ext, origin='lower', vmin=lightcone.xH_box.min(), vmax=lightcone.xH_box.max())
fig.colorbar(im, ax=ax21, pad=0.01, fraction=0.048)


for ax in [ax0, ax1, ax2]:
    #ax.set_ylabel('Dec [deg]', size=20)
    #ax.set_xlabel(r'$\nu_{\rm obs}$ [MHz]', size=20)
    ax.set_xlabel('z', size=16)
    ax.set_ylabel('x [Mpc]', size=16)

for ax in [ax01, ax11, ax21]:
    #ax.set_ylabel('Dec [deg]', size=20)
    #ax.set_xlabel('RA [deg]', size=20)

    ax.set_ylabel('y [Mpc]', size=16)
    ax.set_xlabel('x [Mpc]', size=16)

plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.2
plt.subplots_adjust(hspace=0.3, wspace=0.01)
plt.savefig('lc_256Mpc_128.png' , bbox_inches='tight')
