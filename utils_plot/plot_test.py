import numpy as np, matplotlib.pyplot as plt, os
import matplotlib.gridspec as gridspec
import tools21cm as t2c
from matplotlib import colors

import sys
sys.path.append('/jmain02/home/J2AD005/jck02/mxb47-jck02/SegU-Net')
from utils_network.data_generator import LightConeGenerator_SegRec

class MidpointNormalize(colors.Normalize):
    """
    Created by Joe Kington.
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    # set the colormap and centre the colorbar
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

class adjust_axis:
    def __init__(self, axis, varr, xy, to_round=10, step=5, fmt=int):
        self.axis = axis
        self.varr = varr
        self.to_round = to_round
        self.step = step
        self.fmt = fmt
        
        loc_f = self.get_axis_locs()
        if(xy == 'x'):
            plt.xticks(loc_f)
            axis.set_xticklabels([int(round(varr[i_n])) for i_n in loc_f])
        elif(xy == 'y'):
            plt.yticks(loc_f)
            axis.set_yticklabels([int(round(varr[i_n])) for i_n in loc_f])
        
    def get_axis_locs(self):    
        v_max = int(round(self.varr.max()/self.to_round)*self.to_round) if int(round(self.varr.max()/self.to_round)*self.to_round) <= self.varr.max() else int(round(self.varr.max()/self.to_round)*self.to_round)-self.to_round
        v_min = int(round(self.varr.min()/self.to_round)*self.to_round) if int(round(self.varr.min()/self.to_round)*self.to_round) >= self.varr.min() else int(round(self.varr.min()/self.to_round)*self.to_round)+self.to_round
        v_plot = np.arange(v_min, v_max+self.step, self.step)
        loc_v = np.array([np.argmin(abs(self.varr-v_plot[i])) for i in range(v_plot.size)]).astype(self.fmt)
        return loc_v


path_input = '/jmain02/home/J2AD005/jck02/mxb47-jck02/data/inputs/dataLC_128_train_060921_untar/'
redshifts = np.loadtxt(path_input+'lc_redshifts.txt')

dg = LightConeGenerator_SegRec(path=path_input, data_temp=np.arange(4), data_shape=(128, 128), batch_size=2, shuffle=True)
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
