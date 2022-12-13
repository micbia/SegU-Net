import numpy as np
import tools21cm as t2c

rseed = 2022
params = {'HII_DIM':1000, 'BOX_LEN':1500.}
path_in = '/store/ska/sk02/lightcones/EOS21/'
path_out = '/scratch/snx3000/mibianco/test_segunet/'

print('load data')
dT = np.load(path_in+'dT_EOS21_EoR.npy')
xHI = np.load(path_in+'xHI_EOS21_EoR.npy')
redshift = np.loadtxt(path_in+'redshift_EOS21_EoR.txt')

print('lc noise')
uvfile = '/store/ska/sk02/lightcones/EOS21/uvmap_1000_z7-10.pkl'
tobs = 1000
lc_noise = t2c.noise_lightcone(ncells=dT.shape[0], zs=redshift, obs_time=tobs, boxsize=params['BOX_LEN'], save_uvmap=uvfile, n_jobs=1)

dT1 = t2c.subtract_mean_signal(dT, los_axis=2) 
dT2, redshifts = t2c.smooth_lightcone(dT1, z_array=redshift, box_size_mpc=params['BOX_LEN'])
dT3, _ = t2c.smooth_lightcone(dT1 + lc_noise, z_array=redshifts, box_size_mpc=params['BOX_LEN'])

gal_fg = t2c.galactic_synch_fg(z=redshift, ncells=params['HII_DIM'], boxsize=params['BOX_LEN'], rseed=rseed)
#exgal_fg = t2c.extragalactic_pointsource_fg(z=lightcone.lightcone_redshifts, ncells=params['HII_DIM'], boxsize=params['BOX_LEN'])
np.save(path_out+'dTexfrg_EOS21_EoR.npy', gal_fg)

#dT3wdg = t2c.rolling_wedge_removal_lightcone(lightcone=dT3, redshifts=redshifts)
dT4, _ = t2c.smooth_lightcone(t2c.subtract_mean_signal(dT+gal_fg+lc_noise, los_axis=2), z_array=redshift, box_size_mpc=params['BOX_LEN'])
#dT3exgf, _ = t2c.smooth_lightcone(t2c.subtract_mean_signal(dT+exgal_fg, los_axis=2), z_array=redshfit, box_size_mpc=params['BOX_LEN'])
#dT3f, _ = t2c.smooth_lightcone(t2c.subtract_mean_signal(dT1+exgal_fg+gal_fg, los_axis=2), z_array=redshfit, box_size_mpc=params['BOX_LEN'])
#dT3fn, _ = t2c.smooth_lightcone(t2c.subtract_mean_signal(dT+lc_noise+exgal_fg+gal_fg, los_axis=2), z_array=redshfit, box_size_mpc=params['BOX_LEN'])

#dT3tot = t2c.rolling_wedge_removal_lightcone(lightcone=dTfg, redshifts=redshifts)

smt_xn, redshifts = t2c.smooth_lightcone(xHI, z_array=redshift, box_size_mpc=params['BOX_LEN'])
mask_xH = smt_xn>0.5

#t2c.save_cbin(path_out+'dT_21cm.bin', dT)
t2c.save_cbin(path_out+'dT2_21cm.bin', dT2)
t2c.save_cbin(path_out+'dT3_21cm.bin', dT3)
#t2c.save_cbin(path_out+'dT3wdg_21cm.bin', dT3wdg)
#t2c.save_cbin(path_out+'dT3gf_21cm.bin', dT3gf)
#t2c.save_cbin(path_out+'dT3exgf_21cm.bin', dT3exgf)
#t2c.save_cbin(path_out+'dT3f_21cm.bin', dT3f)
#t2c.save_cbin(path_out+'dT3fn_21cm.bin', dT3fn)
t2c.save_cbin(path_out+'dT4_21cm.bin', dT4)
t2c.save_cbin(path_out+'xHI_21cm.bin', xHI)
t2c.save_cbin(path_out+'xH_21cm.bin', mask_xH)
redshifts = np.savetxt(path_out+'lc_redshifts.txt', redshifts)