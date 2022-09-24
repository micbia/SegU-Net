import numpy as np, matplotlib.pyplot as plt, os
import matplotlib.gridspec as gridspec
import py21cmfast as p2c

p2c.config['direc'] = '/scratch/snx3000/mibianco/21cmFAST-cache'

rseed = 2022
z_min, z_max = 7, 12
zeta, Rmfp, Tvir = 39.204, 12.861, 4.53
LX, Tvir_X = 42., 5.

params = {'HII_DIM':128, 'DIM':512, 'BOX_LEN':256}
#params = {'HII_DIM':128, 'DIM':512, 'BOX_LEN':256, 'USE_2LPT': False, 'USE_INTERPOLATION_TABLES': True, 'MINIMIZE_MEMORY': True}
#params = {'HII_DIM':200, 'DIM':600, 'BOX_LEN':300}
a_params = {'HII_EFF_FACTOR':zeta, 'R_BUBBLE_MAX':Rmfp, 'ION_Tvir_MIN':Tvir, 'L_X':LX, 'X_RAY_Tvir_MIN':Tvir_X}
c_params = {'OMm':0.27, 'OMb':0.046, 'SIGMA_8':0.82, 'POWER_INDEX':0.96}

tobs = 1000

lightcone = p2c.run_lightcone(redshift=z_min, max_redshift=z_max,
                              user_params=params,
                              astro_params=a_params,
                              cosmo_params=c_params,
                              flag_options={"USE_TS_FLUCT": True},
                              lightcone_quantities=("brightness_temp", 'xH_box'),
                              global_quantities=("brightness_temp", 'xH_box'), 
                              random_seed=rseed,
                              write=True)

path = '/scratch/snx3000/mibianco/test_segunet/'
np.save(path+'test_lc.npy', lightcone.brightness_temp)
print(lightcone.brightness_temp.shape)
print(lightcone.lightcone_dimensions, 'cMpc')
print(lightcone.lightcone_redshifts.min(), lightcone.lightcone_redshifts.max())
