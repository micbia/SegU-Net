import numpy as np, os, sys, json, tarfile
import tools21cm as t2c, py21cmfast as p2c 

from datetime import datetime 
from skopt.sampler import Lhs
from skopt.space import Space
from glob import glob
from sklearn.decomposition import PCA as sciPCA

path_out = sys.argv[1]
path_out += '/' if path_out[-1]!='/' else ''

print(' Starting at: %s' %(datetime.now().strftime('%H:%M:%S')))

"""
coeval_cubes = p2c.run_coeval(
    redshift = 9.0,
    user_params = {"HII_DIM": 300, "BOX_LEN": 600, "USE_INTERPOLATION_TABLES": True, 'N_THREADS': 8},
    cosmo_params = p2c.CosmoParams(SIGMA_8=0.8),
    astro_params = p2c.AstroParams({"HII_EFF_FACTOR":20.0}),
    random_seed=12345)

#coeval_cubes[0].brightness_temp_struct.save('%sbox_z9_zeta20.00_tvir5.00_rmfp15.00_21cmFast_brightness_temp_600cMpc.h5' %(path_out))
np.save('%sbox_z9_zeta20.00_tvir5.00_rmfp15.00_21cmFast_brightness_temp_600cMpc.npy' %(path_out), coeval_cubes.brightness_temp)

"""
try:
    os.makedirs(path_out)
    os.makedirs(path_out+'data')
    os.makedirs(path_out+'images')
    os.makedirs(path_out+'parameters')
except:
    pass

name_of_the_run = path_out[path_out[:-1].rfind('/')+1:-1]

# Change working directory:
os.chdir(path_out+'..')
cwd = os.getcwd()


# 21cmFAST parameters
#uvfile = '/store/ska/sk09/segunet/uvmap_128_z7-20.pkl'
#uvfile = '/store/ska/sk09/segunet/uvmap_200_z7-35.pkl'
uvfile = '/store/ska/sk02/lightcones/EOS21/uvmap_1000_z7-10.pkl'

#params = {'HII_DIM':128, 'DIM':512, 'BOX_LEN':256}
params = {'HII_DIM':200, 'DIM':600, 'BOX_LEN':300, 'USE_INTERPOLATION_TABLES': False}
c_params = {'OMm':0.27, 'OMb':0.046, 'SIGMA_8':0.82, 'POWER_INDEX':0.96}
z_min, z_max = 5.000, 29.999
tobs = 1000.
COMPRESS = False
MAKE_PLOT = False

path_cache = '/scratch/snx3000/mibianco/21cmFAST-cache/'
p2c.config['direc'] = path_cache

with open(path_out+'parameters/user_params.txt', 'w') as file:
    file.write(json.dumps(params))

with open(path_out+'parameters/cosm_params.txt', 'w') as file:
    file.write(json.dumps(c_params))


# Start loop
i, rseed = 0, 2022
print(' generated seed:\t %d' %rseed)

# Latin Hypercube Sampling of parameters
space = Space([(10., 100.), (5., 20.), (np.log10(1e4), np.log10(2e5)), (38.0, 42.), (100., 1500.)]) 
lhs_sampling = np.array(Lhs(criterion="maximin", iterations=10000).generate(dimensions=space.dimensions, n_samples=1, random_state=rseed))
eff_fact, Rmfp, Tvir, LX, E0 = lhs_sampling[0]

# Define astronomical parameters
a_params = {'HII_EFF_FACTOR':eff_fact, 'R_BUBBLE_MAX':Rmfp, 'ION_Tvir_MIN':Tvir, 'L_X': LX, 'NU_X_THRESH': E0}

print(' calculate lightcone...') 
lightcone = p2c.run_lightcone(redshift=z_min, max_redshift=z_max, 
                                user_params=params, astro_params=a_params, cosmo_params=c_params, 
                                lightcone_quantities=("brightness_temp", 'xH_box'), 
                                flag_options={"USE_TS_FLUCT": True},
                                global_quantities=("brightness_temp", 'xH_box'), 
                                direc=path_cache, random_seed=rseed) 

np.savetxt('%slc_redshifts.txt' %(path_out), lightcone.lightcone_redshifts, fmt='%.5f')
dT = lightcone.brightness_temp
t2c.save_cbin(path_out+'data/xHI_21cm_i%d.bin' %i, lightcone.xH_box)
t2c.save_cbin(path_out+'data/dT_21cm_i%d.bin' %i, dT)

lc_noise = t2c.noise_lightcone(ncells=lightcone.brightness_temp.shape[0], 
                                zs=lightcone.lightcone_redshifts, 
                                obs_time=tobs, save_uvmap=uvfile, 
                                boxsize=params['BOX_LEN'], n_jobs=1)

#print(' calculate foregrounds...')
gal_fg = t2c.galactic_synch_fg(z=lightcone.lightcone_redshifts, ncells=params['HII_DIM'], boxsize=params['BOX_LEN'], rseed=rseed)
exgal_fg = t2c.extragalactic_pointsource_fg(z=lightcone.lightcone_redshifts, ncells=params['HII_DIM'], boxsize=params['BOX_LEN'], rseed=rseed)

print(' calculate dT and mask...') 
dT1 = t2c.subtract_mean_signal(lightcone.brightness_temp, los_axis=2)  
dT2, redshifts = t2c.smooth_lightcone(dT1, z_array=lightcone.lightcone_redshifts, box_size_mpc=params['BOX_LEN']) 
dT3, _ = t2c.smooth_lightcone(t2c.subtract_mean_signal(dT + lc_noise, los_axis=2), z_array=lightcone.lightcone_redshifts, box_size_mpc=params['BOX_LEN']) 
dT4, _ = t2c.smooth_lightcone(t2c.subtract_mean_signal(dT + lc_noise + gal_fg, los_axis=2), z_array=lightcone.lightcone_redshifts, box_size_mpc=params['BOX_LEN'])
dT5, _ = t2c.smooth_lightcone(t2c.subtract_mean_signal(dT + lc_noise + exgal_fg + gal_fg, los_axis=2), z_array=lightcone.lightcone_redshifts, box_size_mpc=params['BOX_LEN'])

print(' calculate PCA...')
data = dT5
data_flat = np.reshape(data, (-1, data.shape[2]))
pca = sciPCA(n_components=7)
datapca = pca.fit_transform(data_flat)
pca_FG = pca.inverse_transform(datapca)
dT5pca = np.reshape(data_flat - pca_FG, (params['HII_DIM'], params['HII_DIM'], data.shape[2]))

#xHI = lightcone.xH_box
smt_xn, _ = t2c.smooth_lightcone(lightcone.xH_box, z_array=lightcone.lightcone_redshifts, box_size_mpc=params['BOX_LEN']) 
mask_xH = smt_xn>0.5

print(' save outputs...') 
t2c.save_cbin(path_out+'data/xH_21cm_i%d.bin' %i, mask_xH)
t2c.save_cbin(path_out+'data/dT2_21cm_i%d.bin' %i, dT2) # smooth(dT - avrg_dT)
t2c.save_cbin(path_out+'data/dT3_21cm_i%d.bin' %i, dT3) # smooth(dT + noise - avrg_dT)
t2c.save_cbin(path_out+'data/dT4_21cm_i%d.bin' %i, dT4) # smooth(dT + noise + gf - avrg_dT)
t2c.save_cbin(path_out+'data/dT5_21cm_i%d.bin' %i, dT5)  # smooth(dT + noise + gf + exgf - avrg_dT)
np.save(path_out+'data/dTexgf_21cm_i%d.npy' %i, exgal_fg[..., 0]) # just the point sourcess
t2c.save_cbin(path_out+'data/dT5pca_21cm_i%d.bin' %i, dT5pca)

# save parameters values
with open('%sastro_params.txt' %(path_out+'parameters/'), 'a') as f:
    f.write('# HII_EFF_FACTOR: The ionizing efficiency of high-z galaxies\n')
    f.write('# R_BUBBLE_MAX: Mean free path in Mpc of ionizing photons within ionizing regions\n')
    f.write('# ION_Tvir_MIN: Minimum virial Temperature of star-forming haloes in log10 units\n')
    f.write('#i\teff_f\tRmfp\tTvir\tseed\n')
    f.write('%d\t%.3f\t%.3f\t%.3f\t%d\n' %(i, eff_fact, Rmfp, Tvir, rseed))

#os.system('rm -r %s' %path_cache)
print('... done at %s.' %(datetime.now().strftime('%H:%M:%S')))
