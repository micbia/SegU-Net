import numpy as np, os, sys, json, pickle
import tools21cm as t2c, py21cmfast as p2c 
from datetime import datetime, date 
from tqdm import tqdm
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

def GenerateSeed(): 
    # create seed for 21cmFast 
    seed = [var for var in datetime.now().strftime('%d%H%M%S')] 
    np.random.shuffle(seed) 
    return int(''.join(seed)) 

path_out = sys.argv[1]
path_out += '/' if path_out[-1]!='/' else '' 
 
path_chache = '/cosma6/data/dp004/dc-bian1/_cache%d/' %rank
try: 
    os.makedirs(path_chache)
except:
    pass
p2c.config['direc'] = path_chache

try:
    os.makedirs(path_out+'data')
    os.makedirs(path_out+'images')
    os.makedirs(path_out+'parameters')
except:



# resume
if not (os.path.exists('%sastro_params_rank%d.txt' %(path_out+'parameters/', rank))):
    loop_resume = 0
    i_start = int(loop_start+rank*perrank)

    with open(path_out+'parameters/user_params.txt', 'w') as file:
        file.write(json.dumps(params))
    with open(path_out+'parameters/cosm_params.txt', 'w') as file:
        file.write(json.dumps(c_params))
else:
    loop_resume = int(np.loadtxt('%sastro_params_rank%d.txt' %(path_out+'parameters/', rank))[:,0].max())
    print(' Rank=%d resumes itration from i=%d.' %(rank, loop_resume))
    i_start = int(loop_resume + 1)

if(rank == nprocs-1):
    i_end = int(loop_start+(rank+1)*perrank)
else:
    i_end = int(loop_start+(rank+1)*perrank)
    if(i_end != loop_end):
        i_end = loop_end

# LC astrophysical and cosmological parameters 
u_params = {'HII_DIM':512, 'DIM':512, 'BOX_LEN':512} 
c_params = {'OMm':0.27, 'OMb':0.046, 'SIGMA_8':0.82, 'POWER_INDEX':0.96} 
z_min, z_max = 7, 11
eff_fact, Rmfp, Tvir, LX = 42.580, 12.861, 4.539, 32.
a_params = {'HII_EFF_FACTOR':eff_fact, 'R_BUBBLE_MAX':Rmfp, 'ION_Tvir_MIN':Tvir, 'L_X':LX} 
tobs = 1000.

for i in range(i_start, i_end):
    rseed = 2020
    print(' generated seed:\t %d' %rseed)

    # save parameters values
    with open('%sastro_params_rank%d.txt' %(path_out+'parameters/', rank), 'a') as f:
        if(i == 0 and rank == 0):
            f.write('# HII_EFF_FACTOR: The ionizing efficiency of high-z galaxies\n')
            f.write('# R_BUBBLE_MAX: Mean free path in Mpc of ionizing photons within ionizing regions\n')
            f.write('# ION_Tvir_MIN: Minimum virial Temperature of star-forming haloes in log10 units\n')
            f.write('#i\tz\teff_f\tRmfp\tTvir\tx_n\n')
            f.write('%d\t\t%.3f\t%.3f\t%.3f\t%d\n' %(i, eff_fact, Rmfp, Tvir, rseed))
        else:
            f.write('%d\t\t%.3f\t%.3f\t%.3f\t%d\n' %(i, eff_fact, Rmfp, Tvir, rseed))

    print(' calculate lightcone...') 
    #lightcone = p2c.run_lightcone(redshift=z_min, max_redshift=z_max, user_params=u_params, astro_params=a_params, cosmo_params=c_params, flag_options={"USE_TS_FLUCT": True}, lightcone_quantities=("brightness_temp", 'xH_box'), global_quantities=("brightness_temp", 'xH_box'), direc=path_chache, random_seed=rseed) 
    lightcone = p2c.run_lightcone(redshift=z_min, max_redshift=z_max, 
                                user_params=u_params, astro_params=a_params, cosmo_params=c_params, 
                                lightcone_quantities=("brightness_temp", 'xH_box'), 
                                global_quantities=("brightness_temp", 'xH_box'), 
                                direc=path_chache, random_seed=rseed) 
    
    uvfile = '/cosma6/data/dp004/dc-bian1/uvmap_128_z7-20.pkl'
    lc_noise = t2c.noise_lightcone(ncells=lightcone.brightness_temp.shape[0], 
                                zs=lightcone.lightcone_redshifts, 
                                obs_time=tobs, save_uvmap=uvfile, 
                                boxsize=u_params['BOX_LEN'], n_jobs=1)

    print(' calculate dT and mask...') 
    dT2 = t2c.subtract_mean_signal(lightcone.brightness_temp, los_axis=2)  
    dT2_smt, redshifts = t2c.smooth_lightcone(dT2, z_array=lightcone.lightcone_redshifts, box_size_mpc=u_params['BOX_LEN']) 
    dT3, redshifts = t2c.smooth_lightcone(dT2 + lc_noise, z_array=lightcone.lightcone_redshifts, box_size_mpc=u_params['BOX_LEN']) 

    smt_xn, redshifts = t2c.smooth_lightcone(lightcone.xH_box, z_array=lightcone.lightcone_redshifts, box_size_mpc=u_params['BOX_LEN']) 
    mask_xn = smt_xn>0.5 

    print(' save outputs...') 
    t2c.save_cbin(filename='%slc_%dMpc_dT3.dat' %(path_out, u_params['BOX_LEN']), data=dT3) 
    t2c.save_cbin(filename='%slc_%dMpc_mask.dat' %(path_out, u_params['BOX_LEN']), data=mask_xn) 
    """
    t2c.save_cbin(filename='%slc_%dMpc_dT.dat' %(path_out, u_params['BOX_LEN']), data=lightcone.brightness_temp) 
    t2c.save_cbin(filename='%slc_%dMpc_xH.dat' %(path_out, u_params['BOX_LEN']), data=lightcone.xH_box) 
    """
    np.savetxt('%slc_%dMpc_redshift.txt' %(path_out, u_params['BOX_LEN']), lightcone.lightcone_redshifts, fmt='%.5f')

    # delete chache
    os.system('rm %s*.h5' %path_chache)

print('... rank %d done.' %rank)
