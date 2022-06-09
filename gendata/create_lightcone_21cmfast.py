import matplotlib as mpl
mpl.use('agg')
import numpy as np, matplotlib.pyplot as plt, os, sys
import tools21cm as t2c, py21cmfast as p21
import matplotlib.gridspec as gridspec
import random, json

from datetime import datetime, date
from glob import glob
from tqdm import tqdm

def GenerateSeed():
    # create seed for 21cmFast
    seed = [var for var in datetime.now().strftime('%d%H%M%S')]
    np.random.shuffle(seed)
    return int(''.join(seed))

user_params = {'HII_DIM':128, 'DIM':384, 'BOX_LEN':256}
cosmo_params = {'OMm':0.27, 'OMb':0.046, 'SIGMA_8':0.82, 'POWER_INDEX':0.96}

uvfile = '/cosma6/data/dp004/dc-bian1/uvmap_128_z7-20.pkl'
tobs = 1000


if not (os.path.exists('%sastro_params_rank%d.txt' %(path_out+'parameters/', rank))):
    loop_resume = 0
    
    with open(path_out+'parameters/user_params.txt', 'w') as file:
        file.write(json.dumps(params))

    with open(path_out+'parameters/cosm_params.txt', 'w') as file:
        file.write(json.dumps(c_params))
else:
    loop_resume = int(np.loadtxt('%sastro_params_rank%d.txt' %(path_out+'parameters/', rank))[:,0].max())



path_out = sys.argv[1]
#rank = int(sys.argv[2])
rank = int(os.environ['SLURM_ARRAY_TASK_ID'])

path_out += '/' if path_out[-1]!='/' else ''

path_chache = p21.config['direc']

loop_start, loop_end = 0, 7000
#loop_start, loop_end = np.loadtxt('parameters/todo_r%d.txt' %rank, dtype=int)


def get_dir_size(dir):
    """Returns the "dir" size in bytes."""
    total = 0
    try:
        for entry in os.scandir(dir):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except NotAdirError:
        return os.path.getsize(dir)
    except PermissionError:
        return 0
    return total


if(loop_resume == 0):
    i = int(loop_start)
elif(loop_resume != 0):
    print(' Rank=%d resumes itration from i=%d.' %(rank, loop_resume))
    i = int(loop_resume + 1)


while i < i_end:
    # astronomical & cosmological parameters
    eff_fact = random.gauss(52.5, 20.)  # eff_fact = [5, 100]
    Rmfp = random.gauss(12.5, 5.)       # Rmfp = [5, 20]
    Tvir = random.gauss(4.65, 0.5)      # Tvir = [log10(1e4), log10(2e5)]
    
    # Define astronomical parameters
    a_params = {'HII_EFF_FACTOR':eff_fact, 'R_BUBBLE_MAX':Rmfp, 'ION_Tvir_MIN':Tvir}

    # lightcone data
    lightcone = p21c.run_lightcone(redshift=self.z_min, max_redshift=self.z_max, astro_params=astro_params, cosmo_params=self.cosmo_params, user_params=self.user_params, lightcone_quantities=("brightness_temp", 'xH_box'), direc=p21c.config['direc'], random_seed=seed, cleanup=True)

    dataLC_dTb = lightcone.brightness_temp
    redshiftLC = lightcone.lightcone_redshifts
    dataLC_dTb1 = t2c.subtract_mean_signal(dataLC_dTb, los_axis=2)

    noise_cone = t2c.noise_lightcone(ncells=dataLC_dTb1.shape[0], zs=redshiftLC, obs_time=tobs, boxsize=user_params['BOX_LEN'], n_jobs=1, save_uvmap=uvfile)
    dataLC_dTb3, redshiftLC3 = t2c.smooth_lightcone(lightcone=dataLC_dTb1+noise_cone, z_array=redshiftLC, box_size_mpc=user_params['BOX_LEN'])

    dataLC_xH = lightcone.xH_box
    mask_LCxH = (LC_xH>0.5).astype(int)

    for j in range(len(dataLC_dTb.shape[-1])):

    # Mean neutral fraction
    xn = np.mean(cube.xH_box)


    dT = cube.brightness_temp
    dT1 = t2c.subtract_mean_signal(signal=dT, los_axis=2)
    
    # mask are saved such that 1 in neutral region and 0 in ionized region
    t2c.save_cbin(path_out+'data/xH_21cm_i%d.bin' %i, cube.xH_box)
    t2c.save_cbin(path_out+'data/dT1_21cm_i%d.bin' %i, dT1)
    # save parameters values
    with open('%sastro_params_rank%d.txt' %(path_out+'parameters/', rank), 'a') as f:
        if(i == 0 and rank == 0):
            f.write('# HII_EFF_FACTOR: The ionizing efficiency of high-z galaxies\n')
            f.write('# R_BUBBLE_MAX: Mean free path in Mpc of ionizing photons within ionizing regions\n')
            f.write('# ION_Tvir_MIN: Minimum virial Temperature of star-forming haloes in log10 units\n')
            f.write('#i\tz\teff_f\tRmfp\tTvir\tx_n\n')

        f.write('%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' %(i, z, eff_fact, Rmfp, Tvir, np.mean(xn)))
    
    # if output dir is more than 15 GB of size, compress and remove files in data/
    if(get_dir_size(path_out) / 1e9 >= 15):
        if(rank == 0):
            try:
                strd = np.loadtxt(path_out+'written.txt', dtype=str, delimiter='\n')
            except:
                strd = np.array([])

            os.system('tar -czvf %s_part%d.tar.gz %s' %(path_out[path_out[:-1].rfind('/')+1:-1], strd.size+1, path_out[path_out[:-1].rfind('/')+1:-1]))
            os.system('rm %sdata/*.bin' %path_out)

            np.savetxt(path_out+'written.txt', np.append(strd, ['%s written %s_part%d.tar.gz' %(datetime.now().strftime('%d/%m/%Y %H:%M:%S'), path_out[path_out[:-1].rfind('/')+1:-1], strd.size+1)]), delimiter='\n', fmt='%s')
        print(' \n Data created exeed 15GB. Compression completed...')

    # update while loop index
    i += 1

print('... finished' )
