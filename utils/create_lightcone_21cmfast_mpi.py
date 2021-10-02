import numpy as np, os, sys, json, pickle
import tools21cm as t2c, py21cmfast as p2c 

from datetime import datetime, date 
from tqdm import tqdm
from mpi4py import MPI

from skopt.sampler import Lhs
from skopt.space import Space

path_out = sys.argv[1]
path_out += '/' if path_out[-1]!='/' else ''
try:
    os.makedirs(path_out)
    os.makedirs(path_out+'data')
    os.makedirs(path_out+'images')
    os.makedirs(path_out+'parameters')
except:
    pass

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

print(' Starting rank %d at: %s' %(rank, datetime.now().strftime('%H:%M:%S')))

# 21cmFAST parameters
uvfile = '/cosma6/data/dp004/dc-bian1/uvmap_128_z7-20.pkl'
params = {'HII_DIM':128, 'DIM':512, 'BOX_LEN':256} 
c_params = {'OMm':0.27, 'OMb':0.046, 'SIGMA_8':0.82, 'POWER_INDEX':0.96} 
z_min, z_max = 7, 11
tobs = 1000.
MAKE_PLOT = False

#path_cache = '/cosma6/data/dp004/dc-bian1/21cmFAST-cache/'
path_cache = '/cosma6/data/dp004/dc-bian1/_cache%d/' %rank
p2c.config['direc'] = path_cache

if not (os.path.exists(path_cache)):
    os.makedirs(path_cache)
else:
    os.system('rm %s*h5' %path_cache)

loop_start, loop_end = 0, 1500
#loop_start, loop_end = np.loadtxt('parameters/todo_r%d.txt' %arr_idx, dtype=int)
perrank = (loop_end-loop_start)//nprocs

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


def GenerateSeed():
    # create seed for 21cmFast
    seed = [var for var in datetime.now().strftime('%d%H%M%S')]
    np.random.shuffle(seed)
    return int(''.join(seed))


# Set loop starting index per processor
if not (os.path.exists('%sastro_params_rank%d.txt' %(path_out+'parameters/', rank))):
    loop_resume = 0
    i = int(loop_start+rank*perrank)

    with open(path_out+'parameters/user_params.txt', 'w') as file:
        file.write(json.dumps(params))

    with open(path_out+'parameters/cosm_params.txt', 'w') as file:
        file.write(json.dumps(c_params))
else:
    loop_resume = int(np.loadtxt('%sastro_params_rank%d.txt' %(path_out+'parameters/', rank))[:,0].max())
    
    print(' Rank=%d resumes itration from i=%d.' %(rank, loop_resume))
    i = int(loop_resume + 1)

# Set loop ending index per processor
if(rank != nprocs-1):
    i_end = int(loop_start+(rank+1)*perrank)
else:
    i_end = loop_end


print(' Processors repartition:\n rank %d\t%d\t%d' %(rank, i, i_end)) 
while i < i_end:
    rseed = GenerateSeed()
    print(' generated seed:\t %d' %rseed)

    # Latin Hypercube Sampling of parameters
    space = Space([(10., 100.), (5., 20.), (np.log10(1e4), np.log10(2e5))]) 
    lhs_sampling = np.array(Lhs(criterion="maximin", iterations=10000).generate(dimensions=space.dimensions, n_samples=1, random_state=GenerateSeed()))
    eff_fact, Rmfp, Tvir = lhs_sampling[0]

    # Define astronomical parameters
    a_params = {'HII_EFF_FACTOR':eff_fact, 'R_BUBBLE_MAX':Rmfp, 'ION_Tvir_MIN':Tvir}

    # Create 21cmFast cube
    try:
        os.system('rm %s*h5' %path_cache)
    except:
        pass

    print(' calculate lightcone...') 
    lightcone = p2c.run_lightcone(redshift=z_min, max_redshift=z_max, 
                                  user_params=params, astro_params=a_params, cosmo_params=c_params, 
                                  lightcone_quantities=("brightness_temp", 'xH_box'), 
                                  #flag_options={"USE_TS_FLUCT": True},
                                  global_quantities=("brightness_temp", 'xH_box'), 
                                  direc=path_cache, random_seed=rseed) 
    
    lc_noise = t2c.noise_lightcone(ncells=lightcone.brightness_temp.shape[0], 
                                   zs=lightcone.lightcone_redshifts, 
                                   obs_time=tobs, save_uvmap=uvfile, 
                                   boxsize=params['BOX_LEN'], n_jobs=1)

    print(' calculate dT and mask...') 
    dT2 = t2c.subtract_mean_signal(lightcone.brightness_temp, los_axis=2)  
    #dT2_smt, redshifts = t2c.smooth_lightcone(dT2, z_array=lightcone.lightcone_redshifts, box_size_mpc=params['BOX_LEN']) 
    dT3, redshifts = t2c.smooth_lightcone(dT2+lc_noise, z_array=lightcone.lightcone_redshifts, box_size_mpc=params['BOX_LEN']) 
    smt_xn, redshifts = t2c.smooth_lightcone(lightcone.xH_box, z_array=lightcone.lightcone_redshifts, box_size_mpc=params['BOX_LEN']) 
    mask_xn = smt_xn>0.5

    print(' save outputs...') 
    t2c.save_cbin(path_out+'data/xH_21cm_i%d.bin' %i, mask_xn)
    t2c.save_cbin(path_out+'data/dT3_21cm_i%d.bin' %i, dT3)
    #t2c.save_cbin(path_out+'data/dT1_21cm_i%d.bin' %i, lightcone.brightness_temp)
    np.savetxt('%slc_redshifts.txt' %(path_out), lightcone.lightcone_redshifts, fmt='%.5f')

    # save parameters values
    with open('%sastro_params_rank%d.txt' %(path_out+'parameters/', rank), 'a') as f:
        if(i == 0 and rank == 0):
            f.write('# HII_EFF_FACTOR: The ionizing efficiency of high-z galaxies\n')
            f.write('# R_BUBBLE_MAX: Mean free path in Mpc of ionizing photons within ionizing regions\n')
            f.write('# ION_Tvir_MIN: Minimum virial Temperature of star-forming haloes in log10 units\n')
            f.write('#i\teff_f\tRmfp\tTvir\tseed\n')
            f.write('%d\t%.3f\t%.3f\t%.3f\t%d\n' %(i, eff_fact, Rmfp, Tvir, rseed))
        else:
            f.write('%d\t%.3f\t%.3f\t%.3f\t%d\n' %(i, eff_fact, Rmfp, Tvir, rseed))

    
    # if output dir is more than 15 GB of size, compress and remove files in data/
    if(get_dir_size(path_out) / 1e9 >= 15):
        comm.Barrier()  # wait that all proc are done
        
        # start with compression on rank=0
        if(rank == 0):
            try:
                strd = np.loadtxt('%swritten.txt' %(path_out), dtype=str, delimiter='\n')
            except:
                strd = np.array([])

            # compress data
            os.system('tar -czvf %s_part%d.tar.gz %s' %(path_out+'../'+path_out[path_out[:-1].rfind('/')+1:-1], strd.size+1, path_out))
            
            np.savetxt('%swritten.txt' %(path_out), np.append(strd, ['%s written %s_part%d.tar.gz' %(datetime.now().strftime('%d/%m/%Y %H:%M:%S'), path_out[path_out[:-1].rfind('/')+1:-1], strd.size+1)]), delimiter='\n', fmt='%s')
            print(' \n Data created exeed 15GB. Compression completed...')

            os.system('rm %sdata/*.bin' %path_out)  # delete uncompressed data

        comm.Barrier()  # wait for deletion to be completed
    i += 1 # update while loop index

# wait that all processors are done before concluding the job
comm.Barrier()
#nr_procs_done = 1
#nr_procs_done = comm.gather(nr_procs_done, root=0)
if(rank == 0):
    print(' Gather done:\t%s\n' %datetime.now().strftime('%H:%M:%S'))
    # merge the different astro_params_rank*.txt files into one
    for i_p in range(nprocs):
        data = np.loadtxt('%sastro_params_rank%d.txt' %(path_out+'parameters/', i_p))
        if(i_p == 0): 
            stack_data = data 
        else: 
            stack_data = np.vstack((stack_data, data)) 
    np.savetxt('%sastro_params.txt' %(path_out+'parameters/'), stack_data, header='HII_EFF_FACTOR: The ionizing efficiency of high-z galaxies\nR_BUBBLE_MAX: Mean free path in Mpc of ionizing photons within ionizing regions\nION_Tvir_MIN: Minimum virial Temperature of star-forming haloes in log10 units\ni\teff_f\tRmfp\tTvir\tseed', fmt='%d\t%.3f\t%.3f\t%.3f\t%d')

    # compres remaining data
    try:
        strd = np.loadtxt('%swritten.txt' %(path_out), dtype=str, delimiter='\n')
    except:
        strd = np.array([])

    os.system('tar -czvf %s_part%d.tar.gz %s' %(path_out+'../'+path_out[path_out[:-1].rfind('/')+1:-1], strd.size+1, path_out))
    np.savetxt('%swritten.txt' %(path_out), np.append(strd, ['%s written %s_part%d.tar.gz' %(datetime.now().strftime('%d/%m/%Y %H:%M:%S'), path_out[path_out[:-1].rfind('/')+1:-1], strd.size+1)]), delimiter='\n', fmt='%s')
    #os.system('rm %sdata/*.bin' %path_out)  # delete uncompressed data
    os.system('mv %s../*tar.gz %sdata/' %(path_out, path_out))

# all ranks wait that rank=0 is done
comm.Barrier() 

# remove ranks cache directories
os.system('rm -r %s' %path_cache)
print('... rank %d done at %s.' %(rank, datetime.now().strftime('%H:%M:%S')))