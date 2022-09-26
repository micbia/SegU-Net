import numpy as np, os, sys, tarfile
import tools21cm as t2c, py21cmfast as p2c 

from datetime import datetime 
from glob import glob
from sklearn.decomposition import PCA as sciPCA

import sys
sys.path.insert(0,'../')
from utils.other_utils import get_dir_size

path_input = sys.argv[1]
path_input += '/' if path_input[-1] != '/' else ''
path_out = path_input if sys.argv[2] == 'same' else sys.argv[2]
path_out += '/' if path_out[-1] != '/' else ''  
try:
    os.makedirs(path_out)
    os.makedirs(path_out+'data')
    os.makedirs(path_out+'images')
    os.makedirs(path_out+'parameters')
except:
    pass

name_of_the_run = path_out[path_out[:-1].rfind('/')+1:-1]

# MPI setup
rank = int(os.environ['SLURM_ARRAY_TASK_ID'])
nprocs = int(os.environ['SLURM_ARRAY_TASK_COUNT'])

print(' Starting rank %d at: %s' %(rank, datetime.now().strftime('%H:%M:%S')))

# 21cmFAST parameters
COMPRESS = False
#RERUN = ['dT3', 'dT4', 'dT4pca', 'dT5', 'dT5pca']
RERUN = ['dT4pca']
nr = 4      # componens to remove in PCA
uvfile = '/store/ska/sk09/segunet/uvmap_128_z7-20.pkl'
z_min, z_max = 7, 11
tobs = 1000.
MAKE_PLOT = False

# Loop parameters
loop_start, loop_end = 0, 10000
perrank = (loop_end-loop_start)//nprocs
"""
path_cache = '/scratch/snx3000/mibianco/_cache%d/' %rank
if not (os.path.exists(path_cache)):
    os.makedirs(path_cache)
else:
    os.system('rm %s*h5' %path_cache)
p2c.config['direc'] = path_cache
"""

# read parameters files
try:
    params = eval(open(path_input+'parameters/user_params.txt', 'r').read())
    c_params = eval(open(path_input+'parameters/cosm_params.txt', 'r').read())
    astro_params = np.loadtxt(path_input+'parameters/astro_params.txt')
    redshifts = np.loadtxt(path_input+'lc_redshifts.txt')
except FileNotFoundError as error:
    print(error)

# Set loop ending index per processor
i_start = int(loop_start+rank*perrank)
if(rank != nprocs-1):
    i_end = int(loop_start+(rank+1)*perrank)
else:
    i_end = loop_end

# Start loop
print(' Processors repartition:\n rank %d\t%d\t%d' %(rank, i_start, i_end)) 
for i in range(i_start, i_end):
    #if not (os.path.exists(path_out+'data/dT3_21cm_i%d.bin' %i)):
    if ('dT' in RERUN and not (os.path.exists(path_input+'data/dT_21cm_i%d.bin' %i) or os.path.exists(path_out+'data/dT_21cm_i%d.bin' %i)) or 'xHI' in RERUN and not (os.path.exists(path_input+'data/xHI_21cm_i%d.bin' %i) or os.path.exists(path_out+'data/xHI_21cm_i%d.bin' %i))):
        # Define astronomical parameters
        eff_fact, Rmfp, Tvir, rseed = astro_params[i, 1:]
        a_params = {'HII_EFF_FACTOR':eff_fact, 'R_BUBBLE_MAX':Rmfp, 'ION_Tvir_MIN':Tvir}
        print(' Re-run random seed:\t %d' %rseed)
        
        path_cache = './'
        try:
            os.system('rm %s*h5' %path_cache)
        except:
            pass

        print(' re-calculate lightcone...') 
        lightcone = p2c.run_lightcone(redshift=z_min, max_redshift=z_max, 
                                    user_params=params, astro_params=a_params, cosmo_params=c_params, 
                                    lightcone_quantities=("brightness_temp", 'xH_box'), 
                                    #flag_options={"USE_TS_FLUCT": True},
                                    global_quantities=("brightness_temp", 'xH_box'), 
                                    direc=path_cache, random_seed=rseed) 
        
        dT = lightcone.brightness_temp
        t2c.save_cbin(path_out+'data/dT_21cm_i%d.bin' %i, dT)
        t2c.save_cbin(path_out+'data/xHI_21cm_i%d.bin' %i, lightcone.xH_box)
    
    if('dT2' in RERUN and not (os.path.exists(path_input+'data/dT2_21cm_i%d.bin' %i) or os.path.exists(path_out+'data/dT2_21cm_i%d.bin' %i))):
        dT = t2c.read_cbin(path_input+'data/dT_21cm_i%d.bin' %i)
        dT2, _ = t2c.smooth_lightcone(t2c.subtract_mean_signal(dT, los_axis=2), z_array=redshifts, box_size_mpc=params['BOX_LEN']) 
        t2c.save_cbin(path_out+'data/dT2_21cm_i%d.bin' %i, dT2) # smooth(dT - avrg_dT)
    if('dT3' in RERUN or 'dT4' in RERUN or 'dT5' in RERUN):
        dT = t2c.read_cbin(path_input+'data/dT_21cm_i%d.bin' %i)
        lc_noise = t2c.noise_lightcone(ncells=params['HII_DIM'], 
                                        zs=redshifts, 
                                        obs_time=tobs, save_uvmap=uvfile,
                                        boxsize=params['BOX_LEN'], n_jobs=1)
    if('dT3' in RERUN and not (os.path.exists(path_input+'data/dT3_21cm_i%d.bin' %i) or os.path.exists(path_out+'data/dT3_21cm_i%d.bin' %i))):
        dT3, _ = t2c.smooth_lightcone(t2c.subtract_mean_signal(dT + lc_noise, los_axis=2), z_array=redshifts, box_size_mpc=params['BOX_LEN']) 
        t2c.save_cbin(path_out+'data/dT3_21cm_i%d.bin' %i, dT3) # smooth(dT + noise - avrg_dT)
    if('dT4' in RERUN and not (os.path.exists(path_input+'data/dT4_21cm_i%d.bin' %i) or os.path.exists(path_out+'data/dT4_21cm_i%d.bin' %i))):
        gal_fg = t2c.galactic_synch_fg(z=redshifts, ncells=params['HII_DIM'], boxsize=params['BOX_LEN'], rseed=rseed)
        dT4, _ = t2c.smooth_lightcone(t2c.subtract_mean_signal(dT + lc_noise + gal_fg, los_axis=2), z_array=redshifts, box_size_mpc=params['BOX_LEN'])
        t2c.save_cbin(path_out+'data/dT4_21cm_i%d.bin' %i, dT4) # smooth(dT + noise + gf - avrg_dT)
    if('dT4pca' in RERUN and not (os.path.exists(path_input+'data/dT4pca%s_21cm_i%d.bin' %(str(nr), i)) or os.path.exists(path_out+'data/dT4pca%s_21cm_i%d.bin' %(str(nr), i)))):
        dT4 = t2c.read_cbin(path_input+'data/dT4_21cm_i%d.bin' %i)
        data_flat = np.reshape(dT4, (-1, dT4.shape[2]))
        pca = sciPCA(n_components=nr)
        datapca = pca.fit_transform(data_flat)
        pca_FG = pca.inverse_transform(datapca)
        dT4pca = np.reshape(data_flat - pca_FG, dT4.shape)
        t2c.save_cbin(path_out+'data/dT4pca%s_21cm_i%d.bin' %(str(nr), i), dT4pca)
    if('dT5' in RERUN and not (os.path.exists(path_input+'data/dT5_21cm_i%d.bin' %i) or os.path.exists(path_out+'data/dT5_21cm_i%d.bin' %i))):
        gal_fg = t2c.galactic_synch_fg(z=redshifts, ncells=params['HII_DIM'], boxsize=params['BOX_LEN'], rseed=rseed)
        exgal_fg = t2c.extragalactic_pointsource_fg(z=redshifts, ncells=params['HII_DIM'], boxsize=params['BOX_LEN'], rseed=rseed)
        dT5, _ = t2c.smooth_lightcone(t2c.subtract_mean_signal(dT + lc_noise + exgal_fg + gal_fg, los_axis=2), z_array=redshifts, box_size_mpc=params['BOX_LEN'])
        t2c.save_cbin(path_out+'data/dT5_21cm_i%d.bin' %i, dT5)  # smooth(dT + noise + gf + exgf - avrg_dT)
        np.save(path_out+'data/dTexgf_21cm_i%d.npy' %i, exgal_fg[..., 0]) # save just the extragalactic points first slice
    if('dT5pca' in RERUN and not (os.path.exists(path_input+'data/dT5pca_21cm_i%d.bin' %i) or os.path.exists(path_out+'data/dT5pca_21cm_i%d.bin' %i))):
        dT5 = t2c.read_cbin(path_input+'data/dT5_21cm_i%d.bin' %i)
        data_flat = np.reshape(dT5, (-1, dT5.shape[2]))
        pca = sciPCA(n_components=7)
        datapca = pca.fit_transform(data_flat)
        pca_FG = pca.inverse_transform(datapca)
        dT5pca = np.reshape(data_flat - pca_FG, dT5.shape)
        t2c.save_cbin(path_out+'data/dT5pca_21cm_i%d.bin' %i, dT5pca)
    if('xH' in RERUN and not (os.path.exists(path_input+'data/xH_21cm_i%d.bin' %i) or os.path.exists(path_out+'data/xH_21cm_i%d.bin' %i))):
        xHI = t2c.read_cbin(path_input+'data/xHI_21cm_i%d.bin' %i)
        smt_xn, _ = t2c.smooth_lightcone(xHI, z_array=redshifts, box_size_mpc=params['BOX_LEN']) 
        mask_xH = smt_xn>0.5
        t2c.save_cbin(path_out+'data/xH_21cm_i%d.bin' %i, mask_xH)

    # if output dir is more than 15 GB of size, compress and remove files in data/
    if(get_dir_size(path_out) >= 8 and COMPRESS):
        # start with compression on rank=0
        if(rank == nprocs-1):
            if(os.path.isfile(path_out+'written.txt')):
                strd = np.loadtxt('%swritten.txt' %(path_out), dtype=str, delimiter='\n')
                content = np.loadtxt('%scontent.txt' %(path_out+'data/'))
                i_part = strd.size+1
            else:
                strd = np.array([])
                content = np.zeros(loop_end)
                i_part = 1

            # file with content
            idx_content = [int(cont[cont.rfind('_i')+2:cont.rfind('.')]) for i_c, cont in enumerate(glob(path_out+'data/xH_21cm_i*'))]
            content[idx_content] = i_part
            np.savetxt('%sdata/content.txt' %path_out, content, fmt='%d')

            # compress data
            os.system('tar -czvf %s_part%d.tar.gz %s/' %(name_of_the_run, i_part, name_of_the_run))
            
            # get list of content and prepare it for the data_generator
            mytar = tarfile.open('%s_part%d.tar.gz' %(name_of_the_run, i_part), 'r')
            tar_content = mytar.getmembers()
            tar_names = mytar.getnames()
            np.save('%sdata/tar_content_part%d' %(path_out, i_part), tar_content)
            np.save('%sdata/tar_names_part%d' %(path_out, i_part), tar_names)
            mytar.close()

            # note down the compressed file
            np.savetxt('%swritten.txt' %(path_out), np.append(strd, ['%s written %s_part%d.tar.gz' %(datetime.now().strftime('%d/%m/%Y %H:%M:%S'), path_out[path_out[:-1].rfind('/')+1:-1], i_part)]), delimiter='\n', fmt='%s')
            
            # free the space in the data/ directory
            os.system('rm %sdata/*.bin' %path_out)
            print(' \n Data created exeed 15GB. Compression completed...')


# wait that all processors are done before concluding the job
if(rank == nprocs-1  and COMPRESS):
    print(' Gather done:\t%s\n' %datetime.now().strftime('%H:%M:%S'))
    # merge the different astro_params_rank*.txt files into one
    for i_p in range(nprocs):
        data = np.loadtxt('%sastro_params_rank%d.txt' %(path_out+'parameters/', i_p))
        if(i_p == 0): 
            stack_data = data 
        else: 
            stack_data = np.vstack((stack_data, data)) 
    np.savetxt('%sastro_params.txt' %(path_out+'parameters/'), stack_data, header='HII_EFF_FACTOR: The ionizing efficiency of high-z galaxies\nR_BUBBLE_MAX: Mean free path in Mpc of ionizing photons within ionizing regions\nION_Tvir_MIN: Minimum virial Temperature of star-forming haloes in log10 units\ni\teff_f\tRmfp\tTvir\tseed', fmt='%d\t%.3f\t%.3f\t%.3f\t%d')

    if(os.path.isfile(path_out+'written.txt')):
        strd = np.loadtxt('%swritten.txt' %(path_out), dtype=str, delimiter='\n')
        content = np.loadtxt('%sdata/content.txt' %(path_out))
        i_part = strd.size+1
    else:
        strd = np.array([])
        content = np.zeros(loop_end)
        i_part = 1

    # file with content
    idx_content = [int(cont[cont.rfind('_i')+2:cont.rfind('.')]) for i_c, cont in enumerate(glob(path_out+'data/xH_21cm_i*'))]
    content[idx_content] = i_part
    np.savetxt('%sdata/content.txt' %path_out, content, fmt='%d')
    
    # compress data
    os.system('tar -czvf %s_part%d.tar.gz %s/' %(name_of_the_run, i_part, name_of_the_run))

    # get list of content and prepare it for the data_generator
    mytar = tarfile.open('%s_part%d.tar.gz' %(name_of_the_run, i_part), 'r')
    tar_content = mytar.getmembers()
    tar_names = mytar.getnames()
    np.save('%sdata/tar_content_part%d' %(path_out, i_part), tar_content)
    np.save('%sdata/tar_names_part%d' %(path_out, i_part), tar_names)
    mytar.close()
    
    # note down the compressed file
    np.savetxt('%swritten.txt' %(path_out), np.append(strd, ['%s written %s_part%d.tar.gz' %(datetime.now().strftime('%d/%m/%Y %H:%M:%S'), path_out[path_out[:-1].rfind('/')+1:-1], i_part)]), delimiter='\n', fmt='%s')
    
    # free the space in the data/ directory
    os.system('rm %sdata/*.bin' %path_out)
    os.system('mv %s../*tar.gz %sdata/' %(path_out, path_out))

# remove ranks cache directories
#os.system('rm -r %s' %path_cache)
print('... rank %d done at %s.' %(rank, datetime.now().strftime('%H:%M:%S')))
