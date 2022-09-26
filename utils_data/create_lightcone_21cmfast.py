import numpy as np, random, os, sys, json, tarfile
import tools21cm as t2c, py21cmfast as p2c 

from datetime import datetime 
from skopt.sampler import Lhs
from skopt.space import Space
from glob import glob
from sklearn.decomposition import PCA as sciPCA

import sys
sys.path.insert(0,'../')
from utils.other_utils import get_dir_size, GenerateSeed

path_out = sys.argv[1]
path_out += '/' if path_out[-1]!='/' else ''
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

# MPI setup
rank = int(os.environ['SLURM_ARRAY_TASK_ID'])
nprocs = int(os.environ['SLURM_ARRAY_TASK_COUNT'])

print(' Starting rank %d at: %s' %(rank, datetime.now().strftime('%H:%M:%S')))

# 21cmFAST parameters
c_params = {'OMm':0.27, 'OMb':0.046, 'SIGMA_8':0.82, 'POWER_INDEX':0.96}
uvfile = '/store/ska/sk09/segunet/uvmap_128_z7-20.pkl'
params = {'HII_DIM':128, 'DIM':512, 'BOX_LEN':256}
z_min, z_max = 7.000, 11.000
#uvfile = '/store/ska/sk09/segunet/uvmap_200_z5-35.pkl'
#params = {'HII_DIM':200, 'DIM':600, 'BOX_LEN':300, 'USE_INTERPOLATION_TABLES': False}
#z_min, z_max = 5.000, 29.999
tobs = 1000.
COMPRESS = False
MAKE_PLOT = False

# Loop parameters
loop_start, loop_end = 0, 10000
perrank = (loop_end-loop_start)//nprocs

path_cache = '/scratch/snx3000/mibianco/_cache%d/' %rank
if not (os.path.exists(path_cache)):
    os.makedirs(path_cache)
else:
    os.system('rm %s*h5' %path_cache)
p2c.config['direc'] = path_cache


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


# Start loop
print(' Processors repartition:\n rank %d\t%d\t%d' %(rank, i, i_end)) 
while i < i_end:
    rseed = GenerateSeed()
    print(' generated seed:\t %d' %rseed)

    random.seed(rseed)
    eff_fact, Rmfp, Tvir = random.gauss(82., 18.), random.gauss(17.5, 4.5), random.gauss(4.7, 0.2)

    # Latin Hypercube Sampling of parameters
    #space = Space([(10., 100.), (5., 20.), (np.log10(1e4), np.log10(2e5)), (38.0, 42.), (100., 1500.)]) 
    #space = Space([(10., 250.), (5., 20.), (4., 6.), (38.0, 42.), (100., 1500.)]) 
    #lhs_sampling = np.array(Lhs(criterion="maximin", iterations=10000).generate(dimensions=space.dimensions, n_samples=1, random_state=rseed))
    #eff_fact, Rmfp, Tvir, LX, E0 = lhs_sampling[0]

    # Define astronomical parameters
    #a_params = {'HII_EFF_FACTOR':eff_fact, 'R_BUBBLE_MAX':Rmfp, 'ION_Tvir_MIN':Tvir, 'L_X': LX, 'NU_X_THRESH': E0}
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
    
    np.savetxt('%slc_redshifts.txt' %(path_out), lightcone.lightcone_redshifts, fmt='%.5f')
    
    xHI = lightcone.xH_box
    if((np.mean(xHI[...,0]) <= 0.1) * (np.mean(xHI[...,-1]) >= 0.9)):
        lc_noise = t2c.noise_lightcone(ncells=lightcone.brightness_temp.shape[0], 
                                    zs=lightcone.lightcone_redshifts, 
                                    obs_time=tobs, save_uvmap=uvfile, 
                                    boxsize=params['BOX_LEN'], n_jobs=1)

        #print(' calculate foregrounds...')
        gal_fg = t2c.galactic_synch_fg(z=lightcone.lightcone_redshifts, ncells=params['HII_DIM'], boxsize=params['BOX_LEN'], rseed=rseed)
        exgal_fg = t2c.extragalactic_pointsource_fg(z=lightcone.lightcone_redshifts, ncells=params['HII_DIM'], boxsize=params['BOX_LEN'], rseed=rseed)

        print(' calculate dT and mask...') 
        dT = lightcone.brightness_temp
        dT1 = t2c.subtract_mean_signal(lightcone.brightness_temp, los_axis=2)  
        dT2, redshifts = t2c.smooth_lightcone(dT1, z_array=lightcone.lightcone_redshifts, box_size_mpc=params['BOX_LEN']) 
        dT3, _ = t2c.smooth_lightcone(t2c.subtract_mean_signal(dT + lc_noise, los_axis=2), z_array=lightcone.lightcone_redshifts, box_size_mpc=params['BOX_LEN']) 
        dT4, _ = t2c.smooth_lightcone(t2c.subtract_mean_signal(dT + lc_noise + gal_fg, los_axis=2), z_array=lightcone.lightcone_redshifts, box_size_mpc=params['BOX_LEN'])
        dT5, _ = t2c.smooth_lightcone(t2c.subtract_mean_signal(dT + lc_noise + exgal_fg + gal_fg, los_axis=2), z_array=lightcone.lightcone_redshifts, box_size_mpc=params['BOX_LEN'])
        
        print(' calculate PCA...')
        data = dT4
        data_flat = np.reshape(data, (-1, data.shape[2]))
        pca = sciPCA(n_components=7)
        datapca = pca.fit_transform(data_flat)
        pca_FG = pca.inverse_transform(datapca)
        dT4pca = np.reshape(data_flat - pca_FG, data.shape)

        smt_xn, _ = t2c.smooth_lightcone(xHI, z_array=lightcone.lightcone_redshifts, box_size_mpc=params['BOX_LEN']) 
        mask_xH = smt_xn>0.5

        print(' save outputs...') 
        t2c.save_cbin(path_out+'data/xH_21cm_i%d.bin' %i, mask_xH)
        t2c.save_cbin(path_out+'data/xHI_21cm_i%d.bin' %i, xHI)
        t2c.save_cbin(path_out+'data/dT_21cm_i%d.bin' %i, dT)
        t2c.save_cbin(path_out+'data/dT2_21cm_i%d.bin' %i, dT2) # smooth(dT - avrg_dT)
        t2c.save_cbin(path_out+'data/dT3_21cm_i%d.bin' %i, dT3) # smooth(dT + noise - avrg_dT)
        t2c.save_cbin(path_out+'data/dT4_21cm_i%d.bin' %i, dT4) # smooth(dT + noise + gf - avrg_dT)
        t2c.save_cbin(path_out+'data/dT5_21cm_i%d.bin' %i, dT5)  # smooth(dT + noise + gf + exgf - avrg_dT)
        np.save(path_out+'data/dTexgf_21cm_i%d.npy' %i, exgal_fg[..., 0]) # just the point sourcess
        #t2c.save_cbin(path_out+'data/dT5pca_21cm_i%d.bin' %i, dT5pca)
        t2c.save_cbin(path_out+'data/dT4pca_21cm_i%d.bin' %i, dT4pca)

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
        
        i += 1 # update while loop index
    else:
        continue

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
os.system('rm -r %s' %path_cache)
print('... rank %d done at %s.' %(rank, datetime.now().strftime('%H:%M:%S')))
