import numpy as np, os, sys, tarfile
import tools21cm as t2c

from datetime import datetime 
from glob import glob

#import tensorflow as tf
#from wedgeTF import wedge_removal_tf, multiplicative_factor, calculate_k_cube, calculate_blackman
from sklearn.decomposition import PCA as sciPCA

sys.path.insert(0,'../')
from utils.other_utils import get_dir_size

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

# Change working directory
os.chdir(path_out+'..')
cwd = os.getcwd()

# MPI setup
rank = int(os.environ['SLURM_ARRAY_TASK_ID'])
nprocs = int(os.environ['SLURM_ARRAY_TASK_COUNT'])

print(' Starting rank %d at: %s' %(rank, datetime.now().strftime('%H:%M:%S')))

# 21cmFAST parameters
params = {'HII_DIM':128, 'DIM':512, 'BOX_LEN':256} 
cosmo_params = {'OMm':0.27, 'OMb':0.046, 'SIGMA_8':0.82, 'POWER_INDEX':0.96} 
MAKE_PLOT = False
COMPRESS = False

# Loop parameters
loop_start, loop_end = 0, 1500
perrank = (loop_end-loop_start)//nprocs

# Set loop ending index per processor
i_start = int(loop_start+rank*perrank)
if(rank != nprocs-1):
    i_end = int(loop_start+(rank+1)*perrank)
else:
    i_end = loop_end

"""
# for the wedge remove
astro_params = np.loadtxt(path_out+'parameters/astro_params.txt')
redshifts = np.loadtxt(path_out+'lc_redshifts.txt')
chunk_length = 129
cell_size = params['BOX_LEN'] / params['HII_DIM']
MF = tf.constant([multiplicative_factor(z, cosmo_params['OMm']) for z in redshifts], dtype=tf.float32)
k_cube, delta_k = calculate_k_cube(params['HII_DIM'], chunk_length, cell_size)
BM, buffer = calculate_blackman(chunk_length, delta_k)
"""

# Start loop
print(' Processors repartition:\n rank %d\t%d\t%d' %(rank, i_start, i_end)) 
for i in range(i_start, i_end):
    #dT3 = t2c.read_cbin('%sdata/dT3_21cm_i%d.bin' %(path_out, i))
    #dT4 = t2c.read_cbin('%sdata/dT4_21cm_i%d.bin' %(path_out, i))
    dT5 = t2c.read_cbin('%sdata/dT5_21cm_i%d.bin' %(path_out, i))
 
    print(' calculate PCA...')
    data = dT5
    data_flat = np.reshape(data, (-1, data.shape[2]))
    pca = sciPCA(n_components=7)
    datapca = pca.fit_transform(data_flat)
    pca_FG = pca.inverse_transform(datapca)
    dT5pca = np.reshape(data_flat - pca_FG, (params['HII_DIM'], params['HII_DIM'], data.shape[2]))

    #print(' calculate wedge...') 
    #dT3wdg = 
    #dT3wdg = wedge_removal_tf(OMm=cosmo_params['OMm'], redshifts=redshifts, HII_DIM=params['HII_DIM'], cell_size=cell_size, Box=tf.constant(dT3, dtype=tf.float32), chunk_length=chunk_length, blackman=True, MF=MF, k_cube_data=(k_cube, delta_k), blackman_data=(BM, buffer))

    print(' save outputs...') 
    #t2c.save_cbin(path_out+'data/dT3wdg_21cm_i%d.bin' %i, dT3wdg)
    t2c.save_cbin(path_out+'data/dT5pca_21cm_i%d.bin' %i, dT5pca)

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
if(rank == nprocs-1 and COMPRESS):
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
print('... rank %d done at %s.' %(rank, datetime.now().strftime('%H:%M:%S')))
