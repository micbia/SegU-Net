import numpy as np, os, sys, json, pickle
import tools21cm as t2c, py21cmfast as p2c 
from datetime import datetime, date 
from tqdm import tqdm

import matplotlib.ticker as plticker
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
def GenerateSeed(): 
    # create seed for 21cmFast 
    seed = [var for var in datetime.now().strftime('%d%H%M%S')] 
    np.random.shuffle(seed) 
    return int(''.join(seed)) 
 
path_out, rank, perrank = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
path_out += '/' if path_out[-1]!='/' else '' 
 
#path_chache = '/cosma6/data/dp004/dc-bian1/21cmFAST-cache/'
path_chache = '/cosma6/data/dp004/dc-bian1/_cache%d/' %rank
try: 
    os.makedirs(path_chache)
except:
    pass
p2c.config['direc'] = path_chache 
 
uvfile = '/cosma6/data/dp004/dc-bian1/uvmap_128_z7-20.pkl'
"""
uvfile_s = '/cosma6/data/dp004/dc-bian1/uvmap_128_z7-20.pkl'
if(os.path.exists(uvfile_s)): 
    print(' uv-file exist')
    infile = open(uvfile_s, 'rb') 
    uvfile = pickle.load(infile)
    infile.close()
else: 
    print(' NO uv-FILE') 
    #sys.exit() 
"""
path_chache = '/cosma6/data/dp004/dc-bian1/_cache%d/' %rank
p2c.config['direc'] = path_chache 
 
# LC astrophysical and cosmological parameters 
u_params = {'HII_DIM':128, 'DIM':384, 'BOX_LEN':256} 
c_params = {'OMm':0.27, 'OMb':0.046, 'SIGMA_8':0.82, 'POWER_INDEX':0.96} 
z_min, z_max = 7, 11
eff_fact, Rmfp, Tvir = 42.580, 12.861, 4.539 
a_params = {'HII_EFF_FACTOR':eff_fact, 'R_BUBBLE_MAX':Rmfp, 'ION_Tvir_MIN':Tvir} 
tobs = 1000. 
 
loop_start, loop_end = 0, 500
i_start = int(loop_start+rank*perrank) 
i_end = int(loop_start+(rank+1)*perrank)

MAKE_PLOT = False
for i in range(i_start, i_end): 
    rseed = GenerateSeed() 
    print(' generated seed:\t %d' %rseed) 
    print(' calculate lightcone...') 
    lightcone = p2c.run_lightcone(redshift=z_min, max_redshift=z_max, user_params=u_params, astro_params=a_params, cosmo_params=c_params, lightcone_quantities=("brightness_temp", 'xH_box'), global_quantities=("brightness_temp", 'xH_box'), direc=path_chache, random_seed=rseed) 
    
    print(' calculate noise...') 
    lc_noise = t2c.noise_lightcone(ncells=lightcone.brightness_temp.shape[0], zs=lightcone.lightcone_redshifts, obs_time=tobs, save_uvmap=uvfile, boxsize=u_params['BOX_LEN'], n_jobs=1)

    print(' calculate dT and mask...') 
    dT2 = t2c.subtract_mean_signal(lightcone.brightness_temp, los_axis=2)  
    dT2_smt, redshifts = t2c.smooth_lightcone(dT2, z_array=lightcone.lightcone_redshifts, box_size_mpc=u_params['BOX_LEN']) 
    dT3, redshifts = t2c.smooth_lightcone(dT2 + lc_noise, z_array=lightcone.lightcone_redshifts, box_size_mpc=u_params['BOX_LEN']) 
    
    smt_xn, redshifts = t2c.smooth_lightcone(lightcone.xH_box, z_array=lightcone.lightcone_redshifts, box_size_mpc=u_params['BOX_LEN']) 
    mask_xn = smt_xn>0.5 

    # delete chache
    os.system('rm %s*.h5' %path_chache)
    
    # mask are saved such that 1 in neutral region and 0 in ionized region 
    print(' save outputs...') 
    t2c.save_cbin(filename=path_out+'data/lc_256Mpc_dT_i%d.dat' %i, data=dT2_smt) 
    t2c.save_cbin(filename=path_out+'data/lc_256Mpc_dT3_i%d.dat' %i, data=dT3)
    t2c.save_cbin(filename=path_out+'data/lc_256Mpc_mask_i%d.dat' %i, data=mask_xn)
    
    with open('%sastro_params_rank%d.txt' %(path_out+'parameters/', rank), 'a') as f:
        if(i == 0 and rank == 0):
            f.write('# HII_EFF_FACTOR: The ionizing efficiency of high-z galaxies\n')
            f.write('# R_BUBBLE_MAX: Mean free path in Mpc of ionizing photons within ionizing regions\n')
            f.write('# ION_Tvir_MIN: Minimum virial Temperature of star-forming haloes in log10 units\n')
            f.write('#i\teff_f\tRmfp\tTvir\tseed\n')
            f.write('%d\t%.3f\t%.3f\t%.3f\t%d\n' %(i, eff_fact, Rmfp, Tvir, rseed))
        else:
            f.write('%d\t%.3f\t%.3f\t%.3f\t%d\n' %(i, eff_fact, Rmfp, Tvir, rseed))


    if(MAKE_PLOT):
        plt.rcParams['font.size'] = 20
        plt.rcParams['xtick.direction'] = 'out'
        plt.rcParams['ytick.direction'] = 'out'
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True

        my_ext = [z_min, z_max, 0, u_params['BOX_LEN']]
        idx_plot = u_params['HII_DIM']//2

        fig, axes = plt.subplots(figsize=(25, 15))
        gs = gridspec.GridSpec(3, 1)
        ax0 = plt.subplot(gs[0])
        im = ax0.imshow(mask_xn[:,idx_plot,:], origin='lower', cmap='jet', aspect='auto', extent=my_ext)
        ax1 = plt.subplot(gs[1])
        im = ax1.imshow(dT2_smt[:,idx_plot,:], origin='lower', cmap='jet', aspect='auto', extent=my_ext)
        ax2 = plt.subplot(gs[2])
        im = ax2.imshow(dT3[:,idx_plot,:], origin='lower', cmap='jet', aspect='auto', extent=my_ext)
        fig.colorbar(im, label=r'$\delta T_{\rm b}$ [mK]', ax=ax1, pad=0.02, cax=fig.add_axes([0.905, 0.13, 0.01, 0.22]))

        ax0.set_ylabel('$Mpc$', size=16)
        ax1.set_ylabel('$Mpc$', size=16)
        ax2.set_ylabel('$Mpc$', size=16)
        ax2.set_xlabel('z', size=16)
        ax0.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
        ax0.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
        ax0.yaxis.set_minor_locator(plticker.MultipleLocator(base=10))
        ax0.yaxis.set_major_locator(plticker.MultipleLocator(base=50))
        ax1.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
        ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
        ax1.yaxis.set_minor_locator(plticker.MultipleLocator(base=10))
        ax1.yaxis.set_major_locator(plticker.MultipleLocator(base=50))
        ax2.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
        ax2.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
        ax2.yaxis.set_minor_locator(plticker.MultipleLocator(base=10))
        ax2.yaxis.set_major_locator(plticker.MultipleLocator(base=50))

        plt.subplots_adjust(hspace=0.01)
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax1.get_xticklabels(), visible=False)

        plt.savefig('%simages/plotLC_dTb_i%d.png' %(path_out, i), bbox_inches='tight')
        plt.close()
print('...done.')

'''
path_out = sys.argv[1]
#rank = int(sys.argv[2])
#rank = int(os.environ['SLURM_ARRAY_TASK_ID'])
rank = 0

path_out += '/' if path_out[-1]!='/' else ''

path_chache = '/cosma6/data/dp004/dc-bian1/21cmFAST-cache/'
p2c.config['direc'] = path_chache

loop_start, loop_end = 0, 7000
#nprocs = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
nprocs = 1
#loop_start, loop_end = np.loadtxt('parameters/todo_r%d.txt' %rank, dtype=int)
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


params = {'HII_DIM':128, 'DIM':384, 'BOX_LEN':256}
c_params = {'OMm':0.27, 'OMb':0.046, 'SIGMA_8':0.82, 'POWER_INDEX':0.96}

uvfile = '/cosma6/data/dp004/dc-bian1/uvmap_128_z7-20.pkl'
if(os.path.exists(uvfile)):
    print(' uv-file exist')
else:
    print(' NO uv-FILE')
    sys.exit()

tobs = 1000
MAKE_PLOT = True

if not (os.path.exists('%sastro_params_rank%d.txt' %(path_out+'parameters/', rank))):
    loop_resume = 0
    
    with open(path_out+'parameters/user_params.txt', 'w') as file:
        file.write(json.dumps(params))

    with open(path_out+'parameters/cosm_params.txt', 'w') as file:
        file.write(json.dumps(c_params))
else:
    loop_resume = int(np.loadtxt('%sastro_params_rank%d.txt' %(path_out+'parameters/', rank))[:,0].max())


if(loop_resume == 0):
    i = int(loop_start+rank*perrank)
elif(loop_resume != 0):
    print(' Rank=%d resumes itration from i=%d.' %(rank, loop_resume))
    i = int(loop_resume + 1)

#if(rank != int(os.environ['SLURM_ARRAY_TASK_MAX'])):
if(rank != 1):
    i_end = int(loop_start+(rank+1)*perrank)
else:
    i_end = int(loop_start+(rank+1)*perrank)
    if(i_end != loop_end):
        i_end = loop_end

while i < i_end:
    # astronomical & cosmological parameters
    #eff_fact = random.gauss(52.5, 20.)  # eff_fact = [5, 100]
    #Rmfp = random.gauss(12.5, 5.)       # Rmfp = [5, 20]
    #Tvir = random.gauss(4.65, 0.5)      # Tvir = [log10(1e4), log10(2e5)]
    rseed = GenerateSeed()
    z_min = 7
    z_max = 11
    eff_fact = 42.580
    Rmfp = 12.861
    Tvir = 4.539

    # Define astronomical parameters
    a_params = {'HII_EFF_FACTOR':eff_fact, 'R_BUBBLE_MAX':Rmfp, 'ION_Tvir_MIN':Tvir}

    
    lightcone = p2c.run_lightcone(redshift=z_min, max_redshift=z_max, user_params=params, astro_params=a_params,
                                cosmo_params=c_params, lightcone_quantities=("brightness_temp", 'xH_box'), global_quantities=("brightness_temp", 'xH_box'), 
                                direc=path_chache, random_seed=rseed)
    
    lc_noise = t2c.noise_lightcone(ncells = lightcone.brightness_temp.shape[0], zs = lightcone.lightcone_redshifts, 
                                    obs_time = tobs, save_uvmap = uvfile, boxsize = params['BOX_LEN'])
    
    dT2 = t2c.subtract_mean_signal(lightcone.brightness_temp, los_axis=2) 
    dT2_smt, redshifts = t2c.smooth_lightcone(dT2, z_array=lightcone.lightcone_redshifts, box_size_mpc=params['BOX_LEN'])
    dT3, redshifts = t2c.smooth_lightcone(dT2 + lc_noise, z_array=lightcone.lightcone_redshifts, box_size_mpc=params['BOX_LEN'])

    smt_xn, redshifts = t2c.smooth_lightcone(lightcone.xH_box, z_array=lightcone.lightcone_redshifts, box_size_mpc=params['BOX_LEN'])
    mask_xn = smt_xn>0.5
    
    # mask are saved such that 1 in neutral region and 0 in ionized region
    t2c.save_cbin(filename=path_out+'data/lc_256Mpc_dT%d.dat' %i, data=dT2_smt)
    t2c.save_cbin(filename=path_out+'data/lc_256Mpc_dT3%d.dat' %i, data=dT3)
    t2c.save_cbin(filename=path_out+'data/lc_256Mpc_mask%d.dat' %i, data=mask_xn)

    if(MAKE_PLOT):
        idx_plot = lightcone.brightness_temp.shape[-1]//2
        fig = plt.figure(figsize=(22, 15))
        gs = gridspec.GridSpec(nrows=3, ncols=2, width_ratios=[3,1], height_ratios=[1, 1, 1])
        ax0 = fig.add_subplot(gs[0,0])
        ax0.set_title('$\zeta$ = %.3f        $R_{mfp}$ = %.3f Mpc        $log_{10}(T_{vir}^{min})$ = %.3f        $t_{obs}=%d\,h$' %(zeta, Rmfp, Tvir, tobs), fontsize=18)
        ax0.imshow(dT3[:,64,:], cmap='jet', origin='lower')
        ax01 = fig.add_subplot(gs[0,1])
        ax01.set_title('$z$ = %.3f   $\delta T_b$=%.3f' %(lightcone.lightcone_redshifts[i], np.mean(dT3[:,:,i])), fontsize=18)
        ax01.imshow(dT3[:,:,idx_plot], cmap='jet', extent=my_ext, origin='lower')
        ax1 = fig.add_subplot(gs[1,0])
        ax1.imshow(mask_xn[:,64,:], cmap='jet', origin='lower')
        ax11 = fig.add_subplot(gs[1,1])
        ax11.set_title('$z$ = %.3f   $x^v_{HI}$=%.3f' %(lightcone.lightcone_redshifts[i], np.mean(mask_xn[:,:,i])), fontsize=18)
        ax11.imshow(mask_xn[:,:,idx_plot], cmap='jet', extent=my_ext, origin='lower')
        ax2 = fig.add_subplot(gs[2,0])
        ax2.imshow(dT2_smt[:,64,:], cmap='jet', origin='lower')
        ax21 = fig.add_subplot(gs[2,1])
        ax21.set_title('$z$ = %.3f   $\delta T_b$ =%.3f' %(lightcone.lightcone_redshifts[i], np.mean(dT2[:,:,i])), fontsize=18)
        ax21.imshow(dT2_smt[:,:,idx_plot], cmap='jet', extent=my_ext, origin='lower')

        idx_x = np.linspace(0, lightcone.shape[-1]-1, 7, endpoint=True, dtype=int)
        idx_y = np.linspace(0, lightcone.shape[0]-1, 7, endpoint=True, dtype=int)
        for ax in [ax0, ax1, ax2]:
            ax.set_xlabel('z', size=16)
            ax.set_ylabel('x [Mpc]', size=16)
            ax.set_xticks(idx_x), ax.set_yticks(idx_y);
            ax.set_xticklabels([round(zl, 2) for zl in redshifts[idx_x]]);
            ax.set_yticklabels(np.array(idx_y*lightcone.cell_size, dtype=int));
            #ax.label_outer()
            ax.tick_params(axis='both', length=5, width=1.2)
            ax.tick_params(which='minor', axis='both', length=5, width=1.2)

        for ax in [ax01, ax11, ax21]:
            ax.set_ylabel('y [Mpc]', size=16)
            ax.set_xlabel('x [Mpc]', size=16)

        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.linewidth'] = 1.2
        plt.subplots_adjust(hspace=0.3, wspace=0.05)
        plt.savefig(path_out+'images/lc_256Mpc_%d.png' %i, bbox_inches='tight'), plt.close()

    
    # save parameters values
    with open('%sastro_params_rank%d.txt' %(path_out+'parameters/', rank), 'a') as f:
        if(i == 0 and rank == 0):
            f.write('# HII_EFF_FACTOR: The ionizing efficiency of high-z galaxies\n')
            f.write('# R_BUBBLE_MAX: Mean free path in Mpc of ionizing photons within ionizing regions\n')
            f.write('# ION_Tvir_MIN: Minimum virial Temperature of star-forming haloes in log10 units\n')
            f.write('#i\teff_f\tRmfp\tTvir\tseed\n')

        f.write('%d\t%.3f\t%.3f\t%.3f\t%d\n' %(i, eff_fact, Rmfp, Tvir, rseed))
    """
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
    """
    
    # Create 21cmFast LC
    if(rank == 0):
        try:
            os.system('rm %s*h5' %path_chache)
        except:
            pass

    # update while loop index
    i += 1
print('... rank=%d finished' %rank)
'''