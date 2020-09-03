import matplotlib as mpl
mpl.use('agg')
import numpy as np, matplotlib.pyplot as plt, os, sys
import tools21cm as t2c, py21cmfast as p21
import matplotlib.gridspec as gridspec
import random, json

from datetime import datetime
from glob import glob
from tqdm import tqdm
from mpi4py import MPI

#path_out = '/gpfs/scratch/userexternal/mbianco0/data3D_128_080820/'
path_out = sys.argv[1]
path_out += '/' if path_out[-1]!='/' else ''
path_chache = '/gpfs/scratch/userexternal/mbianco0/21cmFAST-cache/'
p21.config['direc'] = path_chache

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

loop_end = 7000
loop_start = 0
perrank = (loop_end-loop_start)//nprocs


def GenerateSeed():
    # create seed for 21cmFast
    seed = [var for var in datetime.now().strftime('%d%H%M%S')]
    np.random.shuffle(seed)
    return int(''.join(seed))


params = {'HII_DIM':128, 'DIM':384, 'BOX_LEN':256}
c_params = {'OMm':0.27, 'OMb':0.046, 'SIGMA_8':0.82, 'POWER_INDEX':0.96}
my_ext = [0, params['BOX_LEN'], 0, params['BOX_LEN']]
tobs = 1000


if not (os.path.exists('%sastro_params_rank%d.txt' %(path_out+'parameters/', rank))):
    loop_resume = 0
    
    with open(path_out+'parameters/user_params.txt', 'w') as file:
        file.write(json.dumps(params))

    with open(path_out+'parameters/cosm_params.txt', 'w') as file:
        file.write(json.dumps(c_params))
else:
    loop_resume = int(np.loadtxt('%sastro_params_rank%d.txt' %(path_out+'parameters/', rank))[:,0].max())


if(loop_resume == 0):
    i = loop_start+rank*perrank
elif(loop_resume != 0):
    print('Output path already exist.\nRank=%d resumes itration from i=%d.' %(rank, loop_resume))
    i = loop_resume + 1


while i < loop_start+(rank+1)*perrank:
    # astronomical & cosmological parameters
    z = np.random.uniform(7, 9)         # z = [7, 9]
    eff_fact = random.gauss(52.5, 20.)  # eff_fact = [5, 100]
    Rmfp = random.gauss(12.5, 5.)       # Rmfp = [5, 20]
    Tvir = random.gauss(4.65, 0.5)      # Tvir = [log10(1e4), log10(2e5)]
    
    # Define astronomical parameters
    a_params = {'HII_EFF_FACTOR':eff_fact, 'R_BUBBLE_MAX':Rmfp, 'ION_Tvir_MIN':Tvir}

    # Create 21cmFast cube
    if(i%1 == 0):
        try:
            os.system('rm %s*h5' %path_chache)
        except:
            pass
        comm.Barrier()
        
        ic = p21.initial_conditions(user_params=params, cosmo_params=c_params, random_seed=GenerateSeed())

    cube = p21.run_coeval(redshift=z, init_box=ic, astro_params=a_params, zprime_step_factor=1.05)

    # Mean neutral fraction
    xn = np.mean(cube.xH_box)

    if(xn > 0.1 and xn <= 0.8):
        print('processor: %d/%d   idx=%d\n z=%.3f, xn=%.3f, eff_fact=%.3f, Rmfp=%.3f, Tvir=%.3f' %(rank, nprocs, i, z, xn, eff_fact, Rmfp, Tvir))

        dT = cube.brightness_temp
        dT1 = t2c.subtract_mean_signal(signal=dT, los_axis=2)
        
        """
        # calculate uv-coverage 
        file_uv, file_Nant = 'uv_coverage_%d/uvmap_z%.3f.npy' %(params['HII_DIM'], z), 'uv_coverage_%d/Nantmap_z%.3f.npy' %(params['HII_DIM'], z)

        if(os.path.exists(file_uv) and os.path.exists(file_Nant)):
            uv = np.load(file_uv)
            Nant = np.load(file_Nant)
        else:
            uv, Nant = t2c.get_uv_daily_observation(params['HII_DIM'], z, filename=None, total_int_time=6.0, int_time=10.0, boxsize=params['BOX_LEN'], declination=-30.0, verbose=True)
            
            np.save(file_uv, uv)
            np.save(file_Nant, Nant)
        
                
        # Noise cube
        np.random.seed(GenerateSeed())
        noise_cube = t2c.noise_cube_coeval(params['HII_DIM'], z, depth_mhz=None, obs_time=tobs, filename=None, boxsize=params['BOX_LEN'], total_int_time=6.0, int_time=10.0, declination=-30.0, uv_map=uv, N_ant=Nant, fft_wrap=False, verbose=False)
        
        dT2 = dT1 + noise_cube

        # Smooth the data to resolution corresponding to maximum baseline of 2 km
        dT3 = t2c.smooth_coeval(dT2, cube.redshift, box_size_mpc=cube.user_params.HII_DIM, max_baseline=2.0, ratio=1.0, nu_axis=2)

        smt_xn = t2c.smooth_coeval(cube.xH_box, cube.redshift, box_size_mpc=cube.user_params.HII_DIM, max_baseline=2.0, ratio=1.0, nu_axis=2)
        mask_xn = smt_xn>0.5
        """
        # mask are saved such that 1 in neutral region and 0 in ionized region
        t2c.save_cbin(path_out+'data/xH_21cm_i%d.bin' %i, cube.xH_box)
        t2c.save_cbin(path_out+'data/dT1_21cm_i%d.bin' %i, dT1)
        
        if(i%50):
            ps, ks, n_modes = t2c.power_spectrum_1d(dT, kbins=20, box_dims=cube.user_params.BOX_LEN,return_n_modes=True, binning='log')
            idx=params['HII_DIM']//2

            fig = plt.figure(figsize=(11, 4))
            fig.suptitle('z=%.3f   $x_n$=%.2f   $\zeta$=%.3f   $R_{mfp}$=%.3f   $T_{vir}^{min}$=%.3f' %(z, xn, eff_fact, Rmfp, Tvir), fontsize=15)
            gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1.5, 1])
            ax0 = fig.add_subplot(gs[0,0])
            ax0.loglog(ks, ps*ks**3/2/np.pi**2)
            ax0.set_xlabel('k (Mpc$^{-1}$)', fontsize=12), ax0.set_ylabel('$\Delta^2_\mathrm{21}$', fontsize=12)
            ax1 = fig.add_subplot(gs[0,1])
            ax1.imshow(dT[:,:,idx], origin='lower', cmap='jet')
            plt.savefig(path_out+'images/test_i%d.png' %i, bbox_inches='tight'), plt.close()
        
            # Plot outputs comparisons
            fig, axs = plt.subplots(1, 2, figsize=(12,7))
            fig.suptitle('z=%.3f\t\t$x_n$=%.2f\n$\zeta$=%.3f\t\t$R_{mfp}$=%.3f\t\t$T_{vir}^{min}$=%.3f' %(z, xn, eff_fact, Rmfp, Tvir), fontsize=18)
            axs[0].set_title('$x_{HII}$', size=16)
            axs[0].imshow(1-cube.xH_box[:,:,idx], origin='lower', cmap='jet', extent=my_ext)
            axs[0].set_xlabel('[Mpc]'), axs[0].set_ylabel('[Mpc]');

            axs[1].set_title('$\delta T_b$', size=16)
            axs[1].imshow(dT1[:,:,idx], origin='lower', cmap='jet', extent=my_ext)
            axs[1].set_xlabel('[Mpc]'), axs[1].set_ylabel('[Mpc]');
            plt.savefig(path_out+'images/slice_i%d.png' %i, bbox_inches='tight'), plt.close()
        
        # save parameters values
        with open('%sastro_params_rank%d.txt' %(path_out+'parameters/', rank), 'a') as f:
            if(i == 0 and rank == 0):
                f.write('# HII_EFF_FACTOR: The ionizing efficiency of high-z galaxies\n')
                f.write('# R_BUBBLE_MAX: Mean free path in Mpc of ionizing photons within ionizing regions\n')
                f.write('# ION_Tvir_MIN: Minimum virial Temperature of star-forming haloes in log10 units\n')
                f.write('#i\tz\teff_f\tRmfp\tTvir\tx_n\n')

            f.write('%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' %(i, z, eff_fact, Rmfp, Tvir, np.mean(xn)))
        
        # if output directory is more than 15 GB of size, compress and remove files in data/
        if(os.path.getsize(path_out)/1e9 > 15):
            if(rank == 0):
                compr_dir = len(glob(path_out+'../*tar.gz'))
                os.system('tar -czvf %s %s' %(path_out, path_out[path_out[:-1].rfind('/')+1:-1]))
                os.system('rm %sdata/*.bin')
            comm.Barrier()

        # update while loop index
        i += 1
    else:
        continue
    
print('...Finished')
