import numpy as np, matplotlib.pyplot as plt, os, sys
import tools21cm as t2c, py21cmfast as p21c
import matplotlib.gridspec as gridspec
import random, json
from datetime import datetime
from glob import glob
from tqdm import tqdm

path_out = sys.argv[1]
path_out += '/' if path_out[-1]!='/' else ''
tot_loop = 3000

def GenerateSeed():
    # create seed for 21cmFast
    seed = [var for var in datetime.now().strftime('%d%H%M%S')]
    np.random.shuffle(seed)
    return int(''.join(seed))
    
params = {'HII_DIM':128, 'DIM':384, 'BOX_LEN':256}
#params = {'HII_DIM':64, 'DIM':192, 'BOX_LEN':128}
c_params = {'OMm':0.27, 'OMb':0.046, 'SIGMA_8':0.82, 'POWER_INDEX':0.96}
my_ext = [0, params['BOX_LEN'], 0, params['BOX_LEN']]
tobs = 1000

# create path output
if not (os.path.exists(path_out)):
    path_out = datetime.now().strftime('data3D_'+str(params['HII_DIM'])+'_%d%m%y/') 
    os.makedirs(path_out)
    os.makedirs(path_out+'images')
    os.makedirs(path_out+'data')
    resume_step = 0
else:
    print('Output path already exist.')
    resume_step = len(glob(path_out+'data/dT1_21cm_i*.bin')) 
    print('Continuing from itration i=%d.' %resume_step)
    assert resume_step != 0

# run 21cmFast
with open(path_out+'user_params.txt', 'w') as file:
     file.write(json.dumps(params))

with open(path_out+'cosm_params.txt', 'w') as file:
     file.write(json.dumps(c_params))

#redshift = np.linspace(7., 9., 100)
redshift = np.arange(7.001, 9.001, 0.001)

if(resume_step == 0):
    i = 0
    pbar = tqdm(total=tot_loop)
else:
    i = resume_step
    pbar = tqdm(total=tot_loop-resume_step)

while i < tot_loop:
    #z = random.choices(redshift, k=1)  # z = [7, 9]
    z = np.random.uniform(7, 9)
    eff_fact = random.gauss(52.5, 20.)  # eff_fact = [5, 100]
    Rmfp = random.gauss(12.5, 5.)       # Rmfp = [5, 20]
    Tvir = random.gauss(4.65, 0.5)      # Tvir = [log10(1e4), log10(2e5)]
    
    """
    # calculate uv-coverage 
    file_uv, file_Nant = 'uv_coverage_%d/uvmap_z%.3f.npy' %(params['HII_DIM'], z), 'uv_coverage_%d/Nantmap_z%.3f.npy' %(params['HII_DIM'], z)

    if(os.path.exists(file_uv) and os.path.exists(file_Nant)):
        uv = np.load(file_uv)
        Nant = np.load(file_Nant)
    else:
        uv, Nant = t2c.get_uv_daily_observation(params['HII_DIM'], z,
                                                filename=None,      # If None, it uses the SKA-Low 2016 configuration.
                                                total_int_time=6.0, # Observation per day in hours.
                                                int_time=10.0,      # Time period of recording the data in seconds.
                                                boxsize=params['BOX_LEN'], declination=-30.0, verbose=True)
        
        np.save(file_uv, uv)
        np.save(file_Nant, Nant)
    
               
    # Noise cube
    np.random.seed(GenerateSeed())
    noise_cube = t2c.noise_cube_coeval(params['HII_DIM'], z, depth_mhz=None,
                                       obs_time=tobs, filename=None, boxsize=params['BOX_LEN'],
                                       total_int_time=6.0, int_time=10.0, declination=-30.0, 
                                       uv_map=uv, N_ant=Nant, fft_wrap=False, verbose=False)
    """
    
    # Define astronomical parameters
    a_params = {'HII_EFF_FACTOR':eff_fact, 'R_BUBBLE_MAX':Rmfp, 'ION_Tvir_MIN':Tvir}

    # Create 21cmFast cube
    if(i%1 == 0):
        ic = p21c.initial_conditions(user_params=params, cosmo_params=c_params, random_seed=GenerateSeed())
        os.system('rm /home/michele/21CMMC_Boxes/*h5')
    cube = p21c.run_coeval(redshift=z, init_box=ic, astro_params=a_params, zprime_step_factor=1.05)

    # Mean neutral fraction
    xn = np.mean(cube.xH_box)

    if(xn > 0.1 and xn <= 0.8):
        dT = cube.brightness_temp
        """
        ps, ks, n_modes = t2c.power_spectrum_1d(dT, kbins=20, box_dims=cube.user_params.BOX_LEN,return_n_modes=True, binning='log')

        fig = plt.figure(figsize=(11, 4))
        fig.suptitle('z=%.3f   $x_n$=%.2f   $\zeta$=%.3f   $R_{mfp}$=%.3f   $T_{vir}^{min}$=%.3f' %(z, xn, eff_fact, Rmfp, Tvir), fontsize=15)
        gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1.5, 1])
        ax0 = fig.add_subplot(gs[0,0])
        ax0.loglog(ks, ps*ks**3/2/np.pi**2)
        ax0.set_xlabel('k (Mpc$^{-1}$)', fontsize=12), ax0.set_ylabel('$\Delta^2_\mathrm{21}$', fontsize=12)
        ax1 = fig.add_subplot(gs[0,1])
        ax1.imshow(dT[:,:,10], origin='lower', cmap='jet')
        plt.savefig(path_out+'images/test_i%d.png' %i, bbox_inches='tight'), plt.close()
        """
        dT1 = t2c.subtract_mean_channelwise(dT, axis=2)
        """
        dT2 = dT1 + noise_cube

        # Smooth the data to resolution corresponding to maximum baseline of 2 km
        dT3 = t2c.smooth_coeval(dT2, cube.redshift, box_size_mpc=cube.user_params.HII_DIM, max_baseline=2.0, ratio=1.0, nu_axis=2)

        smt_xn = t2c.smooth_coeval(cube.xH_box, cube.redshift, box_size_mpc=cube.user_params.HII_DIM, max_baseline=2.0, ratio=1.0, nu_axis=2)
        mask_xn = smt_xn>0.5
        """
        # mask are saved such that 1 in neutral region and 0 in ionized region
        t2c.save_cbin(path_out+'data/xH_21cm_i%d.bin' %i, cube.xH_box)
        t2c.save_cbin(path_out+'data/dT1_21cm_i%d.bin' %i, dT1)

        # Plot outputs comparisons
        fig, axs = plt.subplots(1, 2, figsize=(12,7))
        idx=10
        fig.suptitle('z=%.3f\t\t$x_n$=%.2f\n$\zeta$=%.3f\t\t$R_{mfp}$=%.3f\t\t$T_{vir}^{min}$=%.3f' %(z, xn, eff_fact, Rmfp, Tvir), fontsize=18)
        axs[0].set_title('$x_{HII}$', size=16)
        axs[0].imshow(1-cube.xH_box[:,:,idx], origin='lower', cmap='jet', extent=my_ext)
        axs[0].set_xlabel('[Mpc]'), axs[0].set_ylabel('[Mpc]');

        axs[1].set_title('$\delta T_b$', size=16)
        axs[1].imshow(dT1[:,:,idx], origin='lower', cmap='jet', extent=my_ext)
        axs[1].set_xlabel('[Mpc]'), axs[1].set_ylabel('[Mpc]');
        plt.savefig(path_out+'images/slice_i%d.png' %i, bbox_inches='tight'), plt.close()
        
        # save parameters values
        with open(path_out+'astro_params.txt', 'a') as f:
            if(i==0 and resume_step==0):
                f.write('# HII_EFF_FACTOR: The ionizing efficiency of high-z galaxies\n')
                f.write('# R_BUBBLE_MAX: Mean free path in Mpc of ionizing photons within ionizing regions\n')
                f.write('# ION_Tvir_MIN: Minimum virial Temperature of star-forming haloes in log10 units\n')
                f.write('#i\tz\teff_f\tRmfp\tTvir\tx_n\n')
            else:
                pass
            f.write('%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' %(i, z, eff_fact, Rmfp, Tvir, np.mean(xn)))
        
        pbar.update(1) 
        i += 1
    else:
        continue
    
pbar.close()
print('...Finished')
