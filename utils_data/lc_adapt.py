import numpy as np
import tools21cm as t2c

res = 1600/1024
params = {'HII_DIM':128, 'BOX_LEN':res*128}
nr = 4

path_in = '/store/ska/sk02/lightcones/EOS16/'
path_out = '/scratch/snx3000/mibianco/test_segunet/'
#uvfile = '/store/ska/sk02/lightcones/EOS21/uvmap_1000_z7-11.pkl'
uvfile = '/store/ska/sk02/lightcones/EOS16/uvmap_1024_z7-11.pkl'
tobs = 1000

print('load dT')
dT = np.load(path_in+'dT_EOS16_EoR.npy')
xHI = np.load(path_in+'xHI_EOS16_EoR.npy')
redshift = np.loadtxt(path_in+'redshift_EOS16_EoR.txt')
lc_noise = t2c.noise_lightcone(ncells=dT.shape[0], zs=redshift, obs_time=tobs, boxsize=1600., save_uvmap=uvfile, n_jobs=1)
gal_fg = t2c.galactic_synch_fg(z=redshift, ncells=1024, boxsize=1600., rseed=2023)

nr_cuts = dT.shape[0]//128
print(nr_cuts)

if(dT.shape[0] % 128 != 0):
    dTstep1 = np.vstack((dT, dT[0:128*(nr_cuts+1)-dT.shape[0], :,:]))
    dT = np.hstack((dTstep1, dTstep1[:, 0:128*(nr_cuts+1)-dT.shape[1], :]))

    xHIstep1 = np.vstack((xHI, xHI[0:128*(nr_cuts+1)-xHI.shape[0], :,:]))
    xHI = np.hstack((xHIstep1, xHIstep1[:, 0:128*(nr_cuts+1)-xHI.shape[1], :]))

    lc_noisestep1 = np.vstack((lc_noise, lc_noise[0:128*(nr_cuts+1)-lc_noise.shape[0], :,:]))
    lc_noise = np.hstack((lc_noisestep1, lc_noisestep1[:, 0:128*(nr_cuts+1)-lc_noise.shape[1], :]))
else:
    pass

nr_cuts = dT.shape[0]//128
index_cuts = np.zeros((nr_cuts**2, 4), dtype=np.int)

count = 0
for i in range(nr_cuts):
    i_start, i_end = i*128, (i+1)*128
    for j in range(nr_cuts):
        j_start, j_end = j*128, (j+1)*128
        subvol_dT = dT[i_start:i_end, j_start:j_end, :]
        subvol_xHI = xHI[i_start:i_end, j_start:j_end, :]
        sub_lc_noise = lc_noise[i_start:i_end, j_start:j_end, :]
        sub_gal_fg = gal_fg[i_start:i_end, j_start:j_end, :]

        index_cuts[count] = [i_start, i_end, j_start, j_end]

        dT1 = t2c.subtract_mean_signal(subvol_dT, los_axis=2)
        dT2, redshifts = t2c.smooth_lightcone(dT1, z_array=redshift, box_size_mpc=params['BOX_LEN'])
        dT3, _ = t2c.smooth_lightcone(dT1 + sub_lc_noise, z_array=redshifts, box_size_mpc=params['BOX_LEN'])

        #rseed = np.random.randint(0, 1e6)
        #gal_fg = t2c.galactic_synch_fg(z=redshift, ncells=params['HII_DIM'], boxsize=params['BOX_LEN'], rseed=rseed)
        #dT4, _ = t2c.smooth_lightcone(t2c.subtract_mean_signal(subvol_dT+gal_fg+sub_lc_noise, los_axis=2), z_array=redshift, box_size_mpc=params['BOX_LEN'])
        dT4, _ = t2c.smooth_lightcone(t2c.subtract_mean_signal(subvol_dT+sub_gal_fg+sub_lc_noise, los_axis=2), z_array=redshift, box_size_mpc=params['BOX_LEN'])

        smt_xn, redshifts = t2c.smooth_lightcone(subvol_xHI, z_array=redshift, box_size_mpc=params['BOX_LEN'])
        mask_xH = smt_xn>0.5

        t2c.save_cbin('%sdT2_21cm_i%d.bin' %(path_out, count), dT2)
        t2c.save_cbin('%sdT3_21cm_i%d.bin' %(path_out, count), dT3)
        t2c.save_cbin('%sdT4_21cm_i%d.bin' %(path_out, count), dT4)
        t2c.save_cbin('%sxHI_21cm_i%d.bin' %(path_out, count), xHI)
        t2c.save_cbin('%sxH_21cm_i%d.bin' %(path_out, count), mask_xH)
        count += 1

np.savetxt(path_out+'lc_redshifts.txt', redshifts)