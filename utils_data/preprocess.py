import numpy as np, tools21cm as t2c
from ps_eor import datacube, fgfit, fitutil, pspec, psutil, ml_gpr
from scipy.integrate import quadrature

from tqdm import tqdm
from astropy.cosmology import LambdaCDM

z, depth_mhz = 8, 20
indexes = np.array([232])

path_input = '/store/ska/sk09/segunet/inputs/dataLC_128_pred_190922/'
path_out = '/store/ska/sk09/segunet/inputs/preprocess_dataLC_128_pred_190922/data/'
astro_par = np.loadtxt(path_input+'parameters/astro_params.txt')

with open(path_input+'parameters/user_params.txt', 'r') as file:
    user_par = eval(file.read())
    box_len = user_par['BOX_LEN']
    n_pix = user_par['HII_DIM']
    angl_size = np.linspace(0, box_len, n_pix)

with open(path_input+'parameters/cosm_params.txt', 'r') as file:
    cosm_par = eval(file.read())
    t2c.set_hubble_h(1)
    t2c.set_omega_lambda(1-cosm_par['OMm'])
    t2c.set_omega_matter(cosm_par['OMm'])
    t2c.set_omega_baryon(0.045)
    t2c.set_sigma_8(0.82)
    t2c.set_ns(0.97)
    psutil.set_cosmology(LambdaCDM(H0=100, Om0=cosm_par['OMm'], Ode0=1-cosm_par['OMm'], Ob0=0.045))

redshift = np.loadtxt(path_input+'lc_redshifts.txt')

for i in tqdm(indexes):
    data_dT2 = t2c.read_cbin(path_input+'data/dT2_21cm_i%d.bin' %(i))
    data_dT4 = t2c.read_cbin(path_input+'data/dT4_21cm_i%d.bin' %(i))
    data_dTskleanpca = t2c.read_cbin(path_input+'data/dT4pca4_21cm_i%d.bin' %(i))
    data_xH = t2c.read_cbin(path_input+'data/xH_21cm_i%d.bin' %(i))
    data_xHI = t2c.read_cbin(path_input+'data/xHI_21cm_i%d.bin' %(i))

    assert redshift[0] == redshift.min()
    assert redshift[-1] == redshift.max()

    #max_cdist = t2c.z_to_cdist(redshift.max()) # max redshi
    #min_cdist = t2c.z_to_cdist(redshift.min())
    #box_dims = [box_len, box_len, np.abs(min_cdist - max_cdist)]

    # The units assumed for box_dims defines the unit of k
    dT2_sub, box_dims, cut_redshift = t2c.get_lightcone_subvolume(lightcone=data_dT2, redshifts=redshift, central_z=z, depth_mhz=depth_mhz, odd_num_cells=False, subtract_mean=False, fov_Mpc=box_len)
    idx_i, idx_j = np.argmin(np.abs(cut_redshift.min() - redshift)), np.argmin(np.abs(cut_redshift.max() - redshift))
    cut_freqs = t2c.z_to_nu(cut_redshift)
    min_freqs, max_freqs = cut_freqs.min()*1e6, cut_freqs.max()*1e6
    
    t2c.save_cbin(path_out+'dT4pca4_z%d_%dMHz_i%d.bin' %(z, depth_mhz, i), data_dTskleanpca[...,idx_i:idx_j+1])
    t2c.save_cbin(path_out+'dT2_z%d_%dMHz_i%d.bin' %(z, depth_mhz, i), dT2_sub)
    t2c.save_cbin(path_out+'xHI_z%d_%dMHz_i%d.bin' %(z, depth_mhz, i), data_xHI[...,idx_i:idx_j+1])
    t2c.save_cbin(path_out+'xH_z%d_%dMHz_i%d.bin' %(z, depth_mhz, i), data_xH[...,idx_i:idx_j+1])
    t2c.save_cbin(path_out+'dT4_z%d_%dMHz_i%d.bin' %(z, depth_mhz, i), data_dT4[...,idx_i:idx_j+1])
    np.savetxt(path_out+'redshifts_z%d_%dMHz.txt' %(z, depth_mhz), cut_redshift)
    np.savetxt(path_out+'box_dims_z%d_%dMHz.txt' %(z, depth_mhz), box_dims, header='in cMpc')

    assert dT2_sub.shape == data_xHI[...,idx_i:idx_j+1].shape

    redshift_ps = redshift[::-1]
    freqs_ps = t2c.z_to_nu(redshift_ps)
    dT2_ps = np.moveaxis(data_dT2, -1, 0)[::-1]
    dT4_ps = np.moveaxis(data_dT4, -1, 0)[::-1]
    xH_ps = np.moveaxis(data_xH, -1, 0)[::-1]
    box_dims_ps = box_dims[::-1]

    FoV = t2c.angular_size_comoving(cMpc=box_len, z=redshift_ps).mean() # in [deg]

    res = np.radians(FoV) / n_pix # in [rad]
    df = abs(np.diff(freqs_ps).mean()) * 1e6 # in [Hz]

    meta = datacube.ImageMetaData.from_res(res, (n_pix, n_pix))
    meta.wcs.wcs.cdelt[2] = df

    # image cube in K, freqs in Hz
    image_cube2 = datacube.CartImageCube(dT2_ps * 1e-3, freqs_ps * 1e6, meta)
    image_cube4 = datacube.CartImageCube(dT4_ps * 1e-3, freqs_ps * 1e6, meta)

    # image to visibility, keeping modes between 30 to 500 wavelengths
    vis_cube2 = image_cube2.ft(30, 500)
    vis_cube4 = image_cube4.ft(30, 500)

    # Fake noise cube which is required for the FG fitting
    noise_cube = vis_cube2.new_with_data(np.random.randn(*vis_cube4.data.shape) * 1e-6)

    # POLYNOMIAL FITTING
    poly_fitter = fgfit.PolyForegroundFit(4, 'power_poly')
    poly_fit = poly_fitter.run(vis_cube4, noise_cube)
    img_cube_poly = poly_fit.sub.image()
    dT4poly = np.moveaxis(img_cube_poly.data[::-1], 0, 2).astype(np.float32) * 1e3 # in mK
    t2c.save_cbin('%sdT4poly_z%d_%dMHz_i%d.bin' %(path_out, z, depth_mhz, i), dT4poly[...,idx_i:idx_j+1])

    # PCA FITTING
    pca_fitter = fgfit.PcaForegroundFit(4)
    pca_fit = pca_fitter.run(vis_cube4, noise_cube)
    img_cube_pca = pca_fit.sub.image()
    dT4pca = np.moveaxis(img_cube_pca.data[::-1], 0, 2).astype(np.float32) * 1e3 # in mK
    t2c.save_cbin('%sdT4pcafit_z%d_%dMHz_i%d.bin' %(path_out, z, depth_mhz, i), dT4pca[...,idx_i:idx_j+1])

    # GPR FITTING
    vis_cube_s = vis_cube4.get_slice(min_freqs, max_freqs)
    noise_cube_s = noise_cube.get_slice(min_freqs, max_freqs)
    gpr_config = fitutil.GprConfig()
    gpr_config.fg_kern = fitutil.GPy.kern.RBF(1, name='fg')
    gpr_config.fg_kern.lengthscale.constrain_bounded(20, 60)
    gpr_config.eor_kern = fitutil.GPy.kern.Matern32(1, name='eor')
    gpr_config.eor_kern.lengthscale.constrain_bounded(0.2, 1.2)
    gpr_fitter = fgfit.GprForegroundFit(gpr_config)
    gpr_res_s = gpr_fitter.run(vis_cube_s, noise_cube_s)
    img_cube_gpr = gpr_res_s.sub.image()
    dT4gpr = np.moveaxis(img_cube_gpr.data[::-1], 0, 2).astype(np.float32) * 1e3 # in mK
    t2c.save_cbin('%sdT4gpr_z%d_%dMHz_i%d.bin' %(path_out, z, depth_mhz, i), dT4gpr)
    assert dT2_sub.shape == dT4gpr.shape

