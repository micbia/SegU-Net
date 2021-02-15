import numpy as np, matplotlib.pyplot as plt, os
import tools21cm as t2c, py21cmfast as p21c
from py21cmfast import plotting
import random, zipfile
from astropy.cosmology import FlatLambdaCDM
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef
import matplotlib.gridspec as gridspec
from itertools import permutations
from datetime import datetime
import pickle
from utils_network.prediction import SegUnet21cmPredict
from utils.other_utils import RescaleData


uvfile = '/home/michele/Documents/PhD_Sussex/output/ML/dataset/inputs/uvmap_128_z7-20.pkl'
if not (os.path.exists(uvfile)):
    print('uv-map pickle not found')
else:
    uvs = pickle.load(open(uvfile, 'rb'))
    
# create seed for 21cmFast
str_seed = [var for var in datetime.now().strftime('%d%H%M%S')]
np.random.shuffle(str_seed)
seed = int(''.join(str_seed))

outpath = '/home/michele/Documents/PhD_Sussex/output/ML/dataset/outputs/new/02-10T23-52-36_128slice/predictions_1/compare_tobs/'

title_a = '\t\t _    _ _   _      _   \n\t\t| |  | | \ | |    | |  \n\t\t| |  | |  \| | ___| |_ \n\t\t| |  | | . ` |/ _ \ __|\n\t\t| |__| | |\  |  __/ |_ \n\t\t \____/|_| \_|\___|\__|\n'
title_b = ' _____              _ _      _         ___  __                \n|  __ \            | (_)    | |       |__ \/_ |               \n| |__) | __ ___  __| |_  ___| |_ ___     ) || | ___ _ __ ___  \n|  ___/ `__/ _ \/ _` | |/ __| __/ __|   / / | |/ __| `_ ` _ \ \n| |   | | |  __/ (_| | | (__| |_\__ \  / /_ | | (__| | | | | |\n|_|   |_|  \___|\__,_|_|\___|\__|___/ |____||_|\___|_| |_| |_|\n'
print(title_a+'\n'+title_b)

from tensorflow.keras.models import load_model
from utils_network.metrics import iou, iou_loss, dice_coef, dice_coef_loss, balanced_cross_entropy, phi_coef
from config.net_config import NetworkConfig

def LoadSegUnetModel(cfile):
    avail_metrics = {'binary_accuracy':'binary_accuracy', 'iou':iou, 'dice_coef':dice_coef, 'iou_loss':iou_loss, 'dice_coef_loss':dice_coef_loss, 'phi_coef':phi_coef, 'mse':'mse', 'mae':'mae', 'binary_crossentropy':'binary_crossentropy', 'balanced_cross_entropy':balanced_cross_entropy}
    conf = NetworkConfig(conf_file)

    PATH_OUT = conf.resume_path
    MODEL_EPOCH = conf.best_epoch
    METRICS = [avail_metrics[m] for m in np.append(conf.loss, conf.metrics)]
    cb = {func.__name__:func for func in METRICS if not isinstance(func, str)}
    model_loaded = load_model('%smodel-sem21cm_ep%d.h5' %(PATH_OUT+'checkpoints/', MODEL_EPOCH), custom_objects=cb)
    
    print(' Loaded model:\n %smodel-sem21cm_ep%d.h5' %(conf.resume_path, MODEL_EPOCH))
    return model_loaded

# load model
conf_file = '/home/michele/Documents/PhD_Sussex/output/ML/SegNet/tests/runs/'
conf_file += 'net2D_021020.ini'
model = LoadSegUnetModel(conf_file)

tobs = np.array([200, 500, 700, 900, 1000, 1200, 2000])

xHI = np.array([0.2, 0.5, 0.8])
redshift = np.array([7.310, 8.032, 8.720])
idxs = np.array(range(redshift.size))
zeta = 39.204   #65.204
Rmfp = 12.861   #11.861
Tvir = 4.539    #4.539

params = {'HII_DIM':128, 'DIM':384, 'BOX_LEN':256}
my_ext = [0, params['BOX_LEN'], 0, params['BOX_LEN']]
c_params = {'OMm':0.27, 'OMb':0.046, 'SIGMA_8':0.82, 'POWER_INDEX':0.96}
a_params = {'HII_EFF_FACTOR':zeta, 'R_BUBBLE_MAX':Rmfp, 'ION_Tvir_MIN':Tvir}

# late, middle and early stage of reionization
bins = np.array([[0.18, 0.22],[0.48,0.52],[0.78,0.82]])
compare_tobs_mean = np.zeros((bins.shape[0], tobs.size))
compare_tobs_std = np.zeros((bins.shape[0], tobs.size))

ic = p21c.initial_conditions(user_params=params, cosmo_params=c_params, random_seed=2021)
compare_tobs = np.zeros((redshift.size, tobs.size))

for i in tqdm(range(len(redshift))):
    z = redshift[i]

    cube = p21c.run_coeval(redshift=z, init_box=ic, astro_params=a_params, zprime_step_factor=1.05)

    dT = cube.brightness_temp
    xH = cube.xH_box

    uv = uvs['%.3f' %z]
    Nant = uvs['Nant']

    print('z = %.3f  x_n = %.3f  zeta = %.3f  R_mfp = %.3f  T_vir = %.3f' %(z, np.mean(cube.xH_box), zeta, Rmfp, Tvir))
    
    for j, t in enumerate(tobs):
        noise_cube = t2c.noise_cube_coeval(params['HII_DIM'], z, depth_mhz=None, obs_time=t, filename=None, boxsize=params['BOX_LEN'], total_int_time=6.0, int_time=10.0, declination=-30.0, uv_map=uv, N_ant=Nant, verbose=True, fft_wrap=False)
        dT1 = t2c.subtract_mean_signal(dT, los_axis=2)
        dT2 = dT1 + noise_cube
        dT3 = t2c.smooth_coeval(dT2, z, box_size_mpc=params['HII_DIM'], max_baseline=2.0, ratio=1.0, nu_axis=2)
        smt_xn = t2c.smooth_coeval(xH, z, box_size_mpc=params['HII_DIM'], max_baseline=2.0, ratio=1.0, nu_axis=2)
        mask_xn = smt_xn>0.5

        # calculate error and prediction
        X_tta = SegUnet21cmPredict(unet=model, x=dT3, TTA=False)
        X_seg = np.round(np.mean(X_tta, axis=0))
        
        # calculate MCC score
        phicoef_seg = matthews_corrcoef(mask_xn.flatten(), X_seg.flatten())

        compare_tobs[i,j] = phicoef_seg
        np.savetxt('%scompare_tobs.txt' %(outpath), compare_tobs)
    
    os.system('rm /home/michele/21CMMC_Boxes/*h5')
