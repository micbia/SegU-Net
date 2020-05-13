import os, random, numpy as np
from other_utils import get_data, save_cbin
from tqdm import tqdm

PATH = '/home/michele/Documents/PhD_Sussex/output/ML/outputs/outputs_Segnet/inputs/data3D_128_180320_valid/'
PATH_OUT = '/home/michele/Documents/PhD_Sussex/output/ML/outputs/outputs_Segnet/inputs/data2D_128_180320_valid/'

try:
    os.makedirs(PATH_OUT)
    os.makedirs(PATH_OUT+'/data')
except:
    pass

IM_SHAPE = (128, 128, 128)
X, Y = get_data(PATH+'data/', IM_SHAPE)

for index in tqdm(range(X.shape[0])):
    x, y = X[index], Y[index]
    idx = 0
    for i in range(IM_SHAPE[0]):
        save_cbin('%sdata/images_21cm_i%d.bin' %(PATH_OUT, idx), x.squeeze()[i,:,:])
        save_cbin('%sdata/masks_21cm_i%d.bin' %(PATH_OUT, idx), y.squeeze()[i,:,:])
        idx += 1
    for j in range(IM_SHAPE[1]):
        save_cbin('%sdata/images_21cm_i%d.bin' %(PATH_OUT, idx), x.squeeze()[:,j,:])
        save_cbin('%sdata/masks_21cm_i%d.bin' %(PATH_OUT, idx), y.squeeze()[:,j,:])
        idx += 1
    for k in range(IM_SHAPE[2]):
        save_cbin('%sdata/images_21cm_i%d.bin' %(PATH_OUT, idx), x.squeeze()[:,:,k])
        save_cbin('%sdata/masks_21cm_i%d.bin' %(PATH_OUT, idx), y.squeeze()[:,:,k])
        idx += 1
