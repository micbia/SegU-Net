import numpy as np, os
import matplotlib.pyplot as plt
import tools21cm as t2c

from tqdm import tqdm
from sklearn.decomposition import PCA as sciPCA
path_in = '/store/ska/sk02/lightcones/EOS21/test_dataset/'
path_out = '/scratch/snx3000/mibianco/test_pca/'
nr = 4

for i in tqdm(range(1)):
    fin = path_in+'dT4pca%d_21cm_i%d.bin' %(nr, i)
    fout = path_out+'dT4pca%d_21cm_i%d.bin' %(nr, i)
    if not (os.path.exists(fin) or os.path.exists(fout)):
        dT4 = t2c.read_cbin(path_in+'dT4_21cm.bin')
        data_flat = np.reshape(dT4, (-1, dT4.shape[2]))
        pca = sciPCA(n_components=nr)
        datapca = pca.fit_transform(data_flat)
        pca_FG = pca.inverse_transform(datapca)
        dT4pca = np.reshape(data_flat - pca_FG, dT4.shape)
        t2c.save_cbin(fout, dT4pca)
    else:
        dT4pca = t2c.read_cbin(fout)

plt.imshow(dT4pca[:,64,:], cmap='jet', aspect='auto')
plt.show(), plt.clf()