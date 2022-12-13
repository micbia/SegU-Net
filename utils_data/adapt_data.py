import tools21cm as t2c, numpy as np

path_in = '/store/ska/sk02/lightcones/EOS21/test_dataset/'
path_out = '/scratch/snx3000/mibianco/test_skalow/data/'
fout = 'xHI'
print(fout)

#data = np.load(path_in + fout + '_EOS21_EoR.npy')
data = t2c.read_cbin(path_in+fout+'_21cm.bin')
print(data.shape)

nr_cuts = data.shape[0]//128
print(nr_cuts)

if(data.shape[0] % 128 != 0):
    datastep1 = np.vstack((data, data[0:128*(nr_cuts+1)-data.shape[0], :,:]))
    print(datastep1.shape)
    data = np.hstack((datastep1, datastep1[:, 0:128*(nr_cuts+1)-data.shape[1], :]))
else:
    pass

nr_cuts = data.shape[0]//128
print(nr_cuts)
print(data.shape)

data_cuts = np.zeros((nr_cuts**2, 128, 128, data.shape[2]), dtype=np.float32)
index_cuts = np.zeros((nr_cuts**2, 4), dtype=np.int)

count = 0
for i in range(nr_cuts):
    i_start, i_end = i*128, (i+1)*128
    for j in range(nr_cuts):
        j_start, j_end = j*128, (j+1)*128
        #data_cuts[count] = data[i_start:i_end, j_start:j_end, :]
        t2c.save_cbin('%s_21cm_i%d.bin' %(path_out+fout, count), data[i_start:i_end, j_start:j_end, :])
        index_cuts[count] = [i_start, i_end, j_start, j_end]
        count += 1

#np.save('%sdata/xHI_21cm.npy' %path_out, data_cuts)
#np.savetxt('%sindex_cuts.txt' %path_out, index_cuts, fmt='%d', header='i_start  i_end j_start j_end')
