import numpy as np, os
import zipfile
import tarfile 

path = '/cosma6/data/dp004/dc-bian1/inputs/dataLC_128_train_060921/'
content = np.zeros(10000)

for i in range(1,46): 
    var = '%sdata/dataLC_128_train_060921_part%d.tar.gz' %(path, i) 
    myzip = tarfile.open(var, 'r') 
    listname = myzip.getnames() 
    myzip.close() 
    for nms in listname:  
        if('dT3' in nms): 
            idx = int(nms[nms.rfind('_i')+2:nms.rfind('.')]) 
            content[idx] = i 
    np.savetxt('%scontent.txt' %(path+'data/'), content, fmt='%d') 
