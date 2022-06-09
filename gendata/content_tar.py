import numpy as np, tarfile, pandas as pd
from other_utils import Timer 
from glob import glob

t = Timer() 
path = '/cosma6/data/dp004/dc-bian1/inputs/dataLC_128_valid_060921/'
name_tar = path[path[:-1].rfind('/')+1:-1]
arr_tar = glob(path+'data/*tar.gz')

t.start()
for i_part in range(len(arr_tar)):
    part = '%sdata/%s_part%d.tar.gz' %(path, name_tar, i_part)
    #var = path+'data/dataLC_128_valid_060921_part1.tar.gz' 
    #content = np.loadtxt(path+'content.txt', dtype=int) 

    mytar = tarfile.open(part, 'r')
    tar_content = mytar.getmembers()
    tar_names = mytar.getnames()
    np.save('%sdata/tar_content_part%d' %(path, i_part), tar_content)
    np.save('%sdata/tar_names_part%d' %(path, i_part), tar_names)
    mytar.close()
    t.lap('%s_part%d.tar.gz' %(name_tar, i_part))
    
t.stop()