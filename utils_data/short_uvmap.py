import numpy as np, pickle 
uvs = pickle.load(open('uvmap_128_z7-20.pkl', 'rb')) 
redshift = np.arange(7,11.5005,0.001) 
uvs_short = {} 
for i in range(redshift.size+1): 
    if(i < redshift.size): 
        z = redshift[i] 
        uvs_short['%.3f' %z] = uvs['%.3f' %z] 
    else: 
        uvs_short['Nant'] = uvs['Nant'] 
pickle.dump(uvs, open('uvmap_128_z%d-%d.pkl' %(redshift.min(), redshift.max()), 'wb')) 
