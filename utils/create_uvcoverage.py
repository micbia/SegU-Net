import numpy as np, os, tools21cm as t2c
from tqdm import tqdm

params = {'HII_DIM':128, 'DIM':384, 'BOX_LEN':256}
redshift = np.arange(7.001, 20.001, 0.001)

out_path = './'

print("\nCalculate uv-coverage...")
print("z = [%.3f, %.3f] for a total of %d redshift\n" %(redshift.min(), redshift.max(), redshift.size))

for z in redshift:
    file_uv, file_Nant = '%suv_coverage_%d/uvmap_z%.3f.npy' %(out_path, params['HII_DIM'], z), '%suv_coverage_%d/Nantmap_z%.3f.npy' %(out_path, params['HII_DIM'], z)

    if not (os.path.exists(file_uv) and os.path.exists(file_Nant)):
        print('z = %.3f' %z)
        # total_int_time : Observation per day in hours.
        # int_time : Time period of recording the data (sec).
        uv, Nant = t2c.get_uv_daily_observation(params['HII_DIM'], z, filename=None, total_int_time=6.0, int_time=10.0, boxsize=params['BOX_LEN'], declination=-30.0, verbose=True)

        np.save(file_uv, uv)
        np.save(file_Nant, Nant)

    print('\n')
print("...done")
