import numpy as np, os, tools21cm as t2c
from tqdm import tqdm
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank() 
nprocs = comm.Get_size()

params = {'HII_DIM':128, 'DIM':384, 'BOX_LEN':256}
redshift = np.arange(7.001, 20.001, 0.001)

out_path = './'

print("\nCalculate uv-coverage...")
print("z = [%.3f, %.3f] for a total of %d redshift\n" %(redshift.min(), redshift.max(), redshift.size))

resume_step = 0
perrank = redshift.size//nprocs

for i in range(resume_step+rank*perrank, resume_step+(rank+1)*perrank):
    z = redshift[i]
    file_uv, file_Nant = '%suv_coverage_%d/uvmap_z%.3f.npy' %(out_path, params['HII_DIM'], z), '%suv_coverage_%d/Nantmap_z%.3f.npy' %(out_path, params['HII_DIM'], z)

    if not (os.path.exists(file_uv) and os.path.exists(file_Nant)):
        print(' z = %.3f\tprocessor: %d/%d' %(z, rank, nprocs))
        # total_int_time : Observation per day in hours.
        # int_time : Time period of recording the data (sec).
        uv, Nant = t2c.get_uv_daily_observation(params['HII_DIM'], z, filename=None, total_int_time=6.0, int_time=10.0, boxsize=params['BOX_LEN'], declination=-30.0, verbose=False)

        np.save(file_uv, uv)
        np.save(file_Nant, Nant)

    #print('\n')
#print("...done")
