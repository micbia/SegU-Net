from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

redshift = np.arange(7.001, 7.021, 0.001)

resume_step = 0
perrank = redshift.size//nprocs

print('processor:', rank, 'has', range(rank*perrank, (rank+1)*perrank))
"""
for i in range(resume_step+rank*perrank, resume_step+(rank+1)*perrank):
	z = redshift[i]
	print(' redshift %.3f\tprocessor: %d/%d' %(z, rank, nprocs))
"""

i = resume_step+rank*perrank
while i < resume_step+(rank+1)*perrank:
	z = redshift[i]
	print(' redshift %.3f\tprocessor: %d/%d' %(z, rank, nprocs))
	i += 1
