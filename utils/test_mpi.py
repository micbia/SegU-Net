from mpi4py import MPI
import numpy as np, time
from datetime import datetime

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

redshift = np.arange(7.001, 7.021, 0.001)

resume_step = 0
perrank = redshift.size//nprocs

if(rank == 0):
	print('Start test_mpi.py:\t%s\n' %(datetime.now().strftime('%H:%M:%S')))

comm.Barrier()
if(rank == 0):
	time.sleep(10)
	print(' rank %d start:\t%s' %(rank, datetime.now().strftime('%H:%M:%S')))
elif(rank == 1): 
	time.sleep(5)
	print(' rank %d start:\t%s' %(rank, datetime.now().strftime('%H:%M:%S')))
else:
	print(' rank %d start:\t%s' %(rank, datetime.now().strftime('%H:%M:%S')))

comm.Barrier()
print(' rank %d end:\t%s\n' %(rank, datetime.now().strftime('%H:%M:%S')))

"""
if(rank == 0):
	for i in range(10):
		print('count: %d' %i)

comm.Barrier()
print('rank %d resume...' %rank)

print('processor:', rank, 'has', range(rank*perrank, (rank+1)*perrank))
for i in range(resume_step+rank*perrank, resume_step+(rank+1)*perrank):
	z = redshift[i]
	print(' redshift %.3f\tprocessor: %d/%d' %(z, rank, nprocs))


i = resume_step+rank*perrank
while i < resume_step+(rank+1)*perrank:
	z = redshift[i]
	print(' redshift %.3f\tprocessor: %d/%d' %(z, rank, nprocs))
	i += 1
print('...done.')
"""
