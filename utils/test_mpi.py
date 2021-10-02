from mpi4py import MPI
import numpy as np, time
from datetime import datetime

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

redshift = np.arange(7.001, 7.021, 0.001)

resume_step = 0
perrank = redshift.size//nprocs

"""
# Broadcasting a variable
if(rank == 0):
	nr_procs_done = 0
else:
    nr_procs_done = None

nr_procs_done = comm.bcast(nr_procs_done, root=0)

"""
comm.Barrier()
if(rank == 0):
	time.sleep(30)
	print(' rank %d done:\t%s' %(rank, datetime.now().strftime('%H:%M:%S')))
elif(rank == 1): 
	time.sleep(22)
	print(' rank %d done:\t%s' %(rank, datetime.now().strftime('%H:%M:%S')))
elif(rank == 2):
	time.sleep(18)
	print(' rank %d done:\t%s' %(rank, datetime.now().strftime('%H:%M:%S')))
elif(rank == 3):
	time.sleep(10)
	print(' rank %d done:\t%s' %(rank, datetime.now().strftime('%H:%M:%S')))

nr_procs_done = 1
nr_procs_done = comm.gather(nr_procs_done, root=0)
if(rank == 0):
	print(' gather done:\t%s\t%s' %(datetime.now().strftime('%H:%M:%S'), str(nr_procs_done)))

comm.Barrier()




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
