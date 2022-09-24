from re import T
import numpy as np, os, sys
from datetime import datetime

rank = int(os.environ['SLURM_ARRAY_TASK_ID'])
nprocs = int(os.environ['SLURM_ARRAY_TASK_COUNT'])

thread = int(sys.argv[1])
nthreads = int(os.environ['SLURM_NTASKS'])

redshift = np.arange(0, 20)
loop_start, loop_end = 0, redshift.size

perrank = (loop_end-loop_start)//nprocs
perthread = perrank//nthreads
assert perrank*perthread == redshift.size
assert perthread*nthreads == perrank

i_start = int(loop_start+rank*perrank+thread*perthread)
i_end = int(loop_start+rank*perrank+(thread+1)*perthread)

path = '/scratch/snx3000/mibianco/'
with open('%srank%d.txt' %(path, rank), 'w') as f:
	f.wirte('(%d, %d) [%s]' %(rank, thread, ', '.join([str(i) for i in range(i_start, i_end)])))
