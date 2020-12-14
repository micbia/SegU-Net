#!/bin/sh
#SBATCH --job-name=test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=1

#SBATCH --account=Pra17_4382
#SBATCH --partition=gll_usr_prod
#SBATCH --time=24:00:00

#SBATCH --output=msg/test-out.%j
#SBATCH --error=msg/test-err.%j
##SBATCH --mail-type=ALL
##SBATCH --mail-user=mb756@sussex.ac.uk

module load intel intelmpi
module load profile/base autoload python/3.6.4
module load profile/base autoload fftw
module load profile/base autoload gsl

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores	 # pinning threads correctly

srun python test.py
