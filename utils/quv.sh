#!/bin/sh
#SBATCH --job-name=uv-coverage
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=26
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=1

#SBATCH --account=Pra17_4382
#SBATCH --partition=gll_usr_prod
#SBATCH --time=24:00:00

#SBATCH --output=msg/uv-out.%j
#SBATCH --error=msg/uv-err.%j
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mb756@sussex.ac.uk

module load intel intelmpi
module load profile/base autoload python/3.6.4
module load profile/base autoload fftw
module load profile/base autoload gsl

mpiexec -n ${SLURM_NPROCS} python create_uvcoverage_mpi.py
