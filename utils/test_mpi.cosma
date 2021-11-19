#!/bin/sh
#SBATCH --job-name=test
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4

#SBATCH --exclusive
#SBATCH --account=dp004
#SBATCH --partition=cosma6
#SBATCH --time=00:00:40

#SBATCH --output=../logs/test-out.%j
#SBATCH --error=../logs/test-err.%j
##SBATCH --mail-type=ALL
##SBATCH --mail-user=mb756@sussex.ac.uk

module purge

# This adds various useful things to your PATH
module load utils
module unload python/2.7.15

# for py21cmFAST
module load intel_comp/2020-update2
module load intel_mpi/2020-update2
module load fftw/3.3.8

# python env
module load pythonconda3/2020-02

mpiexec -n ${SLURM_NTASKS} python test_mpi.py
