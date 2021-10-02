#!/bin/sh
#SBATCH --job-name=py21cm
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 10

#SBATCH --exclusive
#SBATCH --account=dp004
#SBATCH --partition=cosma6
#SBATCH --time=24:00:00

#SBATCH --output=../logs/data21cm.%j.out
#SBATCH --error=../logs/data21cm.%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=mb756@sussex.ac.uk

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

#DIR='/cosma6/data/dp004/dc-bian1/inputs/data3D_128_valid_190821/'
#mpiexec -n ${SLURM_NTASKS} python create_data_21cmfast_mpi.py $DIR

DIR='/cosma6/data/dp004/dc-bian1/inputs/dataLC_128_valid_060921/'
mpiexec -n ${SLURM_NTASKS} python create_lightcone_21cmfast_mpi.py $DIR