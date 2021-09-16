#!/bin/bash -l
#SBATCH --ntasks 1
#SBATCH -J segunet
#SBATCH -o logs/segunet.%J.out
#SBATCH -e logs/segunet.%J.err
#SBATCH -p cosma6 	
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mb756@sussex.ac.uk

# This adds various useful things to your PATH
module load utils
module unload python/2.7.15

# for py21cmFAST
module load intel_comp/2020-update2
module load intel_mpi/2020-update2
module load fftw/3.3.8

# python env
module load pythonconda3/2020-02

python segUNet.py config/net2D_lc_dT.ini
#python segUNet.py config/net2D_lc_full.ini
