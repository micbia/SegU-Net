#!/bin/sh
#SBATCH --job-name=run
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1

#SBATCH --exclusive
#SBATCH --account=dp004
#SBATCH --partition=cosma6
#SBATCH --time=01:00:00

#SBATCH --output=../logs/run.%j.out
#SBATCH --error=../logs/run.%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=mb756@sussex.ac.uk

module purge

# This adds various useful things to your PATH
module load utils
module unload python/2.7.15

python content_tar.py