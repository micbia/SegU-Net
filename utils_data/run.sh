#!/bin/sh
#SBATCH --job-name=run
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --account=sk09
#SBATCH --time=02:00:00
#SBATCH --output=../logs/run%j.out
#SBATCH --error=../logs/run%j.err
#SBATCH -C gpu
##SBATCH --mem 16G
##SBATCH -c 8

#SBATCH --mail-type=ALL
#SBATCH --mail-user=michele.bianco@epfl.ch

module purge
module load daint-gpu
module load gcc/9.3.0
module load cudatoolkit/10.2.89_3.28-2.1__g52c0314

# export conda on shell
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate py21cmenv
python lc_adapt.py
conda deactivate
