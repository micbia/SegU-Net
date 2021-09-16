#!/bin/sh
#SBATCH --job-name=py21cm
#SBATCH --ntasks 1
#SBATCH --account=dp004
#SBATCH --partition=cosma6
#SBATCH --time=2:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user=mb756@sussex.ac.uk

#SBATCH --array=0-4
#SBATCH --output=../logs/data21cm%A.%j.out
#SBATCH --error=../logs/data21cm%A.%j.err

# This adds various useful things to your PATH
module load utils
module unload python/2.7.15

# for py21cmFAST
module load intel_comp/2020-update2
module load intel_mpi/2020-update2
module load fftw/3.3.8

# python env
module load pythonconda3/2020-02

echo $SLURM_ARRAY_TASK_MAX
#DIR='./new_start'
DIR='/cosma6/data/dp004/dc-bian1/inputs/dataLC_128_180621/'

if [ -d "$DIR" ]; then
    echo " Resume 21cmFast data..."
else
    echo " Create new 21cmFast data..."
    mkdir $DIR
    mkdir $DIR/data
    mkdir $DIR/images
    mkdir $DIR/parameters
fi

#python create_data_21cmfast.py $DIR
python create_LC.py $DIR
