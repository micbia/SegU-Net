#!/bin/sh
#SBATCH --job-name=py21cm
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-core=1

#SBATCH --account=Pra17_4382
#SBATCH --partition=gll_usr_prod
#SBATCH --time=24:00:00

#SBATCH --output=msg/data21cm-out.%j
#SBATCH --error=msg/data21cm-err.%j
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mb756@sussex.ac.uk

module load intel intelmpi
module load profile/base autoload python/3.6.4
module load profile/base autoload fftw
module load profile/base autoload gsl

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores  # pinning threads correctly

#DIR='./new_start'
DIR='/gpfs/scratch/userexternal/mbianco0/data3D_128_030920/'

if [ -d "$DIR" ]; then
    echo " Resume 21cmFast data..."
else
    echo " Create new 21cmFast data..."
    mkdir $DIR
    mkdir $DIR/data
    mkdir $DIR/images
    mkdir $DIR/parameters
fi

srun python create_data_21cmfast_mpi.py $DIR