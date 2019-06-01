#!/bin/bash

#SBATCH --job-name=downpour
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --output=downpour.out
#SBATCH --error=downpour.error
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00

use_cuda='false'
workers=0
opt='adam'
num_proc=3 # number of processors, must be equal to #SBATCH --nodes
out_file=log-$num_proc-$use_cuda-$workers-$opt

module load openmpi/intel/2.0.3
module load cuda/9.2.88

mpirun -np $num_proc /home/ppo208/anaconda3/bin/python  downpour.py --data_path '/scratch/gd66/spring2019/lab4/kaggleamazon/' --disable_cuda  --workers $workers | tee $out_file
