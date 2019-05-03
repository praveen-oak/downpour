#!/bin/bash

#SBATCH --job-name=lab4i
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=lab4.out
#SBATCH --error=lab4.error
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00

use_cuda='false'
workers=8
opt='adam'
num_proc=1 # number of processors, must be equal to #SBATCH --nodes
out_file=log-$num_proc-$use_cuda-$workers-$opt

module load openmpi/intel/2.0.3
module load cuda/9.2.88

mpirun -np 2 --oversubscribe /home/ppo208/anaconda3/bin/python test_mpi.py