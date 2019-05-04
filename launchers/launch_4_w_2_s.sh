#!/bin/bash

#SBATCH --job-name=4_workers_2_steps
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/ppo208/lab4/output/4_workers_2_steps.out
#SBATCH --error=/home/ppo208/lab4/output/4_workers_2_steps.error
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

use_cuda='false'
workers=0
opt='adam'
num_proc=5 # number of processors, must be equal to #SBATCH --nodes
out_file=log-$num_proc-$use_cuda-$workers-$opt

module load openmpi/intel/2.0.3
module load cuda/9.2.88

mpirun -np $num_proc /home/ppo208/anaconda3/bin/python  lab4_singlenode.py --steps 2 --data_path '/scratch/gd66/spring2019/lab4/kaggleamazon/' --disable_cuda  --workers $workers
