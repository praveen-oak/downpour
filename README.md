# downpour
Implementation of the synchronous distributed machine learning algorithm downpour. The project takes a neural net inspired by inception net and then uses distributed pytorch and OPEN MPI packages to implement data parallelism across multiple GPU cores to achieve near perfect linear scalability.


The research paper on the downpour distributed algorithm is here:
https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf

Requirements:
The project has the following dependencies
Python 3.6,
cuda/9.2.88,
openmpi/intel/2.0.3.

In addition pytorch needs to be compiled with the cuda and openmpi libraries. Please refer online resources and guides on how to
accomplish this.
Once you are done, please run the following command in the shell prompt
python -c 'import torch; print(torch.__version__)' 
If your compilation was successful, then the command should print out the following on the console
1.0.0a0+4c11dee

To test if MPI is setup you can run the following
mpirun -np 4python -c 'import torch.distributed as dist; dist.init_process_group(backend="mpi"),print("hello", dist.get_rank())'
It should print the following on the console
hello 1 
hello 3 
hello 2 
hello 0 

If you are running on a standalone node(or personal computer) with multiple GPU cores, no additional softwares are required.
If you are running it on a cluster, please use a cluster batch management tool. I have used my university's HPC cluster which 
has slurm workload manager built in.

The project has two files
1. downpour.py
This file contains the implementation of the algorithm. The file has a main function which accepts input parameters.
The file needs to be invoked from the mpi environment. Details of how to load cuda, mpi modules prior to running the project can be 
found in the launch_multinode.s shell script.

2. launch_multinode.s
This is a shell script which can be used to launch a distributed job. It contains command line arguments to be passed to the python 
file as well as arguments to the slurm workload manager in case you are using a HPC cluster to run the project. 
If you are not using such a cluster, you can ignore all the commands before module load openmpi/intel/2.0.3 command
