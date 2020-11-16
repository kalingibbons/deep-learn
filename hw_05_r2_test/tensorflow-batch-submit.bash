#!/bin/bash

#Account and Email Information
#SBATCH -A <r2-user-name>
#SBATCH --mail-type=end
#SBATCH --mail-user=<your-email-id>

#SBATCH -J PYTHON          # job name
#SBATCH -o outputs/results.o%j # output and error file name (%j expands to jobID)
#SBATCH -e outputs/errors.e%j
#SBATCH -n 1               # Run one process
#SBATCH --cpus-per-task=28 # allow job to multithread across all cores
#SBATCH -p gpuq            # queue (partition) -- defq, ipowerq, eduq, gpuq.
#SBATCH -t 1-00:00:00      # run time (d-hh:mm:ss)
ulimit -v unlimited
ulimit -s unlimited
ulimit -u 1000

module load cuda10.0/toolkit/10.0.130 # loading cuda libraries/drivers 
module load python/intel/3.7          # loading python environment

python3 train_cnn_script.py
