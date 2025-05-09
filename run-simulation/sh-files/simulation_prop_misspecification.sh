#!/bin/bash
#SBATCH --partition=std
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --output=/beegfs/u/*******/sim_Neurips/03_output_prop_misspecification/slurm-%j.out 
#SBATCH --error=/beegfs/u/*******/sim_Neurips/03_errors_prop_misspecification/slurm-%j.err
set -e  # Stop operation on first error
#set -u  # Treat undefined variables as an error
set -x  # Print command lines as they are executed

# Initialize environment
source /sw/batch/init.sh

# Load necessary modules
module unload env
module load env/gcc-13.2.0_openmpi-4.1.6

# Activate your Conda environment
source /usw/*******/anaconda3/bin/activate prop_calib_neurips

# Setup the environment variables for OpenMP
#export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}  # Ensure this matches the cpus-per-task
#export OMP_PROC_BIND=spread
#export OMP_PLACES=cores
#export OMP_SCHEDULE=static
#export OMP_DISPLAY_ENV=verbose
export MPLCONFIGDIR=/beegfs/u/*******/sim_Neurips/matplotlib

# Create required directories
mkdir -p /beegfs/u/*******/sim_Neurips/03_errors_prop_misspecification
mkdir -p /beegfs/u/*******/sim_Neurips/03_output_prop_misspecification
mkdir -p /beegfs/u/*******/sim_Neurips/03_results_prop_misspecification/ranks_$SLURM_JOB_ID
#mkdir -p /beegfs/u/*******/sim_Neurips/results/ranks

# Change to the directory where the script is located
cd /beegfs/u/*******/sim_Neurips/

# Print current directory and echo error log path
echo "Current directory: $(pwd)"
echo "Error log will be saved to: /beegfs/u/*******/sim_Neurips/03_errors_prop_misspecification/error_report_$SLURM_JOB_ID.log"

# Run the Python script with mpirun and redirect error output to an error log
mpirun python sim_prop_misspecification.py 2> /beegfs/u/*******/sim_Neurips/03_errors_prop_misspecification/error_report_$SLURM_JOB_ID.log

# Combine output files after the run
if [ $SLURM_PROCID -eq 0 ]; then
    python - << EOF
import os
import pandas as pd

output_dir = f'03_results_prop_misspecification/ranks_{os.getenv("SLURM_JOB_ID")}'
combined_df = pd.concat([pd.read_pickle(f"{output_dir}/{file}") for file in os.listdir(output_dir) if file.endswith('.pkl')], ignore_index=True)
combined_df.to_pickle(f'03_results_prop_misspecification/results_prop_misspecification_{os.getenv("SLURM_JOB_ID")}.pkl')
EOF
fi
# Print CPUs allowed
echo "CPUs allowed: $(grep Cpus_allowed_list /proc/self/status | awk '{print $2}')" >> /beegfs/u/*******/sim_Neurips/03_output_prop_misspecification/slurm-$SLURM_JOB_ID.out

exit