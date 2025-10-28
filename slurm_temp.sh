#!/bin/bash
#SBATCH --job-name=random
#SBATCH --time=01:00:00
#SBATCH --array=1-500
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --partition=kipac

export OMP_NUM_THREADS=12

#python3 ~/lensing_template/plot_maps.py~/lensing_template/get_cib_tracer.py $SLURM_ARRAY_TASK_ID
~/lensing_template/get_cib_tracer.py $SLURM_ARRAY_TASK_ID
~/lensing_template/combine_tracers.py $SLURM_ARRAY_TASK_ID
