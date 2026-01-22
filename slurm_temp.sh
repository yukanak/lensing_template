#!/bin/bash
#SBATCH --job-name=random
#SBATCH --time=20:00:00
#SBATCH --array=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --partition=kipac
##SBATCH --dependency=afterok:12170112_499

export OMP_NUM_THREADS=12

#python3 /home/users/yukanaka/healqest/pipeline/spt3g_20192020/yuka_misc_scripts/pack_likelihood_products_agoraGfgs.py
#~/lensing_template/compute_combination_weights.py $SLURM_ARRAY_TASK_ID
#python3 ~/lensing_template/check_cinv_alms.py
#python3 ~/lensing_template/plot_maps.py
#~/lensing_template/get_lenz_cib_tracer.py $SLURM_ARRAY_TASK_ID
#~/lensing_template/get_pr3_cib_pr4_kappa_cib_tracer.py $SLURM_ARRAY_TASK_ID
#~/lensing_template/combine_tracers.py $SLURM_ARRAY_TASK_ID
~/lensing_template/tune_combine_tracers.py $SLURM_ARRAY_TASK_ID
#python3 ~/lensing_template/check_lenz_map_mask.py
#python3 ~/lensing_template/get_pr3_cib_pr4_kappa_spectra.py
#python3 ~/lensing_template/get_fiona_cib_kappa_spectra.py
#python3 ~/lensing_template/get_fiona_cib_kappa_spectra_pr3.py
#python3 ~/lensing_template/get_lenz_cib_kappa_spectra.py
#python3 ~/lensing_template/validate_profile_hardening.py
#python3 ~/lensing_template/temp.py
#python3 ~/lensing_template/check_mocks.py
#python3 ~/lensing_template/check_sims_btemplates.py
