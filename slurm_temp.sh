#!/bin/bash
#SBATCH --job-name=random
#SBATCH --time=5:00:00
#SBATCH --array=0
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --partition=kipac
##SBATCH --dependency=afterok:16628571_10

export OMP_NUM_THREADS=12

#python3 /home/users/yukanaka/healqest/pipeline/spt3g_20192020/yuka_misc_scripts/pack_likelihood_products_yuuki.py
#python3 ~/lensing_template/check_cinv_alms.py
#~/lensing_template/get_lenz_cib_tracer.py $SLURM_ARRAY_TASK_ID
#~/lensing_template/get_pr3_cib_pr4_kappa_cib_tracer.py $SLURM_ARRAY_TASK_ID
#python3 ~/lensing_template/check_lenz_map_mask.py
#python3 ~/lensing_template/get_pr3_cib_pr4_kappa_spectra.py
#python3 ~/lensing_template/get_agora545_cib_spectra.py
#python3 ~/lensing_template/get_fiona_cib_kappa_spectra_pr3.py
#python3 ~/lensing_template/get_lenz_cib_kappa_spectra.py
#python3 ~/lensing_template/check_mocks.py
#python3 ~/lensing_template/plot_maps.py
#python3 ~/lensing_template/temp.py
#~/lensing_template/compute_combination_weights.py
#~/lensing_template/check_tune_combine_tracers_dummy_tracers.py
#~/lensing_template/get_dummy_tracer.py $SLURM_ARRAY_TASK_ID
#python3 ~/lensing_template/validate_profile_hardening.py
#python3 ~/lensing_template/check_sims_btemplates.py
#~/lensing_template/combine_tracers_old.py $SLURM_ARRAY_TASK_ID
#~/lensing_template/combine_tracers.py $SLURM_ARRAY_TASK_ID
#~/lensing_template/tune_combine_tracers.py bt_gmv3500_combined_agora545_cib.yaml 5001 1
#~/lensing_template/tune_combine_tracers.py 1 $SLURM_ARRAY_TASK_ID --use_10_sims
#~/lensing_template/tune_combine_tracers.py 1 2
#~/lensing_template/tune_combine_tracers.py 1 3
#~/lensing_template/tune_combine_tracers.py 1 4
#~/lensing_template/tune_combine_tracers.py 1 5
#~/lensing_template/tune_combine_tracers.py 1 6
#~/lensing_template/tune_combine_tracers.py 1 7
#~/lensing_template/tune_combine_tracers.py 1 8
#~/lensing_template/tune_combine_tracers.py 1 9
#~/lensing_template/tune_combine_tracers.py 1 10
#~/lensing_template/tune_combine_tracers_no_normalization.py bt_gmv3500_combined_pr3_cib_pr4_kappa.yaml 0 1
#~/lensing_template/tune_combine_tracers_no_normalization.py bt_gmv3500_combined_pr3_cib_pr4_kappa.yaml 0 1 --use_10_sims
#~/lensing_template/tune_combine_tracers_no_normalization.py bt_gmv3500_combined_agora545_cib.yaml 5001 1
~/lensing_template/tune_combine_tracers_no_normalization_simautocross.py bt_gmv3500_combined_agora545_cib.yaml 5001 1
