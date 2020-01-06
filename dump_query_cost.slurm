#!/bin/bash
#SBATCH --job-name="EQC k=4"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=sched_mit_sloan_sinana
#SBATCH --time=4-00:00:00
#SBATCH --array=1-20


#SBATCH -o /home/sanzeed/seeding_queries/EQC_k4.out
#SBATCH -e /home/sanzeed/seeding_queries/EQC_k4.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sanzeed@mit.edu

module load python/3.6.3
module load sloan/python/modules/3.6

python3 /home/sanzeed/seeding_queries/get_edge_query_costs.py ${SLURM_ARRAY_TASK_ID}