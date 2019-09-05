#!/bin/bash
#SBATCH --job-name="analyzing_spread_query_performance_given_cost"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=sched_mit_sloan_sinana
#SBATCH --time=4-00:00:00
#SBATCH --array=1-10

#SBATCH -o /home/sanzeed/seeding_queries/analyzing_spread_query_performance_given_cost.out
#SBATCH -e /home/sanzeed/seeding_queries/analyzing_spread_query_performance_given_cost.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sanzeed@mit.edu

module load python/3.6.3
module load sloan/python/modules/3.6

python3 /home/sanzeed/seeding_queries/analyzing_spread_query_performance_given_cost.py ${SLURM_ARRAY_TASK_ID}
