#!/bin/bash
#SBATCH --job-name="EQ k=4"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=sched_mit_sloan_sinana
#SBATCH --time=4-00:00:00
#SBATCH --array=1-4


#SBATCH -o /home/sanzeed/seeding_queries/EQ_k4.out
#SBATCH -e /home/sanzeed/seeding_queries/EQ_k4.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sanzeed@mit.edu

module load python/3.6.3
module load sloan/python/modules/3.6

python3 /home/sanzeed/seeding_queries/analyzing_edge_query_performance_given_cost.py ${SLURM_ARRAY_TASK_ID}
