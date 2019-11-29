#!/bin/bash
#SBATCH --job-name="dump_sparsified_graphs"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=sched_mit_sloan_batch
#SBATCH --time=4-00:00:00
#SBATCH --array=11-12

#SBATCH -o /home/sanzeed/seeding_queries/dump_sparsified_graphs.out
#SBATCH -e /home/sanzeed/seeding_queries/dump_sparsified_graphs.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sanzeed@mit.edu

module load python/3.6.3
module load sloan/python/modules/3.6

python3 /home/sanzeed/seeding_queries/dump_sparsified_graphs.py ${SLURM_ARRAY_TASK_ID}

