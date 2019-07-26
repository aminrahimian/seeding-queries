#!/bin/bash
#SBATCH --job-name="k10edge"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=sched_mit_sloan_interactive
#SBATCH --time=4-00:00:00


#SBATCH -o /home/rahimian/seeding_queries_k10_edge/analyzing_edge_query_performance_k_10.out
#SBATCH -e /home/rahimian/seeding_queries_k10_edge/analyzing_edge_query_performance_k_10.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rahimian@mit.edu

module load python/3.6.3
module load sloan/python/modules/3.6

python3 /home/rahimian/seeding_queries_k10_edge/analyzing_edge_query_cost_vs_performance.py
