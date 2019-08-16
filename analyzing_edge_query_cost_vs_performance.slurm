#!/bin/bash
#SBATCH --job-name="k1010e_lc"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=sched_mit_sloan_interactive
#SBATCH --time=4-00:00:00


#SBATCH -o /home/rahimian/seeding_queries_k20_edge_lowcost/analyzing_edge_query_performance_k_1010_lowcost.out
#SBATCH -e /home/rahimian/seeding_queries_k20_edge_lowcost/analyzing_edge_query_performance_k_1010_lowcost.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rahimian@mit.edu

module load python/3.6.3
module load sloan/python/modules/3.6

python3 /home/rahimian/seeding_queries_k20_edge_lowcost/analyzing_edge_query_cost_vs_performance.py
