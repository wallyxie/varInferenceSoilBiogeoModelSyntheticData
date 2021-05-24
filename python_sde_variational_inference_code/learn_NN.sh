#!/bin/bash

#SBATCH -A allisons_lab_gpu ## charged account
#SBATCH -p hugemem ## request larger memory per task; highmem, hugemem, and maxmem are options
#SBATCH --mem 8GB ## total amount of memory requested
#SBATCH -N 1 ## number of nodes
#SBATCH --ntasks--er-node=1 ## number of tasks per node
#SBATCH --error=slurm-%J.err ## write errors in slurm-<jobID>.err file
#SBATCH --mail-type=ALL ## send email for all job milestones
#SBATCH --mail-user=xiehw@uci.edu ## use this email address
