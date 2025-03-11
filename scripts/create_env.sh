#!/bin/bash

#SBATCH -J create_env
#SBATCH -o /data1/choderaj/kaminowb/polaris_challenge_2025/log_files/create_env.out
#SBATCH -e /data1/choderaj/kaminowb/polaris_challenge_2025/log_files/create_env.out
#SBATCH -D /data1/choderaj/kaminowb/polaris_challenge_2025/
#SBATCH --partition=cpu
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00

/data1/choderaj/kaminowb/micromamba/micromamba create -c conda-forge -n polaris_challenge -f env.yml
echo done
