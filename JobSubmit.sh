#!/bin/bash
#SBATCH -p qTRDGPU
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH -t 24:00:00
#SBATCH -e logs/error%A.err
#SBATCH -o logs/out%A.out
#SBATCH -A trends517s113
#SBATCH --oversubscribe
#SBATCH -J L-DiVA-MIMIC-001
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nshaik3@student.gsu.edu

# a small delay at the start often helps
sleep 10s

# activate conda
source /home/users/nshaik3/miniconda3/bin/activate

# CD into your directory
cd /home/users/nshaik3/Desktop/ICML-2026/DiA-Long/
# run the batch script
python3 main.py mimic-cxr

# a delay at the end is also good practice
sleep 10s