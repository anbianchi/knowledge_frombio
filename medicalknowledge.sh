#!/bin/bash
#$ -S /bin/bash
#$ -pe mpi 40
#$ -cwd
#$ -o /NFSHOME/abianchi/log/std_$JOB_ID.out
#$ -e /NFSHOME/abianchi/log/err_$JOB_ID.out
#$ -l h=compute-0-0.local

source ~/.bashrc
conda activate medicalmining

python main.py --dataset "processed_patients_59051patients.csv"

