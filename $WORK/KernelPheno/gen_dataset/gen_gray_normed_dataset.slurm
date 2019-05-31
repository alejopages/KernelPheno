#!/bin/sh
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --job-name=gen_dataset
#SBATCH --error=$WORK/KernelPheno/gen_dataset/%J.err
#SBATCH --output=$WORK/KernelPheno/gen_dataset/%J.out
#SBATCH --partition=jclarke

echo We\'re in the node

module load anaconda


source activate KernelPheno
python -m KernelPheno.kernelpheno.main generate-dataset $WORK/KernelPheno/gray_normed /work/schnablelab/apages/KernelPheno/t1 /home/schnablelab/apages/pysrc/KernelPheno/data/zooniverse_docs/reports/report_04-21-19\:14\:58\:08.clean.csv
