#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=generator-classifiers
#SBATCH --mail-type=END
#SBATCH --mail-user=ls4411@nyu.edu
#SBATCH --output=classify_\%j.out

module purge
module load pytorch/intel/20170226
module load torchvision/0.1.7

cd /scratch/ls4411/Research/GAN/QualityAssessor

DIR_NAME="cifar10"
FILE_NAME="all-dataset"

python classify.py --epochs=200 --train-real=True --dataset=cifar10 --imageSize=32 --cuda --netG=../TrainedNetworks/$DIR_NAME/$FILE_NAME &> /scratch/ls4411/Research/GAN/TrainedNetworks/$DIR_NAME/$FILE_NAME-classify-output.txt

exit 0

## Empty line
