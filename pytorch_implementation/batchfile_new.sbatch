#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=test2
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --mem=50GB
###SBATCH --partition=gpu
#SBATCH --gres=gpu:p1080:1

module load python/intel/2.7.12 pytorch/0.2.0_1 protobuf/intel/3.1.0 spyder/3.1.4
module load torchvision/0.1.8 
#cd pytorch-CycleGAN-and-pix2pix
python train.py --dataroot ./datasets/horse2zebra --batchSize 1 --name horse2zebra_cyclegan --model cycle_gan --display_id 0 --no_dropout --max_dataset_size 667 
python test.py --dataroot ./datasets/horse2zebra --name horse2zebra_cyclegan --model cycle_gan --phase test --no_dropout --which_epoch