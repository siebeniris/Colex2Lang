#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=oneff
#SBATCH --output=%j.out
#SBATCH --time=8:00:00
#SBATCH --mem=16G

source $HOME/.bashrc
conda activate graphEmb

cd $HOME/ColexGraph



env=$1
device=$2


conda activate "$env"
python -m src.feature_prediction.run "$device" output/models/clics oneff 100