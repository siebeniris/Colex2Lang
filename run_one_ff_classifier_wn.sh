#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=wn
#SBATCH --output=%j.wn.out
#SBATCH --time=20:00:00
#SBATCH --mem=8G

source $HOME/.bashrc
conda activate graphEmb

cd $HOME/ColexGraph



env=$1
device=$2


conda activate "$env"

python -m src.feature_prediction.run "$device" output/models oneff 100 "wals+uriel+clics+wn" wn glove concat+avg


