#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=uriel
#SBATCH --output=%j.uriel.out
#SBATCH --time=20:00:00
#SBATCH --mem=32G

source $HOME/.bashrc
conda activate graphEmb

cd $HOME/ColexGraph



env=$1
device=$2


conda activate "$env"

python -m src.feature_prediction.run "$device" output/models oneff 100 "wals+uriel+clics+wn" uriel

python -m src.feature_prediction.run "$device" output/models oneff 100 "wals+uriel+clics" uriel

python -m src.feature_prediction.run "$device" output/models oneff 100 "wals+uriel+wn" uriel