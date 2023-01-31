#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --job-name=uriel
#SBATCH --output=%j.uriel.gpu.out
#SBATCH --time=20:00:00
#SBATCH --mem=16G
#SBATCH --nodelist=nv-ai-01.srv.aau.dk

source $HOME/.bashrc

cd $HOME/ColexGraph


env=$1

conda activate "$env"

#python -m src.feature_prediction.run "$device" output/models oneff 100 "wals+uriel+clics+wn" uriel

#python -m src.feature_prediction.run "$device" output/models oneff 100 "wals+uriel+clics" uriel

python -m src.feature_prediction.run cuda output/models oneff 100 "wals+uriel+wn" uriel
