#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=wn_concept_01
#SBATCH --output=%j.wn_concept_01.out
#SBATCH --time=20:00:00
#SBATCH --mem=8G

source $HOME/.bashrc
conda activate graphEmb

cd $HOME/ColexGraph



env=$1
device=$2


conda activate "$env"



python -m src.feature_prediction.run "$device" output/models oneff 100 "wals+uriel+wn" wn_concept prone concat+sum
