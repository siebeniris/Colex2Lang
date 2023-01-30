#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=prone
#SBATCH --output=%j.prone.out
#SBATCH --time=8:00:00
#SBATCH --mem=32G

source $HOME/.bashrc
conda activate graphEmb

cd $HOME/ColexGraph



env=$1
device=$2


conda activate "$env"

python -m src.feature_prediction.run "$device" output/models oneff 100 "clics" clics prone add+avg

python -m src.feature_prediction.run "$device" output/models oneff 100 "clics" clics prone add+max

python -m src.feature_prediction.run "$device" output/models oneff 100 "clics" clics prone add+sum

python -m src.feature_prediction.run "$device" output/models oneff 100 "clics" clics prone concat+avg

python -m src.feature_prediction.run "$device" output/models oneff 100 "clics" clics prone concat+max

python -m src.feature_prediction.run "$device" output/models oneff 100 "clics" clics prone concat+sum
