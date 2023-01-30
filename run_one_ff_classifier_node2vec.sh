#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=node2vec
#SBATCH --output=%j.node2vec.out
#SBATCH --time=8:00:00
#SBATCH --mem=32G

source $HOME/.bashrc
conda activate graphEmb

cd $HOME/ColexGraph



env=$1
device=$2


conda activate "$env"


python -m src.feature_prediction.run "$device" output/models oneff 100 "clics" clics node2vec add+avg

python -m src.feature_prediction.run "$device" output/models oneff 100 "clics" clics node2vec add+max

python -m src.feature_prediction.run "$device" output/models oneff 100 "clics" clics node2vec add+sum

python -m src.feature_prediction.run "$device" output/models oneff 100 "clics" clics node2vec concat+avg

python -m src.feature_prediction.run "$device" output/models oneff 100 "clics" clics node2vec concat+max

python -m src.feature_prediction.run "$device" output/models oneff 100 "clics" clics node2vec concat+sum
