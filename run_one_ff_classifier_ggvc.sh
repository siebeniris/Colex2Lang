#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=ggvc
#SBATCH --output=%j.ggvc.out
#SBATCH --time=16:00:00
#SBATCH --mem=32G

source $HOME/.bashrc
conda activate graphEmb

cd $HOME/ColexGraph



env=$1
device=$2


conda activate "$env"


python -m src.feature_prediction.run "$device" output/models oneff 100 "clics" clics ggvc add+avg

python -m src.feature_prediction.run "$device" output/models oneff 100 "clics" clics ggvc add+max

python -m src.feature_prediction.run "$device" output/models oneff 100 "clics" clics ggvc add+sum

python -m src.feature_prediction.run "$device" output/models oneff 100 "clics" clics ggvc concat+avg

python -m src.feature_prediction.run "$device" output/models oneff 100 "clics" clics ggvc concat+max

python -m src.feature_prediction.run "$device" output/models oneff 100 "clics" clics ggvc concat+sum
