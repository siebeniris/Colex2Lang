#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=glove
#SBATCH --output=%j.glove.out
#SBATCH --time=12:00:00
#SBATCH --mem=16G

source $HOME/.bashrc
conda activate graphEmb

cd $HOME/ColexGraph


env=$1

conda activate "$env"

python -m src.feature_prediction.run cpu output/models/rerun oneff 100 clics clics glove concat+sum
python -m src.feature_prediction.run cpu output/models/rerun oneff 100 clics clics glove concat+avg
python -m src.feature_prediction.run cpu output/models/rerun oneff 100 clics clics glove concat+max
python -m src.feature_prediction.run cpu output/models/rerun oneff 100 clics clics glove add+sum
python -m src.feature_prediction.run cpu output/models/rerun oneff 100 clics clics glove add+avg
python -m src.feature_prediction.run cpu output/models/rerun oneff 100 clics clics glove add+max
