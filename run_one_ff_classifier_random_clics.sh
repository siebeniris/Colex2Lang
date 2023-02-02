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


conda activate "$env"


python -m src.feature_prediction.run cpu output/models/rerun/random oneff 100 clics
python -m src.feature_prediction.run cpu output/models/rerun/random oneff 100 wn
python -m src.feature_prediction.run cpu output/models/rerun/random oneff 100 uriel