#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=graphEmb_bn
#SBATCH --output=node_embeddings.txt
#SBATCH --time=8:00:00
#SBATCH --mem=128G

source $HOME/.bashrc
conda activate graphEmb

cd $HOME/ColexGraph

env=$1
dataset=$2
model=$3


conda activate "$env"

echo "Train the graph embedding models..."
python -m src.graphs.models data/edgelists/edgelists_"$dataset".csv "$dataset" "$model"
