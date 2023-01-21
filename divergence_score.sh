#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=divergence
#SBATCH --output=%j.out
#SBATCH --time=8:00:00
#SBATCH --mem=128G

source $HOME/.bashrc
conda activate graphEmb

cd $HOME/ColexGraph/src/graphs/Comparing_Graph_Embeddings

python get_divergence_scores.py divergence.csv ~/ColexGraph/data/eval_nodeEmb/
