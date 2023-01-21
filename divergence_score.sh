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

#python get_divergence_scores.py divergence.csv ~/ColexGraph/data/eval_nodeEmb/

./GED -g /home/cs.aau.dk/ng78zb/ColexGraph/data/eval_nodeEmb/wn_concept/edgelists_wn_concept -c /home/cs.aau.dk/ng78zb/ColexGraph/data/eval_nodeEmb/wn_concept/wn_concept.ecg -e /home/cs.aau.dk/ng78zb/ColexGraph/data/eval_nodeEmb/wn_concept/prone_embeddings
