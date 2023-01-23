#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=create_colex_embeddings
#SBATCH --output=%j.out
#SBATCH --time=8:00:00
#SBATCH --mem=64G


source $HOME/.bashrc
conda activate graphEmb

cd $HOME/ColexGraph


echo "creating colexification embeddings for datasets using add."
python src/create_embeddings/colex_embeddings.py wn_concept add data/colex_data/wn_concept.csv

python src/create_embeddings/colex_embeddings.py wn add data/colex_data/wn_all.csv

python src/create_embeddings/colex_embeddings.py clics add data/colex_data/colex_clics3_all.csv


echo "creating colexification embeddings for all datasets using concat."

python src/create_embeddings/colex_embeddings.py wn_concept concat data/colex_data/wn_concept.csv

python src/create_embeddings/colex_embeddings.py wn concat data/colex_data/wn_all.csv

python src/create_embeddings/colex_embeddings.py clics concat data/colex_data/colex_clics3_all.csv
