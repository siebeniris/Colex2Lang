#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=nodeEmb_eval
#SBATCH --output=%j.out
#SBATCH --time=8:00:00
#SBATCH --mem=64G

source $HOME/.bashrc
conda activate graphEmb

cd $HOME/ColexGraph



env=$1
dataset=$2



conda activate "$env"
# step1:
# convert2eval_format
sleep 1
echo "Creating edgelists for evaluation..."

python -m src.graphs.evaluation format "$dataset"


sleep 1
# step 2:
# plac.call(model2embeddings)
echo "Coverting model to embeddings..."
python -m src.graphs.evaluation model2emb "$dataset"
# doesn't work in macos.

sleep 2
echo "Generating ECG for the features..."
python -m src.graphs.evaluation ecg "$dataset"


