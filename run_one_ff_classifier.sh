#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=oneff
#SBATCH --output=%j.out
#SBATCH --time=8:00:00
#SBATCH --mem=32G

source $HOME/.bashrc
conda activate graphEmb

cd $HOME/ColexGraph



env=$1
device=$2


conda activate "$env"
python -m src.feature_prediction.run "$device"


python -m src.feature_prediction.run "$device" yes output/models oneff 100 ggvc clics add+avg

python -m src.feature_prediction.run "$device" yes output/models oneff 100 ggvc clics add+max

python -m src.feature_prediction.run "$device" yes output/models oneff 100 ggvc clics add+sum

python -m src.feature_prediction.run "$device" yes output/models oneff 100 ggvc clics concat+sum

python -m src.feature_prediction.run "$device" yes output/models oneff 100 ggvc clics concat+sum

python -m src.feature_prediction.run "$device" yes output/models oneff 100 ggvc clics concat+sum


python -m src.feature_prediction.run "$device" yes output/models oneff 100 node2vec clics add+avg

python -m src.feature_prediction.run "$device" yes output/models oneff 100 node2vec clics add+max

python -m src.feature_prediction.run "$device" yes output/models oneff 100 node2vec clics add+sum

python -m src.feature_prediction.run "$device" yes output/models oneff 100 node2vec clics concat+avg

python -m src.feature_prediction.run "$device" yes output/models oneff 100 node2vec clics concat+max

python -m src.feature_prediction.run "$device" yes output/models oneff 100 node2vec clics concat+sum


python -m src.feature_prediction.run "$device" yes output/models oneff 100 prone clics add+avg

python -m src.feature_prediction.run "$device" yes output/models oneff 100 prone clics add+max

python -m src.feature_prediction.run "$device" yes output/models oneff 100 prone clics add+sum

python -m src.feature_prediction.run "$device" yes output/models oneff 100 prone clics concat+avg

python -m src.feature_prediction.run "$device" yes output/models oneff 100 prone clics concat+max

python -m src.feature_prediction.run "$device" yes output/models oneff 100 prone clics concat+sum


python -m src.feature_prediction.run "$device" yes output/models oneff 100 glove clics add+avg

python -m src.feature_prediction.run "$device" yes output/models oneff 100 glove clics add+max

python -m src.feature_prediction.run "$device" yes output/models oneff 100 glove clics add+sum

python -m src.feature_prediction.run "$device" yes output/models oneff 100 glove clics concat+avg

python -m src.feature_prediction.run "$device" yes output/models oneff 100 glove clics concat+max

python -m src.feature_prediction.run "$device" yes output/models oneff 100 glove clics concat+sum
