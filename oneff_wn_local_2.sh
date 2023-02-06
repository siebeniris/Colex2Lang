#!/bin/bash

# both wn and wn_concept

python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept node2vec add+max
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept node2vec add+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept node2vec concat+avg

