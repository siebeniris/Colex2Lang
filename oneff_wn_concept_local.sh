#!/bin/bash


python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn

python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn ggvc concat+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn ggvc concat+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn ggvc concat+max
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn ggvc add+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn ggvc add+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn ggvc add+max


python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn glove concat+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn glove concat+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn glove concat+max
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn glove add+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn glove add+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn glove add+max



python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn node2vec concat+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn node2vec concat+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn node2vec concat+max
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn node2vec add+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn node2vec add+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn node2vec add+max




python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn prone concat+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn prone concat+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn prone concat+max
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn prone add+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn prone add+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn prone add+max

