#!/bin/bash


python python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics

python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics ggvc concat+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics ggvc concat+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics ggvc concat+max
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics ggvc add+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics ggvc add+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics ggvc add+max


python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics glove concat+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics glove concat+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics glove concat+max
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics glove add+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics glove add+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics glove add+max



python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics node2vec concat+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics node2vec concat+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics node2vec concat+max
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics node2vec add+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics node2vec add+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics node2vec add+max




python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics prone concat+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics prone concat+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics prone concat+max
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics prone add+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics prone add+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 clics clics prone add+max

