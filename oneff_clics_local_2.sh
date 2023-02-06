#!/bin/bash

conda activate nodemb

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

