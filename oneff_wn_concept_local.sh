#!/bin/bash



python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept ggvc concat+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept ggvc concat+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept ggvc concat+max
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept ggvc add+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept ggvc add+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept ggvc add+max


python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept glove concat+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept glove concat+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept glove concat+max
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept glove add+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept glove add+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept glove add+max



python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept node2vec concat+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept node2vec concat+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept node2vec concat+max
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept node2vec add+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept node2vec add+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept node2vec add+max




python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept prone concat+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept prone concat+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept prone concat+max
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept prone add+sum
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept prone add+avg
python -m src.feature_prediction.run_kfold cpu output/ oneff 100 wn wn_concept prone add+max

