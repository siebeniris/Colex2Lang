#!/bin/bash
#
#SBATCH --partition=prioritized
#SBATCH --job-name=create_language_embeddings
#SBATCH --output=%j.out
#SBATCH --time=8:00:00
#SBATCH --mem=64G


echo "creating language embeddings "

## wn

python src/create_embeddings/language_embeddings.py wn add sum data/colex_data/wn_all.csv
python src/create_embeddings/language_embeddings.py wn add max data/colex_data/wn_all.csv
python src/create_embeddings/language_embeddings.py wn add avg data/colex_data/wn_all.csv


python src/create_embeddings/language_embeddings.py wn concat sum data/colex_data/wn_all.csv
python src/create_embeddings/language_embeddings.py wn concat max data/colex_data/wn_all.csv
python src/create_embeddings/language_embeddings.py wn concat avg data/colex_data/wn_all.csv

### wn_concept


python src/create_embeddings/language_embeddings.py wn_concept add sum data/colex_data/wn_concept.csv
python src/create_embeddings/language_embeddings.py wn_concept add max data/colex_data/wn_concept.csv
python src/create_embeddings/language_embeddings.py wn_concept add avg data/colex_data/wn_concept.csv


python src/create_embeddings/language_embeddings.py wn_concept concat sum data/colex_data/wn_concept.csv
python src/create_embeddings/language_embeddings.py wn_concept concat max data/colex_data/wn_concept.csv
python src/create_embeddings/language_embeddings.py wn_concept concat avg data/colex_data/wn_concept.csv


### clics

python src/create_embeddings/language_embeddings.py clics add sum data/colex_data/colex_clics3_all.csv
python src/create_embeddings/language_embeddings.py clics add max data/colex_data/colex_clics3_all.csv
python src/create_embeddings/language_embeddings.py clics add avg data/colex_data/colex_clics3_all.csv


python src/create_embeddings/language_embeddings.py clics concat sum data/colex_data/colex_clics3_all.csv
python src/create_embeddings/language_embeddings.py clics concat max data/colex_data/colex_clics3_all.csv
python src/create_embeddings/language_embeddings.py clics concat avg data/colex_data/colex_clics3_all.csv

