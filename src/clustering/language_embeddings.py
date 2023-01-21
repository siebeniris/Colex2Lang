import json
import os

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nodevectors import ProNE, Glove, GGVec


def get_language_embeddings(dataset, model_name, inputfile, outputfolder="data/language_embeddings", dim=200):
    # languages are represented by summing the colex representations in one language
    colex_embeddings_file = f"data/colex_embeddings/{dataset}_{model_name}_embeddings"
    colex_vectors = KeyedVectors.load_word2vec_format(colex_embeddings_file, binary=False)

    df = pd.read_csv(inputfile)

    writer = open(os.path.join(outputfolder, f"{dataset}_{model_name}_embeddings"), "w")

    if dataset == "clics":
        lang_len = len(df["ISO639P3code"].dropna().value_counts())
        writer.write(f"{lang_len} 200\n")

        for lang, group in df.groupby(by="ISO639P3code"):
            lang_vec = np.zeros(dim)
            writer.write(str(lang) + ' ')
            for colex_id in group["Colex_ID"]:
                lang_vec += colex_vectors[str(colex_id)]

            # lang_i = sum(colex_j)
            vec = list(lang_vec)

            vec_str = ['%.9f' % val for val in vec]
            vec_str = " ".join(vec_str)
            writer.write(vec_str + '\n')

    elif dataset == "wn":
        lang_len = len(df["LANG3"].dropna().value_counts())
        writer.write(f"{lang_len} 200\n")

        for lang, group in df.groupby(by="LANG3"):
            lang_vec = np.zeros(dim)
            writer.write(str(lang) + ' ')
            for colex_id in group["COLEX_ID"]:
                lang_vec += colex_vectors[str(colex_id)]

            # lang_i = sum(colex_j)
            vec = list(lang_vec)

            vec_str = ['%.9f' % val for val in vec]
            vec_str = " ".join(vec_str)
            writer.write(vec_str + '\n')

    writer.close()


def main(dataset, inputfile):
    # for model_name in ["node2vec", "prone", "glove", "ggvc"]:
    for model_name in ["node2vec", "prone", "ggvc"]:
        print(model_name)
        get_language_embeddings(dataset, model_name, inputfile)


if __name__ == '__main__':
    import plac

    plac.call(main)

#  python src/clustering/language_embeddings.py clics ~/Documents/experiments/LangSim/data/linguistic_features/colex_clics3_all.csv
