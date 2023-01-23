import json
import os

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nodevectors import ProNE, Glove, GGVec


def get_language_embeddings(dataset, colex_metric, lang_metric, model_name, inputfile,
                            outputfolder="data/language_embeddings"):
    # languages are represented by summing the colex representations in one language

    outputfolder = os.path.join(outputfolder, f"{colex_metric}+{lang_metric}")
    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)

    def get_colex_dim(colex_metric):
        if colex_metric == "concat":
            return 200

        elif colex_metric == "add":
            return 100

    def get_vec_string(lang_emb):
        vec = list(lang_emb)
        vec_str = ['%.9f' % val for val in vec]
        vec_str = " ".join(vec_str)
        return vec_str

    def creating_language_vector(connector):
        pass

    colex_embeddings_file = f"data/colex_embeddings/{colex_metric}/{dataset}_{model_name}_embeddings"
    colex_vectors = KeyedVectors.load_word2vec_format(colex_embeddings_file, binary=False)

    df = pd.read_csv(inputfile)

    writer = open(os.path.join(outputfolder, f"{dataset}_{model_name}_embeddings"), "w")

    if dataset == "clics":
        lang_len = len(df["ISO639P3code"].dropna().value_counts())
        colex_dim = get_colex_dim(colex_metric)
        print(
            f"languages {lang_len} -> dataset {dataset}-> {model_name} -> colex_metric {colex_metric} -> lang metric {lang_metric}")
        writer.write(f"{lang_len} {colex_dim}\n")

        for lang, group in df.groupby(by="ISO639P3code"):
            lang_vec = np.zeros(colex_dim)
            writer.write(str(lang) + ' ')

            for colex_id in group["Colex_ID"]:
                lang_vec += colex_vectors[str(colex_id)]

            # lang_i = sum(colex_j)
            vec_str = get_vec_string(lang_vec)

            writer.write(vec_str + '\n')

    elif dataset == "wn" or dataset == "wn_concept":
        lang_len = len(df["LANG3"].dropna().value_counts())

        colex_dim = get_colex_dim(colex_metric)
        writer.write(f"{lang_len} {colex_dim}\n")
        print(
            f"languages {lang_len} -> dataset {dataset} -> {model_name} -> colex_metric {colex_metric} lang metric {lang_metric}")

        for lang, group in df.groupby(by="LANG3"):
            lang_vec = np.zeros(colex_dim)
            writer.write(str(lang) + ' ')

            for colex_id in group["COLEX_ID"]:
                lang_vec += colex_vectors[str(colex_id)]

            # lang_i = sum(colex_j)
            vec_str = get_vec_string(lang_vec)
            writer.write(vec_str + '\n')

    writer.close()


def main(dataset, colex_metric, lang_metric, inputfile):
    # for model_name in ["node2vec", "prone", "glove", "ggvc"]:
    for model_name in ["node2vec", "prone", "ggvc", "glove"]:
        print(model_name)
        get_language_embeddings(dataset, colex_metric, lang_metric, model_name, inputfile)


if __name__ == '__main__':
    import plac

    plac.call(main)

#  python src/clustering/language_embeddings.py clics ~/Documents/experiments/LangSim/data/linguistic_features/colex_clics3_all.csv
