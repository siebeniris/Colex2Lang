import json
import os

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nodevectors import ProNE, Glove, GGVec


def get_language_embeddings(dataset, model_name, inputfile, outputfolder="data/language_embeddings", dim=200, lang_col="ISO639P3code"):
    colex_embeddings_file = f"data/colex_embeddings/{dataset}_{model_name}_embeddings"
    colex_vectors = KeyedVectors.load_word2vec_format(colex_embeddings_file, binary=False)

    df = pd.read_csv(inputfile)
    lang_len = len(df[lang_col].dropna().value_counts())

    writer = open(os.path.join(outputfolder, f"{dataset}_{model_name}_embeddings"), "w")
    writer.write(f"{lang_len} 200\n")

    for lang, group in df.groupby(by=lang_col):
        lang_vec = np.zeros(dim)
        writer.write(str(lang) + ' ')
        for colex_id in group["Colex_ID"]:
            lang_vec += colex_vectors[str(colex_id)]

            vec = list(lang_vec)

            vec_str = ['%.9f' % val for val in vec]
            vec_str = " ".join(vec_str)
            writer.write(vec_str + '\n')

    writer.close()


def main(dataset, inputfile):
    for model_name in ["node2vec", "prone", "glove", "ggvc"]:
        print(model_name)
        get_language_embeddings(dataset,model_name, inputfile)


if __name__ == '__main__':
    import plac
    plac.call(main)


