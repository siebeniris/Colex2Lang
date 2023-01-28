import json
import os

import plac
from gensim.models import KeyedVectors


def get_langs_from_lang_embeddings(filepath, dataset, outputfolder="data/language_embeddings"):
    vectors = KeyedVectors.load_word2vec_format(filepath)
    langs = vectors.key_to_index.keys()
    langs = list(langs)
    print(f"lang {len(langs)}")
    with open(os.path.join(outputfolder, f"{dataset}_langs.json"), "w") as f:
        json.dump(langs, f)


if __name__ == '__main__':
    plac.call(get_langs_from_lang_embeddings)
