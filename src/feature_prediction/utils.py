import os
import json
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


def lang_clics_wn():
    lang_embeddings_dir = "data/language_embeddings"
    with open(os.path.join(lang_embeddings_dir, "clics_langs.json")) as f:
        clics_langs = json.load(f)
    with open(os.path.join(lang_embeddings_dir, "wn_concept_langs.json")) as f:
        wn_concept_langs = json.load(f)
    with open(os.path.join(lang_embeddings_dir, "wn_langs.json")) as f:
        wn_langs = json.load(f)

    clics_langs = set(clics_langs)
    wn_concept_langs = set(wn_concept_langs)
    wn_langs = set(wn_langs)

    common = list(clics_langs.intersection(wn_langs, wn_concept_langs))
    print(len(common), common[:10])
    with open(os.path.join(lang_embeddings_dir, "common_langs.json"), "w") as f:
        json.dump(common, f)


def get_langs_inter_typpred(file, sep, dataset):
    lang_embeddings_dir = "data/language_embeddings"
    print(f"file -> {file}")
    with open(os.path.join(lang_embeddings_dir, f"{dataset}_langs.json")) as f:
        langs = json.load(f)
    df = pd.read_csv(file, sep=sep)
    print(f"length: {len(df)}")
    df_filter = df[df["wals_code"].isin(langs)]
    print(f'length: {len(df_filter)}')




def feature_maps_info():
    with open("data/TypPred/preprocessed/feature_maps.json") as f:
        feature_maps = json.load(f)
    df_parameter = pd.read_csv("data/cldf-datasets-wals-878ea47/raw/parameter.csv")
    feature2id = dict(zip([name.replace(" ","_") for name in df_parameter["name"]], df_parameter["id"]))
    print(feature2id)

    new_feature_maps = defaultdict(dict)
    for feature, value_dict in feature_maps.items():
        new_feature_maps[feature] = {
            "id": feature2id[feature],
            "values": value_dict
        }
    with open("data/TypPred/preprocessed/feature_dict.json", "w") as f:
        json.dump(new_feature_maps, f)


if __name__ == '__main__':
    import plac

    # plac.call(lang_clics_wn)
    # plac.call(get_langs_inter_typpred)
    plac.call(feature_maps_info)
