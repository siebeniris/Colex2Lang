import os
import json

import pandas as pd
import numpy as np
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
#
# wals_file = "data/wals_by_language.tsv"
# data_dir = "data/TypPred/preprocessed"
#
# with open(os.path.join(data_dir, "wals_features.yaml")) as f:
#     features_dict = yaml.load(f, Loader)
#
# df_wals = pd.read_csv(wals_file, sep="\t")
#
# with open(os.path.join(data_dir, "wals_uriel_clics_wn.json")) as f:
#     wals_uriel_clics_wn_langs = json.load(f)
#
# df_inter = df_wals[df_wals["WALS-ID"].isin(wals_uriel_clics_wn_langs)]
#
# # get the splits of langauges
# df_train = pd.read_csv(os.path.join(data_dir, "train.csv"), index_col=0)
# df_dev = pd.read_csv(os.path.join(data_dir, "dev.csv"), index_col=0)
# df_test = pd.read_csv(os.path.join(data_dir, "test.csv"), index_col=0)
#
# train_langs = df_train.index.dropna().tolist()
# dev_langs = df_dev.index.dropna().tolist()
# test_langs = df_test.index.dropna().tolist()
#
# print("test", len(test_langs), "train", len(train_langs), "dev", len(dev_langs))
#

def deduplicate_wals(outputfile,wals_file="data/wals_by_language.tsv"):
    # there are duplicates per iso code, drop the rows which have fewer feature values.
    df_wals = pd.read_csv(wals_file, sep="\t")
    drop_cols = ["WALS-ID", "Glottocode", "Name", "Latitude", "Longitude"]
    df_wals = df_wals.drop(columns=drop_cols)

    # count the values each row.
    df_wals["NON_NULL"] = df_wals.notnull().sum(axis=1)

    # descending order the dataframe, and drop duplicates by the iso codes.
    df_wals.columns = [x.replace(" ", "_") for x in df_wals.columns]
    df_wals = df_wals.sort_values(by="NON_NULL", ascending=False).drop_duplicates(subset=["ISO"],
                                                                                  keep="first")
    df_wals = df_wals.dropna(subset=["ISO"])

    df_wals.to_csv(outputfile, index=False)


def get_test_data(wals_file="data/TypPred/wals_by_languages.csv", langs_json="data/TypPred/wals+uriel+clics+wn_langs.json"):
    # get test data for wals+uriel+clics+wn
    df_wals = pd.read_csv(wals_file)
    with open(langs_json) as f:
        langs = json.load(f)
    df_inter = df_wals[df_wals["ISO"].isin(langs)]
    df_inter.to_csv("data/TypPred/test_wals+uriel+clics+wn.csv", index=False)


def create_datasets( data_dir="data/TypPred"):
    df_wals = pd.read_csv(os.path.join(data_dir, "wals_by_languages.csv"))

    with open(os.path.join(data_dir, "wals+uriel+clics+wn_langs.json")) as f:
        wals_uriel_clics_wn_langs = json.load(f)









if __name__ == '__main__':
    import plac
    plac.call(deduplicate_wals)
