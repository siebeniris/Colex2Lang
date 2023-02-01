import os
import json

import pandas as pd
import numpy as np
import yaml
from collections import defaultdict
from sklearn.model_selection import train_test_split

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def deduplicate_wals(outputfile, wals_file="data/wals_by_language.tsv"):
    # there are duplicates per iso code, drop the rows which have fewer feature values.
    df_wals = pd.read_csv(wals_file, sep="\t")
    drop_cols = ["WALS-ID", "Glottocode", "Name", "Latitude", "Longitude", "Family"]
    df_wals = df_wals.drop(columns=drop_cols)

    # count the values each row.
    df_wals["NON_NULL"] = df_wals.notnull().sum(axis=1)

    # descending order the dataframe, and drop duplicates by the iso codes.
    df_wals.columns = [x.replace(" ", "_") for x in df_wals.columns]
    df_wals = df_wals.sort_values(by="NON_NULL", ascending=False).drop_duplicates(subset=["ISO"],
                                                                                  keep="first")
    df_wals = df_wals.dropna(subset=["ISO"])

    df_wals.to_csv(outputfile, index=False)


def get_test_data(wals_file="data/TypPred/wals_by_languages.csv",
                  langs_json="data/TypPred/wals+uriel+clics+wn_langs.json"):
    # get test data for wals+uriel+clics+wn
    df_wals = pd.read_csv(wals_file)
    with open(langs_json) as f:
        langs = json.load(f)
    df_inter = df_wals[df_wals["ISO"].isin(langs)]
    df_inter.to_csv("data/TypPred/test_wals+uriel+clics+wn.csv", index=False)


def create_feature_maps(data_dir="data/TypPred"):
    feature_maps = defaultdict(dict)
    feature_counter = defaultdict(dict)

    df_wals = pd.read_csv(os.path.join(data_dir, "wals_by_languages.csv"), index_col=0)

    for feature in df_wals.columns:
        value_counter = df_wals[feature].value_counts().to_dict()
        print(value_counter)
        value_keys = list(value_counter.keys())
        feature_counter[feature] = value_counter
        feature_maps[feature] = {v: k for k, v in enumerate(value_keys)}

    with open(os.path.join(data_dir, "feature_maps.json"), "w") as f:
        json.dump(feature_maps, f)

    with open(os.path.join(data_dir, "feature_counter.json"), "w") as f:
        json.dump(feature_counter, f)


def replace_values(df, feature_maps):
    for feature in df.columns:
        df[feature] = df[feature].replace(feature_maps[feature])
    return df


def create_datasets(data_dir="data/TypPred"):
    with open(os.path.join(data_dir, "feature_maps.json")) as f:
        feature_maps = json.load(f)

    df_train_dev = pd.read_csv(os.path.join(data_dir, "train_dev.csv"), index_col=0)
    df_test = pd.read_csv(os.path.join(data_dir, "test_wals+uriel+clics+wn.csv"), index_col=0)

    df_train_dev_ = replace_values(df_train_dev, feature_maps).drop(columns="NON_NULL")
    df_test_ = replace_values(df_test, feature_maps).drop(columns="NON_NULL")

    df_train_dev_.to_csv(os.path.join(data_dir, "datasets", "train_dev.csv"))
    df_test_.to_csv(os.path.join(data_dir, "datasets", "test.csv"))


def split_data_by_features(train_dev_file=""):
    print(f"{train_dev_file}")
    df = pd.read_csv(train_dev_file)

    df_test = pd.read_csv("data/TypPred/datasets/test.csv")


    features = df.drop(columns=["ISO"]).columns

    output_dir = os.path.join("data/TypPred/datasets/", "features")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open("data/TypPred/wals_features.yaml") as f:
        features_dict_ = yaml.load(f, Loader=Loader)

    feature2id = {}
    id2feature = {}
    for k, v in features_dict_.items():
        for feature_id, feature in v.items():
            feature2id[feature] = feature_id
            id2feature[feature_id] = feature

    # df_lang = df[df["ISO"].isin(langs_list)]
    df_lang = df
    samples = []
    for feature in features:
        if feature !="NON_NULL":
            feature_id = feature2id[feature]
            print(feature_id)
            df_lang_feature = df_lang[["ISO", feature]].dropna()
            df_lang_feature_test = df_test[["ISO", feature]].dropna()
            try:
                train, dev = train_test_split(df_lang_feature, test_size=0.1, random_state=42, shuffle=True,
                                              stratify=feature)

                if len(train) > 0 and len(dev) > 0 and len(df_lang_feature_test):
                    train.to_csv(os.path.join(output_dir, f"train_{feature_id}.csv"), index=False)
                    dev.to_csv(os.path.join(output_dir, f"dev_{feature_id}.csv"), index=False)
                    df_lang_feature_test.to_csv(os.path.join(output_dir, f"test_{feature_id}.csv"), index=False)
            except Exception as msg:
                print(msg)
                train, dev = train_test_split(df_lang_feature, test_size=0.1, random_state=42, shuffle=True)
                print(train)
                print(dev)

            if len(train) > 0 and len(dev) > 0 and len(df_lang_feature_test):
                train.to_csv(os.path.join(output_dir, f"train_{feature_id}.csv"), index=False)
                dev.to_csv(os.path.join(output_dir, f"dev_{feature_id}.csv"), index=False)
                df_lang_feature_test.to_csv(os.path.join(output_dir, f"test_{feature_id}.csv"), index=False)


def create_feature_dict():
    with open("data/TypPred/feature_maps.json") as f:
        feature_maps = json.load(f)

    with open("data/TypPred/wals_features.yaml") as f:
        features_dict_ = yaml.load(f, Loader=Loader)

    feature2id = {}
    id2feature = {}
    for k, v in features_dict_.items():
        for feature_id, feature in v.items():
            feature2id[feature] = feature_id
            id2feature[feature_id] = feature

    feature_dict = defaultdict(dict)
    for feature, values in feature_maps.items():
        if feature != "NON_NULL":
            feature_dict[feature2id[feature]] = {
                "feature": feature,
                "values": values
            }
    with open("data/TypPred/feature_dict.json", "w") as f:
        json.dump(feature_dict, f)


if __name__ == '__main__':
    import plac

    plac.call(split_data_by_features)
