import json
import os
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from termcolor import cprint


def load_lexicon_features(file="data/TypPred/preprocessed/wals_features.yaml"):
    with open(file, "r") as f:
        lexicon_features = yaml.load(f, Loader)

    return list(lexicon_features["lexicon"].values())


# create datasets for training 190 feature classifiers
# for each classifier, there are 1k languages with their values

def restructure_file(filepath, feature_maps, feature_cols_=None, test_gold=False):
    records = defaultdict(dict)
    feature_cols = []

    cols = ["wals_code", "name", "latitude", "longitude", "genus", "family", "countrycodes"]
    cprint(f"loading {filepath}", "green")
    answers = 0
    with open(filepath) as file:
        counter = 0
        for line in file.readlines():

            if not line.startswith("wals_code"):
                l = line.replace("\n", "").split("\t", 7)

                t1, t2, t3, t4, t5, t6, t7, features = l
                records[counter]["wals_code"] = t1
                records[counter]["name"] = t2
                records[counter]["latitude"] = t3
                records[counter]["longitude"] = t4
                records[counter]["genus"] = t5
                records[counter]["family"] = t6
                records[counter]["countrycodes"] = t7

                for t in features.split("|"):
                    # make a feature a column
                    try:
                        t1, t2 = t.split("=", 1)
                        if t1 not in feature_cols:
                            feature_cols.append(t1)

                        records[counter][t1] = t2
                        if not test_gold:
                            feature_maps[t1].append(t2)

                        answers += 1
                    except Exception as msg:
                        # records[counter][t1] = np.nan
                        cprint(f"{msg}", "red")

                counter += 1
    cols = cols + feature_cols
    print(f"records column length {len(cols)}")
    if feature_cols_:
        df_records = pd.DataFrame.from_dict(records, orient="index", columns=feature_cols_)
    else:
        df_records = pd.DataFrame.from_dict(records, orient="index", columns=cols)

    print(df_records.head(5))
    print(f"number of annos: {answers}")
    return df_records


def get_blinded_cells(test_x):
    """
    Get the (Row, col) tuples for "?" in blinded test data
    :param test_x: df.
    :return: matrix of (row, col)
    """
    test_cells = test_x[test_x == "?"].to_numpy()
    print(test_cells)
    test_matrix = list()
    for idrow, row in enumerate(test_cells):
        for idcol, col in enumerate(row):
            if not pd.isnull(col):
                test_matrix.append((idrow, idcol))
    return test_matrix


def preprocess_dataset(df, feature_dict):
    # change the cells to idx.
    drop_cols = ["wals_code", "name", "latitude", "longitude", "genus", "family", "countrycodes"]
    cols = df.columns.tolist()
    cols_features = [col for col in cols if col not in drop_cols]

    for feature in cols_features:
        value_dict = feature_dict[feature]
        df[feature] = df[feature].apply(lambda x: value_dict.get(x, np.nan))

    return df


def create_train_dev_test(input_folder="data/ST2020/data", output_folder="data/TypPred/preprocessed"):
    # feature_name: [value]
    feature_maps = defaultdict(list)
    # populate feature_maps, which should include all the features from train/dev/test

    train_dataset = restructure_file(os.path.join(input_folder, "train.csv"), feature_maps)
    cprint(f"length of the feature maps {len(feature_maps)}", "red")

    dev_blinded = pd.read_csv("data/TypPred/dev_x.csv", index_col=0)
    dev_gold = pd.read_csv("data/TypPred/dev_y.csv", index_col=0)
    dev_blinded = dev_blinded.sort_values(by='wals_code', ascending=True)
    dev_gold = dev_gold.sort_values(by='wals_code', ascending=True)

    # get the feature_maps for dev_gold
    drop_cols = ["wals_code", "name", "latitude", "longitude", "genus", "family", "countrycodes"]
    cols = dev_gold.columns.tolist()
    cols_features = [col for col in cols if col not in drop_cols]
    dev_dataset = dev_blinded.copy()
    for feature in cols_features:
        feature_maps[feature] += dev_gold[feature].dropna().tolist()
        dev_dataset[feature] = dev_dataset[feature].replace("?", pd.NA) # change the "?" with np.nan

    # dev_dataset is the dev data without "?"
    dev_dataset = dev_dataset[cols]
    dev_blinded = dev_blinded[cols]  # for later get the dev gold dataset.
    assert dev_dataset.wals_code.tolist() == dev_gold.wals_code.tolist()
    assert dev_dataset.columns.tolist() == dev_gold.columns.tolist()

    cprint(f"length of the feature maps {len(feature_maps)}", "red")

    # get feature dictionaries for statistics and feature value ids.
    feature_dict = defaultdict(dict)
    feature_counter = defaultdict(dict)
    for feature, feature_values in feature_maps.items():
        feature_counter[feature] = Counter(feature_values)
        feature_values = list(set([x for x in feature_values if x != "?"]))
        feature_dict[feature] = {v: idx for idx, v in enumerate(feature_values)}

    with open(os.path.join(output_folder, "feature_maps.json"), "w") as f:
        json.dump(feature_dict, f)

    with open(os.path.join(output_folder, "feature_counter.json"), "w") as f:
        json.dump(feature_counter, f)

    # concatenate train, dev datasets into one.
    df_concat = pd.concat([train_dataset, dev_dataset], axis=0)
    cprint(f"concatenated dataframe length  {len(df_concat)} column length {len(df_concat.columns)}", "red")
    df_concat = preprocess_dataset(df_concat, feature_dict)
    df_concat.to_csv(os.path.join(output_folder, "train.csv"), index=False)

    ####################################################################
    dev_matrix = get_blinded_cells(dev_blinded)
    dev_gold_df = dev_gold.copy()

    assert dev_blinded.wals_code.tolist() == dev_gold.wals_code.tolist()
    assert dev_blinded.columns.tolist() == dev_gold.columns.tolist()

    dev_gold_cols = dev_gold_df.columns.tolist()
    drop_cols = ["wals_code", "name", "latitude", "longitude", "genus", "family", "countrycodes"]
    feature_names = [x for x in dev_gold_cols if x not in drop_cols]
    for feature in feature_names:
        dev_gold_df[feature] = dev_gold_df[feature].apply(lambda x: pd.NA)

    for idrow, idcol in dev_matrix:
        value = dev_gold.iat[idrow, idcol]
        dev_gold_df.iat[idrow, idcol] = value

    dev_gold_df = preprocess_dataset(dev_gold_df, feature_dict)
    dev_gold_df.to_csv(os.path.join(output_folder, "dev.csv"), index=False)

    ####################################################################

    test_dataset = restructure_file(os.path.join(input_folder, "test_blinded.csv"), feature_maps)
    cprint(f"length of the feature maps {len(feature_maps)}", "red")
    test_dataset = test_dataset.sort_values(by='wals_code', ascending=True)

    # the test gold columns should be the same as the test blinded data
    test_gold_cols = test_dataset.columns.tolist()
    # a list of (row, col) ids for test data.
    test_matrix = get_blinded_cells(test_dataset)

    test_gold = restructure_file(os.path.join(input_folder, "test_gold.csv"), defaultdict(dict), test_gold_cols, True)
    test_gold = test_gold.sort_values(by='wals_code', ascending=True)

    cprint(f"length test gold {len(test_gold)}, test blinded {len(test_dataset)}", "red")

    test_gold_df = test_gold.copy()

    assert test_dataset["wals_code"].tolist() == test_gold_df["wals_code"].tolist()
    assert test_dataset.columns.tolist() == test_gold_df.columns.tolist()

    drop_cols = ["wals_code", "name", "latitude", "longitude", "genus", "family", "countrycodes"]
    feature_names = [x for x in test_gold_cols if x not in drop_cols]
    for feature in feature_names:
        test_gold_df[feature] = test_gold_df[feature].apply(lambda x: pd.NA)

    for idrow, idcol in test_matrix:
        value = test_gold.iat[idrow, idcol]
        test_gold_df.iat[idrow, idcol] = value

    test_gold_df = preprocess_dataset(test_gold_df, feature_dict)
    test_gold_df.to_csv(os.path.join(output_folder, "test.csv"), index=False)


def create_datasets(input_folder="data/ST2020/data", output_folder="data/TypPred/preprocessed"):
    # feature_name: [value]
    feature_maps = defaultdict(list)
    # populate feature_maps

    train_dataset = restructure_file(os.path.join(input_folder, "train.csv"), feature_maps)
    cprint(f"length of the feature maps {len(feature_maps)}", "red")

    dev_dataset = restructure_file(os.path.join(input_folder, "dev.csv"), feature_maps)
    cprint(f"length of the feature maps {len(feature_maps)}", "red")
    test_dataset = restructure_file(os.path.join(input_folder, "test_blinded.csv"), feature_maps)
    cprint(f"length of the feature maps {len(feature_maps)}", "red")

    # get feature dictionaries for statistics and feature value ids.
    feature_dict = defaultdict(dict)
    feature_counter = defaultdict(dict)
    for feature, feature_values in feature_maps.items():
        feature_counter[feature] = Counter(feature_values)
        feature_values = list(set([x for x in feature_values if x != "?"]))
        feature_dict[feature] = {v: idx for idx, v in enumerate(feature_values)}

    with open(os.path.join(output_folder, "feature_maps.json"), "w") as f:
        json.dump(feature_dict, f)

    with open(os.path.join(output_folder, "feature_counter.json"), "w") as f:
        json.dump(feature_counter, f)

    # concatenate train, dev, test datasets into one.
    test_dataset = test_dataset.sort_values(by='wals_code', ascending=True)
    df_concat = pd.concat([train_dataset, dev_dataset, test_dataset], axis=0)
    cprint(f"concatenated dataframe length  {len(df_concat)} column length {len(df_concat.columns)}", "red")
    df_concat = preprocess_dataset(df_concat, feature_dict)
    df_concat.to_csv(os.path.join(output_folder, "train_data.csv"), index=False)

    ####################################################################
    # the test gold columns should be the same as the test blinded data
    test_gold_cols = test_dataset.columns.tolist()
    # a list of (row, col) ids for test data.
    test_matrix = get_blinded_cells(test_dataset)

    test_gold = restructure_file(os.path.join(input_folder, "test_gold.csv"), defaultdict(dict), test_gold_cols, True)
    test_gold = test_gold.sort_values(by='wals_code', ascending=True)

    cprint(f"length test gold {len(test_gold)}, test blinded {len(test_dataset)}", "red")

    test_gold_df = test_gold.copy()

    assert test_dataset["wals_code"].tolist() == test_gold_df["wals_code"].tolist()
    assert test_dataset.columns.tolist() == test_gold_df.columns.tolist()

    drop_cols = ["wals_code", "name", "latitude", "longitude", "genus", "family", "countrycodes"]
    feature_names = [x for x in test_gold_cols if x not in drop_cols]
    for feature in feature_names:
        test_gold_df[feature] = test_gold_df[feature].apply(lambda x: pd.NA)

    for idrow, idcol in test_matrix:
        value = test_gold.iat[idrow, idcol]
        test_gold_df.iat[idrow, idcol] = value

    test_gold_df = preprocess_dataset(test_gold_df, feature_dict)
    test_gold_df.to_csv(os.path.join(output_folder, "test.csv"), index=False)


if __name__ == '__main__':
    # create_datasets()
    create_train_dev_test()
