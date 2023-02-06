import os
import json

import pandas as pd

import torch
import torch.optim as optim
from gensim.models import KeyedVectors

import warnings

warnings.filterwarnings("ignore")
import yaml
from sklearn.model_selection import train_test_split, KFold

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from src.feature_prediction.baseline import train_model_splits

torch.manual_seed(42)


def load_lang_list(langs):
    input_folder = "data/TypPred/"

    if langs == "uriel":
        filepath = os.path.join(input_folder, "wals+uriel_langs.json")
    elif langs == "clics":
        filepath = os.path.join(input_folder, "wals+clics_langs.json")
    elif langs == "wn":
        filepath = os.path.join(input_folder, "wals+wn_langs.json")
    else:
        filepath = None

    if filepath is not None:
        with open(filepath) as f:
            langs_list = json.load(f)

        return langs_list
    else:
        return None


def run(output_folder="output/models", model_name="baseline", langs=None
        ):
    # baseline totally depends on the distribution of training dataset.

    # feature_id: {feature, values}
    with open("data/TypPred/feature_dict.json") as f:
        feature_dict = json.load(f)

    if langs is not None:
        print(f"loading the langs {langs}")
        output_folder = os.path.join(output_folder, langs)
        langs_list = load_lang_list(langs)
    else:
        langs_list = None

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    lexicon_features = ["129A", "130A", "130B", "131A", "132A", "133A", "134A", "135A", "136A", "136B", "137B", "138A"]
    for feature_id in feature_dict:
        print("*" * 40)
        label_dim = len(feature_dict[feature_id]["values"])
        feature = feature_dict[feature_id]["feature"]
        print(f"feature {feature} has {label_dim} labels")

        train_file = f"data/TypPred/datasets/features/train_{feature_id}.csv"
        dev_file = f"data/TypPred/datasets/features/dev_{feature_id}.csv"
        test_file = f"data/TypPred/datasets/features/test_{feature_id}.csv"

        if os.path.exists(train_file):
            train_data = pd.read_csv(train_file)
            dev_data = pd.read_csv(dev_file)
            test_data = pd.read_csv(test_file)

            train_data[feature] = train_data[feature].astype("int")
            dev_data[feature] = dev_data[feature].astype("int")
            test_data[feature] = test_data[feature].astype("int")

            train_data = train_data[train_data["ISO"].isin(langs_list)]
            dev_data = dev_data[dev_data["ISO"].isin(langs_list)]

            if len(train_data) > 0 and len(dev_data) > 0:


                outputfile_name = f"{model_name}_{feature_id}.json"
                outputfile = os.path.join(output_folder, outputfile_name)

                print(f"outputfile path {outputfile}")

                if not os.path.exists(outputfile):

                    langs_train = set(train_data["ISO"].tolist())
                    langs_dev = set(dev_data["ISO"].tolist())
                    langs_test = set(test_data["ISO"].tolist())

                    all_langs = list(set(langs_train) | set(langs_dev) | set(langs_test))
                    num_langs = len(all_langs)
                    lang_dict = {lang: idx for idx, lang in enumerate(all_langs)}
                    print(f"languages in total {num_langs}")

                    df_train_test = pd.concat([train_data, test_data], axis=0)
                    train_data, test_data = train_test_split(df_train_test, test_size=0.1, shuffle=False)

                    df_train_dev = pd.concat([train_data, dev_data], axis=0)
                    if len(df_train_dev) > 10:
                        kf = KFold(n_splits=10)
                        train_dev_splits = kf.split(df_train_dev)
                    else:
                        sample_len = len(df_train_dev)
                        kf = KFold(n_splits=sample_len)
                        train_dev_splits = kf.split(df_train_dev)

                    # output folder for the models.
                    if model_name == "baseline":

                        if langs_list is not None:

                            print(
                                f"train lang {len(langs_train)} dev lang {len(langs_dev)} test lang {len(langs_test)}")


                            train_model_splits( model_name, train_dev_splits, df_train_dev,
                                                       test_data,
                                                       feature,
                                                       feature_id,
                                                       lang_dict,
                                                       langs_list,
                                                       output_folder)

                else:
                    print(f"{outputfile} exists!")
                    print("*" * 50)
        else:
            print(f"train file {train_file} doesn't exist!")


if __name__ == '__main__':
    import plac

    plac.call(run)
