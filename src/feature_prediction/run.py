import os
import json

import pandas as pd
import numpy as np

import torch
from torch import nn
import torch.optim as optim
from gensim.models import KeyedVectors

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings

warnings.filterwarnings("ignore")
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from src.feature_prediction.training_utils import train_model, load_dataset
from src.feature_prediction.models import OneFF

torch.manual_seed(42)


def run(device="cpu", lexicon_only=None, output_folder="output/models", model_name="oneff", epochs=100,
        node_embeddings=None, dataset=None, metric=None):
    # node_embeddings : node2vec
    # dataset: clics
    # metric: add+avg

    print(f"Using {device} device")
    print(f"dataset {dataset} node_embeddings {node_embeddings} metric {metric}")

    with open("data/TypPred/preprocessed/feature_dict.json") as f:
        feature_dict = json.load(f)

    train_file = os.path.join("data/TypPred/preprocessed", "train.csv")
    dev_file = os.path.join("data/TypPred/preprocessed", "dev.csv")
    test_file = os.path.join("data/TypPred/preprocessed", "test.csv")

    train_df = pd.read_csv(train_file)
    dev_df = pd.read_csv(dev_file)
    test_df = pd.read_csv(test_file)

    langs_dev_all = dev_df["wals_code"].tolist()
    langs_train_all = train_df["wals_code"].tolist()
    langs_test_all = test_df["wals_code"].tolist()

    all_langs = list(set(langs_dev_all) | set(langs_train_all) | set(langs_test_all))
    num_langs = len(all_langs)
    lang_dict = {lang: idx for idx, lang in enumerate(all_langs)}
    print(f"languages in total {num_langs}")

    if lexicon_only is not None:
        feature_file = "data/TypPred/preprocessed/wals_features.yaml"
        with open(feature_file) as f:
            features = yaml.load(f, Loader=Loader)
            features = list(features["lexicon"].values())
    else:
        features = list(feature_dict.keys())
    print(features)

    for feature_name in features:
        print("*" * 40)
        label_dim = len(feature_dict[feature_name]["values"])
        feature_id = feature_dict[feature_name]["id"]
        print(f"feature {feature_name} has {label_dim} labels")

        train_data, langs_train = load_dataset(train_df, feature_name)
        dev_data, langs_dev = load_dataset(dev_df, feature_name)
        test_data, langs_test = load_dataset(test_df, feature_name)

        print(f"train lang {len(langs_train)} dev lang {len(langs_dev)} test lang {len(langs_test)}")

        # output folder for the models.

        if model_name == "oneff":
            if metric in ["add+avg", "add+max", "add+sum"]:
                input_dim = 100
                hidden_dim = 200
            else:
                input_dim = 200
                hidden_dim = 400

            model = OneFF(device=device, num_langs=num_langs, input_dim=input_dim, hidden_dim=hidden_dim, label_dim=label_dim, dropout=0.5)

            model = model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=0.001)

            if node_embeddings is not None:
                if dataset is not None:
                    dataset_path = os.path.join("data/language_embeddings", metric,
                                                f"{dataset}_{node_embeddings}_embeddings")
                    language_vectors = KeyedVectors.load_word2vec_format(dataset_path, binary=False)

                    model_name_ = f"{model_name}_{dataset}_{node_embeddings}_{metric}"

                    train_model(model, model_name_, optimizer, train_data, dev_data, test_data, feature_name,
                                feature_id,
                                lang_dict,
                                output_folder,
                                max_epochs=epochs, language_vectors=language_vectors)
            else:
                train_model(model, model_name, optimizer, train_data, dev_data, test_data, feature_name,
                            feature_id,
                            lang_dict,
                            output_folder,
                            max_epochs=epochs, language_vectors=None)


if __name__ == '__main__':
    import plac

    plac.call(run)
