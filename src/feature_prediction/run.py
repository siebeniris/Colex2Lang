import os
import json

import pandas as pd

import torch
import torch.optim as optim
from gensim.models import KeyedVectors

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


def load_lang_list(langs):
    input_folder = "data/TypPred/preprocessed/"

    if langs == "wals+uriel+clics+wn":
        filepath = os.path.join(input_folder, "wals_uriel_clics_wn.json")
    elif langs == "wals+uriel+clics":
        filepath = os.path.join(input_folder, "wals_uriel_clics.json")
    elif langs == "wals+uriel+wn":
        filepath = os.path.join(input_folder, "wals_uriel_wn.json")
    elif langs == "clics":
        filepath = os.path.join(input_folder, "clics.json")
    else:
        filepath = None

    if filepath is not None:
        with open(filepath) as f:
            langs_list = json.load(f)

        return langs_list
    else:
        return None


def run(device="cpu", output_folder="output/models", model_name="oneff", epochs=100,
        langs=None,
        dataset=None,
        node_embeddings=None, metric=None,
        feature_area=None,
        ):
    # node_embeddings : node2vec
    # dataset: clics
    # metric: add+avg

    print(f"Using {device} device")
    print(f"dataset {dataset}, node_embeddings: {node_embeddings}, metric: {metric}")

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

    if feature_area is not None:
        feature_file = "data/TypPred/preprocessed/wals_features.yaml"
        with open(feature_file) as f:
            features = yaml.load(f, Loader=Loader)
        features = list(features[feature_area].values())

    else:
        features = list(feature_dict.keys())
    print(features)

    if langs is not None:
        output_folder = os.path.join(output_folder, langs)
        langs_list = load_lang_list(langs)
    else:
        langs_list = None

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for feature_name in features:
        print("*" * 40)
        label_dim = len(feature_dict[feature_name]["values"])
        feature_id = feature_dict[feature_name]["id"]
        print(f"feature {feature_name} has {label_dim} labels")

        if node_embeddings is not None:
            outputfile_name = f"{model_name}_{dataset}_{node_embeddings}_{metric}_{feature_id}.json"
            outputfile = os.path.join(output_folder, outputfile_name)

        elif dataset is not None:
            outputfile_name = f"{model_name}_{dataset}_{feature_id}.json"
            outputfile = os.path.join(output_folder, outputfile_name)
        else:
            outputfile_name = f"{model_name}_{feature_id}.json"
            outputfile = os.path.join(output_folder, outputfile_name)

        print(f"outputfile path {outputfile}")

        if not os.path.exists(outputfile):

            train_data, langs_train = load_dataset(train_df, feature_name)
            dev_data, langs_dev = load_dataset(dev_df, feature_name)
            test_data, langs_test = load_dataset(test_df, feature_name)

            langs_train = set(train_data["wals_code"].tolist())
            if langs_list is not None:
                intersection = set(langs_train) & set(langs_list)

                if len(list(intersection)) > 0:
                    print(f"intersection {intersection}")

                    print(f"train lang {len(langs_train)} dev lang {len(langs_dev)} test lang {len(langs_test)}")

                    # output folder for the models.
                    if model_name == "oneff":

                        # check if there are overlapping languages in train data
                        if metric in ["add+avg", "add+max", "add+sum"]:
                            input_dim = 100
                            hidden_dim = 75
                        elif metric in ["concat+avg", "concat+max", "concat+sum"]:
                            input_dim = 200
                            hidden_dim = 150
                        else:
                            if dataset == "uriel":
                                input_dim = 512
                                hidden_dim = 300
                            else:
                                input_dim = 100
                                hidden_dim = 75

                        print(f"{dataset} -> input dim {input_dim} hidden dim {hidden_dim}")
                        model = OneFF(device=device, num_langs=num_langs, input_dim=input_dim, hidden_dim=hidden_dim,
                                      label_dim=label_dim, dropout=0.5)

                        model = model.to(device)

                        optimizer = optim.Adam(model.parameters(), lr=0.001)

                        if dataset is not None:
                            # uriel, clics, wn, wn_concept
                            if dataset == "uriel":
                                dataset_path = "data/URIEL/learned_embeddings"
                                uriel_language_vectors = KeyedVectors.load_word2vec_format(dataset_path, binary=False)

                                model_name_ = f"{model_name}_{dataset}"

                                train_model(model, model_name_, optimizer, train_data, dev_data, test_data,
                                            feature_name,
                                            feature_id,
                                            lang_dict,
                                            langs_list,
                                            output_folder,
                                            max_epochs=epochs, language_vectors=uriel_language_vectors)
                            else:
                                if node_embeddings is not None:
                                    dataset_path = os.path.join("data/language_embeddings", metric,
                                                                f"{dataset}_{node_embeddings}_embeddings")
                                    language_vectors = KeyedVectors.load_word2vec_format(dataset_path, binary=False)

                                    model_name_ = f"{model_name}_{dataset}_{node_embeddings}_{metric}"

                                    train_model(model, model_name_, optimizer, train_data, dev_data, test_data,
                                                feature_name,
                                                feature_id,
                                                lang_dict,
                                                langs_list,
                                                output_folder,
                                                max_epochs=epochs, language_vectors=language_vectors)

                        else:
                            train_model(model, model_name, optimizer, train_data, dev_data, test_data, feature_name,
                                        feature_id,
                                        lang_dict,
                                        langs_list,
                                        output_folder,
                                        max_epochs=epochs, language_vectors=None)
        else:
            print(f"{outputfile} exists!")
            print("*" * 50)


if __name__ == '__main__':
    import plac

    plac.call(run)
