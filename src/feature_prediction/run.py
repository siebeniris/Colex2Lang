import os
import json

import pandas as pd

import torch
import torch.optim as optim
from gensim.models import KeyedVectors

import warnings

warnings.filterwarnings("ignore")
import yaml
from sklearn.model_selection import train_test_split

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from src.feature_prediction.training_utils import train_model
from src.feature_prediction.models import OneFF

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


def run(device="cpu", output_folder="output/models", model_name="oneff", epochs=100,
        langs=None,
        dataset=None,
        node_embeddings=None, metric=None,
        ):
    # node_embeddings : node2vec
    # dataset: clics
    # metric: add+avg

    print(f"Using {device} device")
    print(f"dataset {dataset}, node_embeddings: {node_embeddings}, metric: {metric}")

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
    for feature_id in lexicon_features:
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
                    df_train_test = pd.concat([train_data, test_data], axis=0)
                    train_data, test_data = train_test_split(df_train_test, test_size=0.1, shuffle=False)
                    print(train_data)
                    print(test_data)
                    langs_train = set(train_data["ISO"].tolist())
                    langs_dev = set(dev_data["ISO"].tolist())
                    langs_test = set(test_data["ISO"].tolist())

                    all_langs = list(set(langs_train) | set(langs_dev) | set(langs_test))
                    num_langs = len(all_langs)
                    lang_dict = {lang: idx for idx, lang in enumerate(all_langs)}
                    print(f"languages in total {num_langs}")

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
                        model = OneFF(device=device, num_langs=num_langs, input_dim=input_dim,
                                      hidden_dim=hidden_dim,
                                      label_dim=label_dim, dropout=0.5)

                        model = model.to(device)

                        optimizer = optim.Adam(model.parameters(), lr=0.001)
                        if langs_list is not None:

                            print(
                                f"train lang {len(langs_train)} dev lang {len(langs_dev)} test lang {len(langs_test)}")

                            if dataset is not None:
                                # uriel, clics, wn, wn_concept
                                if dataset == "uriel":
                                    dataset_path = "data/URIEL/learned_embeddings"
                                    uriel_language_vectors = KeyedVectors.load_word2vec_format(dataset_path,
                                                                                               binary=False)

                                    model_name_ = f"{model_name}_{dataset}"

                                    train_model(model, model_name_, optimizer, train_data, dev_data, test_data,
                                                feature,
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
                                                    feature,
                                                    feature_id,
                                                    lang_dict,
                                                    langs_list,
                                                    output_folder,
                                                    max_epochs=epochs, language_vectors=language_vectors)

                            else:
                                print("get the model")
                                train_model(model, model_name, optimizer, train_data, dev_data, test_data, feature,
                                            feature_id,
                                            lang_dict,
                                            langs_list,
                                            output_folder,
                                            max_epochs=epochs, language_vectors=None)
                        else:
                            print("get the model")
                            train_model(model, model_name, optimizer, train_data, dev_data, test_data, feature,
                                        feature_id,
                                        lang_dict,
                                        langs_list,
                                        output_folder,
                                        max_epochs=epochs, language_vectors=None)
                else:
                    print(f"{outputfile} exists!")
                    print("*" * 50)
        else:
            print(f"train file {train_file} doesn't exist!")


if __name__ == '__main__':
    import plac

    plac.call(run)
