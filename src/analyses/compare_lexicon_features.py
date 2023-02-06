import json
import pandas as pd
import os

import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from collections import defaultdict
import seaborn as sns

sns.set_theme(style="white")

from sklearn.model_selection import train_test_split, KFold

with open("data/TypPred/feature_dict.json") as f:
    feature_dict = json.load(f)

with open("data/TypPred/wals_features.yaml") as f:
    features = yaml.load(f, Loader=Loader)
lexicon_features = features["Lexicon"]

with open("data/TypPred/feature_maps.json") as f:
    feature_maps = json.load(f)


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


def get_datasets_dist(feature_id, langs):
    #### get train data label distribution dictionary
    feature = lexicon_features[feature_id]
    train_data_label_dist_dict = defaultdict(dict)

    train_file = f"data/TypPred/datasets/features/train_{feature_id}.csv"
    dev_file = f"data/TypPred/datasets/features/dev_{feature_id}.csv"
    test_file = f"data/TypPred/datasets/features/test_{feature_id}.csv"

    langs_list = load_lang_list(langs)

    train_data = pd.read_csv(train_file)
    dev_data = pd.read_csv(dev_file)
    test_data = pd.read_csv(test_file)

    train_data[feature] = train_data[feature].astype("int")
    dev_data[feature] = dev_data[feature].astype("int")
    test_data[feature] = test_data[feature].astype("int")

    train_data = train_data[train_data["ISO"].isin(langs_list)]
    dev_data = dev_data[dev_data["ISO"].isin(langs_list)]

    if len(train_data) > 0 and len(dev_data) > 0:
        df_train_test = pd.concat([train_data, test_data], axis=0)
        train_data, test_data = train_test_split(df_train_test, test_size=0.1, shuffle=False)

        df_train_dev = pd.concat([train_data, dev_data], axis=0)

        train_data_label_dist_dict[feature_id] = df_train_dev[feature].value_counts().to_dict()

    return train_data_label_dist_dict


label_dist = defaultdict(dict)

for feature_id, feature in lexicon_features.items():
    clics_dict = get_datasets_dist(feature_id, "clics")
    total_labels_clics = sum(list(clics_dict[feature_id].values()))
    # ascending order
    wn_dict = get_datasets_dist(feature_id, "wn")

    total_labels_wn = sum(list(wn_dict[feature_id].values()))
    label_dist[feature_id] = {
        "feature": feature,
        "weight_clics": {k: v / total_labels_clics for k, v in clics_dict[feature_id].items()},
        "weight_wn": {k: v / total_labels_wn for k, v in wn_dict[feature_id].items()},
        "id2value": {v: k for k, v in feature_maps[feature].items()}}

#### compare wn and random
lexicon_values_wn_dict = defaultdict(dict)
lexicon_values_clics_dict = defaultdict(dict)

### get the results from wn/clics/wn_concept/random
for feature_id, d in label_dist.items():
    weight_clics = d["weight_clics"]
    weight_wn = d["weight_wn"]
    id2value = d["id2value"]

    with open(f"output/clics/oneff_{feature_id}.json") as f:
        random_clics_report = json.load(f)["test"]["report"]

    with open(f"output/clics/oneff_clics_prone_concat+max_{feature_id}.json") as f:
        clics_report = json.load(f)["test"]["report"]

    random_clics_result = dict()
    clics_result = dict()

    for value_id in id2value:
        if str(value_id) in clics_report:
            clics_result[value_id] = clics_report[str(value_id)]["f1-score"]
        if str(value_id) in random_clics_report:
            random_clics_result[value_id] = random_clics_report[str(value_id)]["f1-score"]

    lexicon_values_clics_dict[feature_id] = {
        "feature": d["feature"],
        "id2value": id2value,
        "label_weights_clics": weight_clics,
        "random_clics": (random_clics_result, random_clics_report["macro avg"]["f1-score"]),
        "clics": (clics_result, clics_report["macro avg"]["f1-score"])
    }

    if os.path.exists(f"output/wn/oneff_{feature_id}.json"):
        with open(f"output/wn/oneff_{feature_id}.json") as f:
            random_wn_report = json.load(f)["test"]["report"]

        with open(f"output/wn/oneff_wn_glove_add+avg_{feature_id}.json") as f:
            wn_report = json.load(f)["test"]["report"]

        with open(f"output/wn/oneff_wn_concept_glove_concat+avg_{feature_id}.json") as f:
            wn_concept_report = json.load(f)["test"]["report"]

        random_wn_result = dict()
        wn_result = dict()
        wn_concept_result = dict()

        for value_id in id2value:
            if str(value_id) in clics_report:
                clics_result[value_id] = clics_report[str(value_id)]["f1-score"]
            if str(value_id) in random_clics_report:
                random_clics_result[value_id] = random_clics_report[str(value_id)]["f1-score"]

            if str(value_id) in random_wn_report:
                random_wn_result[value_id] = random_wn_report[str(value_id)]["f1-score"]
            if str(value_id) in wn_report:
                wn_result[value_id] = wn_report[str(value_id)]["f1-score"]
            if str(value_id) in wn_concept_report:
                wn_concept_result[value_id] = wn_concept_report[str(value_id)]["f1-score"]

        lexicon_values_wn_dict[feature_id] = {
            "feature": d["feature"],
            "id2value": id2value,

            "label_weights_wn": weight_wn,
            "random_wn": (random_wn_result, random_wn_report["macro avg"]["f1-score"]),
            "wn": (wn_result, wn_report["macro avg"]["f1-score"]),
            "wn_concept": (wn_concept_result, wn_concept_report["macro avg"]["f1-score"])
        }


###### to df

def dict2df(lexicon_values_dict, dataset="wn"):
    lexicon_results_dict = defaultdict(dict)

    for feature_id, label_dict in lexicon_values_dict.items():

        print(label_dict)
        feature_results = defaultdict(list)

        feature = label_dict["feature"]
        label_weights = label_dict[f"label_weights_{dataset}"]
        id2value = label_dict["id2value"]

        f1_scores = list()
        values = list()
        models = list()
        weights = list()

        def add_item(model):
            d = label_dict[model][0]
            for v, f in d.items():
                f1_scores.append(f)
                values.append(id2value[v])
                models.append(model)
                weights.append(label_weights[v])

        add_item(f"random_{dataset}")
        add_item(f"{dataset}")
        if dataset == "wn":
            add_item("wn_concept")

        lexicon_results_dict[feature_id] = {
            "Test (F1)": f1_scores,
            "Value": values,
            "Model": models,
            "Weight": weights,
            "Feature": feature
        }

    for feature_id, results in lexicon_results_dict.items():
        df_feature_id = pd.DataFrame.from_dict(results)
        df_feature_id["Model"] = df_feature_id["Model"].replace({"clics": "CLICS",
                                                                 "wn_concept": "WordNet Concept",
                                                                 "random_wn": "Random",
                                                                 "wn": "WordNet",
                                                                 "random_clics": "Random"
                                                                 })

        df_feature_id.to_csv(f"output/lexicon/{dataset}_{feature_id}.csv", index=False)


dict2df(lexicon_values_clics_dict, "clics")
dict2df(lexicon_values_wn_dict, "wn")
