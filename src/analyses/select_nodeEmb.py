import json
import os
from collections import defaultdict

import numpy as np
import warnings
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

warnings.filterwarnings("ignore")

models = ["node2vec", "glove", "ggvc", "prone"]

dataset = "clics"

metrics = ["add+avg", "add+max", "add+sum", "concat+avg", "concat+max", "concat+sum"]

with open("data/TypPred/preprocessed/wals_features.yaml") as f:
    wals_features = yaml.load(f, Loader=Loader)

with open("data/TypPred/preprocessed/feature_dict.json") as f:
    feature_dict = json.load(f)

results_feature_dict = defaultdict(dict)


def get_results(output_dir="output/models/", langs="clics"):
    with open("data/TypPred/preprocessed/feature_dict.json") as f:
        feature_dict = json.load(f)


    folder = os.path.join(output_dir, langs)

    results_feature_dict = defaultdict(dict)
    for feature_area in wals_features:
        results_feature_dict[feature_area] = defaultdict(dict)
        # phonlogy:{"31A":{}}
        for feature_id, feature_name in wals_features[feature_area].items():
            label_dim = len(feature_dict[feature_name]["values"])
            results_feature_dict[feature_area][feature_id] = defaultdict(dict)
            results_feature_dict[feature_area][feature_id]["label_dim"] = label_dim

            train_langs = 0
            train_lang_embeds = 0
            test_lang_embeds = 0
            test_langs = 0
            result_dict_per_feature = dict()
            for file in os.listdir(folder):
                if file.endswith(f"_{feature_id}.json"):
                    print(file)
                    with open(os.path.join(folder, file)) as f:
                        result = json.load(f)
                        if langs in file:
                            # oneff, wn, node2vec, add+avg, feature_id
                            _, t2, t3, t4, _ = file.replace(".json", "").split("_")
                            if result["train"]["lang_embeds_length"] > 0:
                                result_dict_per_feature["_".join((t2, t3, t4))] = {
                                    "test_acc": result["test"]["report"]["accuracy"],
                                    "dev_acc": result["dev"]["report"]["accuracy"],

                                }
                                train_langs = result["train"]["langs_length"]
                                train_lang_embeds = result["train"]["lang_embeds_length"]
                                test_lang_embeds = result["test"]["langs_length"]
                                test_langs = result["test"]["lang_embeds_length"]
                        else:
                            result_dict_per_feature["random"] = {
                                "test_acc": result["test"]["report"]["accuracy"],
                                "dev_acc": result["dev"]["report"]["accuracy"],
                            }
            results_feature_dict[feature_area][feature_id]["results"] = result_dict_per_feature
            results_feature_dict[feature_area][feature_id]["train_langs"]= train_langs
            results_feature_dict[feature_area][feature_id]["train_lang_embeds"] = train_lang_embeds
            results_feature_dict[feature_area][feature_id]["test_langs"] = test_lang_embeds
            results_feature_dict[feature_area][feature_id]["test_lang_embeds"] = test_langs

    with open(f"results_{langs}.json", "w") as f:
        json.dump(results_feature_dict, f)


def get_results_dict(result_file):
    models = ["node2vec", "glove", "prone"]

    with open(result_file) as f:
        results = json.load(f)

    results_dict = defaultdict(list)
    results_dict_zs = defaultdict(list)

    metric_dict = defaultdict(list)
    embed_dict = defaultdict(list)
    for feature, d in results.items():
        for embed, r in d.items():
            if embed != "random":
                model, metric = embed.split("_")
                if model in models:
                    metric_dict[metric].append(r["test_acc"])
                    embed_dict[model].append(r["test_acc"])

            if r["lang_embeds"] > 0:
                # if there are contributions from language embeddings.
                results_dict[embed].append(r["test_acc"])
            else:
                # else:
                results_dict_zs[embed].append(r["test_acc"])
    embed_dict_new = {x: round(np.average(v), 4) for x, v in embed_dict.items()}
    metric_dict_new = {x: round(np.average(v), 4) for x, v in metric_dict.items()}
    new_results_dict = {x: round(np.average(v), 4) for x, v in results_dict.items()}
    new_results_dict_zs = {x: round(np.average(v), 4) for x, v in results_dict_zs.items()}
    sorted(embed_dict_new.items(), key=lambda x: x[1])

    print(f"embed dict {sorted(embed_dict_new.items(), key=lambda x: x[1], reverse=True)}")
    print(f"metric dict {sorted(metric_dict_new.items(), key=lambda x: x[1], reverse=True)}")
    print(f"results dict {sorted(new_results_dict.items(), key=lambda x: x[1], reverse=True)}")
    print(f"results dict zs {sorted(new_results_dict_zs.items(), key=lambda x: x[1], reverse=True)}")

    sorted_results_dict = sorted(new_results_dict.items(), key=lambda x: x[1], reverse=True)
    for result in sorted_results_dict:
        emb, acc = result
        print(f"{emb}, {len(results_dict[emb])}, {acc}")
    print("zs ", "*" * 20)
    sorted_results_dict_zs = sorted(new_results_dict_zs.items(), key=lambda x: x[1], reverse=True)
    for result in sorted_results_dict_zs:
        emb, acc = result
        print(f"{emb}, {len(results_dict_zs[emb])}, {acc}")


if __name__ == '__main__':
    import plac

    plac.call(get_results)
    # plac.call(get_results_dict)
