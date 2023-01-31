import json
import os
from collections import defaultdict

import warnings

warnings.filterwarnings("ignore")

models = ["node2vec", "glove", "ggvc", "prone"]

dataset = "clics"

metrics = ["add+avg", "add+max", "add+sum", "concat+avg", "concat+max", "concat+sum"]


def get_results(output_dir="output/models/", langs="clics"):
    with open("data/TypPred/preprocessed/feature_dict.json") as f:
        feature_dict = json.load(f)

    feature_ids = [v["id"] for v in feature_dict.values()]

    folder = os.path.join(output_dir, langs)
    results = defaultdict(dict)
    for feature in feature_ids:
        results[feature] = defaultdict(dict)

        for file in os.listdir(folder):
            if file.endswith(f"_{feature}.json"):
                print(file)
                with open(os.path.join(folder, file)) as f:
                    result = json.load(f)
                    if "clics" in file:
                        t1, t2, t3, t4, t5 = file.replace(".json", "").split("_")
                        if result["train"]["lang_embeds_length"] > 0:
                            results[feature]["_".join((t3, t4))] = {
                                "test_acc": result["test"]["report"]["accuracy"],
                                "lang_embeds": result["test"]["lang_embeds_length"]
                            }
                    else:
                        results[feature]["random"] = {
                            "test_acc": result["test"]["report"]["accuracy"],
                            "lang_embeds": result["test"]["lang_embeds_length"]
                        }

    with open(f"results_{langs}.json", "w") as f:
        json.dump(results, f)


def get_results_dict(results):
    results_dict = defaultdict(list)
    results_dict_zs = defaultdict(list)

    metric_dict = defaultdict(list)
    embed_dict = defaultdict(list)
    for feature, d in results.items():
        for embed, r in d.items():
            if embed != "random":
                model, metric = embed.split("_")
                metric_dict[metric].append(r["test_acc"])
                embed_dict[model].append(r["test_acc"])

            if r["lang_embeds"] > 0:
                results_dict[embed].append(r["test_acc"])
            else:
                results_dict_zs[embed].append(r["test_acc"])


if __name__ == '__main__':
    import plac

    plac.call(get_results)
