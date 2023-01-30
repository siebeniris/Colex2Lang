import json
import os
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")


models = ["node2vec", "glove", "ggvc", "prone"]

dataset = "clics"

metrics = ["add+avg", "add+max", "add+sum", "concat+avg", "concat+max", "concat+sum"]

with open("data/TypPred/preprocessed/feature_dict.json") as f:
    feature_dict = json.load(f)

feature_ids = [v["id"] for v in feature_dict.values()]


output_dir = "output/models/"
results = defaultdict(dict)
for feature in feature_ids:
    results[feature] = defaultdict(dict)

    for file in os.listdir(output_dir):
        if file.endswith(f"_{feature}.json"):
            print(file)
            with open(os.path.join(output_dir, file)) as f:
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

with open("results.json", "w") as f:
    json.dump(results, f)
