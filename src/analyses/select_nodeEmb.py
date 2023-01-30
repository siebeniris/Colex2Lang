import json
import os

import warnings

warnings.filterwarnings("ignore")

from collections import defaultdict

models = ["node2vec", "glove", "ggvc", "prone"]

dataset = "clics"

metrics = ["add+avg", "add+max", "add+sum", "concat+avg", "concat+max", "concat+sum"]

lexicon_features = ['129A', '130A', '130B', '131A', '132A', '133A', '134A', '135A', '136A', '136B', '137A', '137B',
                    "138A"]

output_dir = "output/models/"
results = defaultdict(dict)
for feature in lexicon_features:
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
                            "lang_embeds": result["test"]["lang_embeds_length"] / result["test"]["langs_length"]
                        }
                else:
                    results[feature]["random"] = {
                        "test_acc": result["test"]["report"]["accuracy"],
                        "lang_embeds": result["test"]["lang_embeds_length"] / result["test"]["langs_length"]
                    }

with open("results.json", "w") as f:
    json.dump(results, f)
