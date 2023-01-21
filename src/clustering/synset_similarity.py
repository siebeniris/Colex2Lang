import json
import os

import pandas as pd
from gensim.models import KeyedVectors
from nodevectors import ProNE, Glove, GGVec
from sklearn.metrics.pairwise import cosine_similarity

def load_model(model_name, dataset, input_folder="data/node_embeddings"):
    if model_name == "node2vec":
        filepath = os.path.join(input_folder, dataset, f"{model_name}.bin")
        return KeyedVectors.load(filepath)
    else:
        filepath = os.path.join(input_folder, dataset, f"{model_name}.zip")
        if model_name == "prone":
            return ProNE.load(filepath).model
        if model_name == "ggvc":
            return GGVec.load(filepath).model
        if model_name == "glove":
            return Glove.load(filepath).model


def main(dataset):
    node2id_file = f"data/colex_node2id/{dataset}_node2id.json"
    with open(node2id_file) as f:
        node2id = json.load(f)

    df = pd.read_csv(f"data/edgelists/edgelists_{dataset}.csv")

    for model_name in ["node2vec", "prone", "ggvc", "glove"]:
        model = load_model(model_name, dataset)
        print(f"loading model {model_name}")
        sims = []
        if model_name == "node2vec":
            for src, tgt in zip(df["target_id"], df["source_id"]):
                sim = model.wv.similarity(src, tgt)
                sims.append(sim)
        else:
            for src, tgt in zip(df["target_id"], df["source_id"]):
                # sim = cosine_similarity(model[src], model[tgt])
                tgt_arr = model[src].reshape(1, -1)
                src_arr = model[tgt].reshape(1, -1)

                sim = cosine_similarity(tgt_arr, src_arr)[0][0]
                sims.append(sim)

        df[f"{model_name}_sim"] = sims
    df.to_csv(f"data/edgelists/edgelists_{dataset}.csv", index=False)


if __name__ == '__main__':
    import plac
    plac.call(main)

