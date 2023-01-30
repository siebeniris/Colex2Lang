import os
import json
import pickle
import pandas as pd

import networkx as nx
import community
import numpy as np
from nodevectors import ProNE, Glove, GGVec
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from termcolor import cprint
from node2vec import Node2Vec

from src.graphs.graph_partition_and_measures import partition_networkx
from src.graphs.models import get_graph_from_feature

N_COMPONENTS = 100
SEED = 42
TEST_SIZE = 0.2


# implement first order method for GGVec and ProNE
# https://github.com/VHRanger/nodevectors/blob/master/examples/link%20prediction.ipynb
# Better for clustering and label prediction tasks.

# https://github.com/sbonner0/unsupervised-graph-embedding
# https://www.semanticscholar.org/reader/dc0d39405c3593c6dad9d26f24a97f7a631ecc5d


def generate_ecg(dataset):
    input_folder = f"/Users/yiyichen/Documents/experiments/LangSim/data/edgelists"
    output_folder = "data/eval_nodeEmb/"
    nodefile = os.path.join(input_folder, dataset, "pmi_node2id.json")

    file = os.path.join(input_folder, dataset, "features_all.csv")

    G = get_graph_from_feature(file, nodefile, "normalized_weight")

    cprint("best partition on the graph ...", "red")
    ml = community.best_partition(G)
    cprint("ecg on the graph ...", "red")
    ec = community.ecg(G, ens_size=32)
    node_ec = ec.partition
    sorted_node_ec = dict(sorted(node_ec.items()))

    ec_W = dict((':'.join(str(k).split(',')), v) for k, v in ec.W.items())
    cprint("pickle ec ...", "green")
    with open(os.path.join(output_folder, dataset, f"{dataset}_partition.json"), "w") as f:
        json.dump(ec.partition, f)

    with open(os.path.join(output_folder, dataset, f"{dataset}_W.json"), "w") as f:
        json.dump(ec_W, f)

    with open(os.path.join(output_folder, dataset, f"{dataset}_CSI.json"), "w") as f:
        json.dump(ec.CSI, f)

    # with open(os.path.join(output_folder, f"{feature}_ecg.pickle"), "wb") as f:
    #     pickle.dump(ec, f)
    cprint("write down ecg file ...", "green")
    with open(os.path.join(output_folder, dataset, f"{dataset}.ecg"), "w") as writer:
        for x in sorted_node_ec.values():
            writer.write(str(x) + "\n")
    cprint(f"{dataset} ->{community.modularity(ml, G)}, {community.modularity(ec.partition, G)}",
           "magenta")


def load_model(model_name, input_folder="data/node_embeddings"):
    if model_name == "node2vec":
        filepath = os.path.join(input_folder, f"{model_name}.bin")
        return KeyedVectors.load(filepath)
    else:
        filepath = os.path.join(input_folder, f"{model_name}.zip")
        if model_name == "prone":
            return ProNE.load(filepath).model
        if model_name == "ggvc":
            return GGVec.load(filepath).model
        if model_name == "glove":
            return Glove.load(filepath).model


def embeddings_format(dataset, output_folder="data/eval_nodeEmb",  input_folder="data/language_embeddings"):
    # datasets = ["wn", "wn_concept", "clics"]
    modelnames = ["node2vec", "glove", "ggvc", "prone"]
    metrics = ["add+avg", "add+max", "add+sum", "concat+avg", "concat+max", "concat+sum"]

    def get_vec_string(lang_emb):
        vec = list(lang_emb)
        vec_str = ['%.9f' % val for val in vec]
        vec_str = " ".join(vec_str)
        return vec_str

    node_folder = f"/Users/yiyichen/Documents/experiments/LangSim/data/edgelists/{dataset}"

    with open(os.path.join(node_folder, "pmi_node2id.json")) as f:
        node2id = json.load(f)

    for model in modelnames:
        for metric in metrics:
            embeddings_path = os.path.join(input_folder, metric, f"{dataset}_{model}_embeddings")
            output_folder_ = os.path.join(output_folder, dataset, metric)
            if not os.path.exists(output_folder_):
                os.mkdir(output_folder_)
            output_path = os.path.join(output_folder_, f"{dataset}_{model}_embeddings")
            embeddings = KeyedVectors.load_word2vec_format(embeddings_path, binary=False)
            size = len(embeddings.key_to_index)
            first_key = list(embeddings.key_to_index)[0]
            embed_size = embeddings[first_key].size

            with open(output_path, "w") as writer:
                writer.write(f"{size} {embed_size}\n")
                for lang, idx in node2id.items():
                    writer.write(str(idx) + ' ')
                    lang_vec = embeddings[lang]
                    vec_str = get_vec_string(lang_vec)
                    writer.write(vec_str + '\n')


def convert2eval_format(dataset, output_folder="data/eval_nodeEmb"):
    # create edgelist
    # using the language normalized_weight from LangSim
    input_folder = f"/Users/yiyichen/Documents/experiments/LangSim/data/edgelists/{dataset}"
    df = pd.read_csv(os.path.join(input_folder, "features_all.csv"))

    with open(os.path.join(input_folder, "pmi_node2id.json")) as f:
        node2id = json.load(f)

    feature = "normalized_weight"
    print(f"length {len(df)}")
    df_feature = df[["target", "source", feature]]
    df_feature = df_feature.dropna(subset=[feature])
    print(df_feature.head())

    df_feature["target"] = df_feature["target"].apply(lambda x: node2id[x])
    df_feature["source"] = df_feature["source"].apply(lambda x: node2id[x])

    with open(os.path.join(output_folder, dataset, f"edgelist"), "w") as writer:
        for tgt, src in zip(df_feature["target"], df_feature["source"]):
            writer.write(f"{tgt} {src}\n")


"""
Use topological features as targets for the graph embeddings to predict as down-stream tasks.
Hypothesis: if an embedding truly has captured a good representation of the graph's topology, it should be 
able to make some accurate predictions about these features.
bin the real-valued features into a series of classes via the use of a histogram.
"""


def main(task, dataset, output_folder="data/eval_nodeEmb"):
    if task == "format":
        convert2eval_format(dataset, output_folder)
    elif task == "emb2id":
        embeddings_format(dataset, output_folder)
    elif task == "ecg":
        generate_ecg(dataset)


if __name__ == '__main__':
    import plac

    plac.call(main)
