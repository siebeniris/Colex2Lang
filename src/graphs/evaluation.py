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
from src.graphs.models import get_synset_pair_graphs

N_COMPONENTS = 64
SEED = 42
TEST_SIZE = 0.2


# implement first order method for GGVec and ProNE
# https://github.com/VHRanger/nodevectors/blob/master/examples/link%20prediction.ipynb
# Better for clustering and label prediction tasks.

# https://github.com/sbonner0/unsupervised-graph-embedding
# https://www.semanticscholar.org/reader/dc0d39405c3593c6dad9d26f24a97f7a631ecc5d


def generate_ecg(dataset):
    colex_folder = "data/edgelists"
    print("reading the files for colex and node2id by feature...")
    output_folder = "data/eval_nodeEmb/"

    file = os.path.join(colex_folder, f"edgelists_{dataset}.csv")

    G = get_synset_pair_graphs(file, dataset)
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


def model2embeddings(dataset="wn", model_folder="data/node_embeddings", output_folder="data/eval_nodeEmb"):
    """
    Convert pretrained graph embedding models to node2vec format for evaluation.
    """
    inputfolder = os.path.join(model_folder, dataset)

    output_folder_colex = os.path.join(output_folder, dataset)
    if not os.path.exists(output_folder_colex):
        os.mkdir(output_folder_colex)

    for file in os.listdir(inputfolder):
        if file.endswith(".zip") or file.endswith(".bin"):
            model_name = file.replace(".zip", "").replace(".bin", "")
            model = load_model(model_name, inputfolder)
            print(dataset, model_name )
            if model_name != "node2vec":
                model_size = len(model)
                first_k = list(model.keys())[0]
                emb_dim = model[first_k].shape[0]
                with open(os.path.join(output_folder_colex, f"{model_name}_embeddings"), "w") as writer:
                    writer.write(f"{model_size} {emb_dim}\n")
                    for key, vec in model.items():
                        vec = list(vec)
                        writer.write(str(key) + ' ')
                        vec_str = ['%.9f' % val for val in vec]
                        vec_str = " ".join(vec_str)
                        writer.write(vec_str + '\n')
            else:
                # node2vec
                keys = list(model.wv.key_to_index)
                model_size = len(keys)
                emb_dim = model.wv[keys[0]].shape[0]

                with open(os.path.join(output_folder_colex, f"{model_name}_embeddings"),
                          "w") as writer1:
                    writer1.write(f"{model_size} {emb_dim}\n")
                    for key in keys:
                        vec = model.wv[key]
                        vec = list(vec)
                        writer1.write(str(key) + ' ')
                        vec_str = ['%.9f' % val for val in vec]
                        vec_str = " ".join(vec_str)
                        writer1.write(vec_str + '\n')


def convert2eval_format(dataset, output_folder="data/eval_nodeEmb"):
    # create edgelist

    df = pd.read_csv(f"data/edgelists/edgelists_{dataset}.csv")

    with open(os.path.join(output_folder, dataset, f"edgelists_{dataset}"), "w") as writer:
        for source_id, target_id in zip(df["source_id"], df["target_id"]):
            writer.write(f"{source_id} {target_id}\n")



"""
Use topological features as targets for the graph embeddings to predict as down-stream tasks.
Hypothesis: if an embedding truly has captured a good representation of the graph's topology, it should be 
able to make some accurate predictions about these features.
bin the real-valued features into a series of classes via the use of a histogram.
"""


def main(task, dataset, model_folder="data/node_embeddings", output_folder="data/eval_nodeEmb"):
    if task == "format":
        convert2eval_format(dataset, output_folder)
    elif task == "model2emb":
        model2embeddings(dataset, model_folder, output_folder)
    elif task == "ecg":
        generate_ecg(dataset)


if __name__ == '__main__':
    import plac
    plac.call(main)

