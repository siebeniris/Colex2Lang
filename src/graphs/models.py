import json
import os

import plac
import pandas as pd
import networkx as nx
from nodevectors import ProNE, GGVec, GraRep, Glove
from termcolor import colored, cprint
from sklearn.decomposition import TruncatedSVD
from node2vec import Node2Vec
import numpy as np


# https://github.com/chihming/awesome-network-embedding

#  https://github.com/thunlp/openne

### Experiment with models:
# node2vec, Verse, ProNE, SPARSE attention graph networks
# measurements https://github.com/ftheberge/graph-partition-and-measures
# https://github.com/VHRanger/nodevectors (node2vec, GGVec, ProNE, GraRep, GloVe)


def get_synset_pair_graphs(file, dataset):
    # source        target          weight  target_id  source_id
    # bn:00051613n  bn:00054891n     520      21559      43098

    df = pd.read_csv(file)
    cprint(df.head(2), "green")

    cprint("creating a weighted graph for the synset pairs...", "magenta")
    G = nx.Graph()
    for tgt, src, weight in zip(df["target_id"], df["source_id"], df["weight"]):
        G.add_edge(tgt, src, weight=weight)

    G = G.to_undirected()

    return G


def generate_prone_embeddings(G, N_COMPONENTS):
    # https://github.com/VHRanger/nodevectors
    cprint("Train ProNE embeddings ...", "blue", attrs=["bold"])
    pne_params = dict(
        n_components=N_COMPONENTS,
        step=5,
        mu=0.2,
        theta=0.5,
    )

    prone_model = ProNE(
        **pne_params
    )
    prone_model.fit_transform(G)
    return prone_model


def generate_ggvc_embeddings(G, N_COMPONENTS):
    cprint("Train GGVC embeddings ...", "blue", attrs=["bold"])
    ggvec_params = dict(
        n_components=N_COMPONENTS,
        order=1,
        tol=0.07,
        tol_samples=10,
        max_epoch=6_000,
        learning_rate=0.1,
        negative_ratio=0.15,
        exponent=0.33,
        verbose=True,
    )
    ggvc_model = GGVec(**ggvec_params)
    ggvc_model.fit_transform(G)
    return ggvc_model


def generate_glove_embeddings(G, N_COMPONENTS):
    # GloVe with random walks.
    glove_params = dict(
        n_components=N_COMPONENTS,
        tol=0.0005,
        max_epoch=6_000,
        learning_rate=0.02,
        max_loss=10.,
        # max_count=50,
        exponent=0.5,
    )
    cprint("Train GloVe embeddings ...", "blue", attrs=["bold"])
    glove_model = Glove(**glove_params)
    glove_model.fit_transform(G)
    return glove_model


def generate_grarep_embeddings(G, N_COMPONENTS):
    cprint("Train GraRep embeddings ...", "blue", attrs=["bold"])

    grarep_params = dict(
        n_components=N_COMPONENTS,
        order=2,
        embedder=TruncatedSVD(
            n_iter=10,
            random_state=42),
        # merger=(lambda x : np.sum(x, axis=0)),
        merger=lambda x: x[-1]
    )
    grep = GraRep(**grarep_params)
    grep.fit_transform(G)
    return grep


def generate_verse_embeddings():
    """
    VERSE(Tsitsulin et al. 2018)

    """
    # https://github.com/xgfs/verse

    pass


def generate_node2vec_embeddings(G, N_COMPONENTS):
    # before : 50
    # default values
    node2vec = Node2Vec(G, dimensions=N_COMPONENTS, walk_length=10, num_walks=80, workers=8,
                        weight_key="weight", temp_folder="data/nodeEmb/node2vec")

    # embed nodes
    model = node2vec.fit(window=5, min_count=2, batch_words=4)
    return model


def generate_node_embeddings_batch(model, N_COMPONENTS=100, dataset="bn", file="data/edgelists/edgelists_bn.csv",
                                   output_folder="data/node_embeddings"):
    G = get_synset_pair_graphs(file, dataset)

    if model == "node2vec":
        embeds = generate_node2vec_embeddings(G, N_COMPONENTS)
    elif model == "prone":
        embeds = generate_prone_embeddings(G, N_COMPONENTS)
    elif model == "grarep":
        embeds = generate_grarep_embeddings(G, N_COMPONENTS)  # doesn't work
    elif model == "ggvc":
        embeds = generate_ggvc_embeddings(G, N_COMPONENTS)
    elif model == "glove":
        embeds = generate_glove_embeddings(G, N_COMPONENTS)  # didn't converge

    if model == "node2vec":
        embeds.save(os.path.join(output_folder, f"{model}.bin"))
    else:
        embeds.save(os.path.join(output_folder, f"{model}"))


def main(inputfile, dataset="bn"):
    output_folder = f"data/node_embeddings/{dataset}"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for model in ["node2vec", "prone", "ggvc", "glove"]:

        if model == "node2vec":
            filepath = os.path.join(output_folder, f"{model}.bin")
            if not os.path.exists(filepath):
                print(model, dataset)
                generate_node_embeddings_batch(model, 100, dataset, inputfile, output_folder)

        if model != "node2vec":
            filepath = os.path.join(output_folder, f"{model}.zip")
            if not os.path.exists(filepath):
                print(model, dataset)
                generate_node_embeddings_batch(model, 100, dataset, inputfile, output_folder)


if __name__ == '__main__':
    plac.call(main)
