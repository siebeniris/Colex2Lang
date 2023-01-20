import json
import os

import pandas as pd
import plac
import tqdm


def get_edgelists_from_freq_dict(file, dataset):
    # build edgelists from synset pairs
    with open(file) as f:
        colex_freq_dict = json.load(f)

    print(f"{len(colex_freq_dict)} pairs of synsets")
    outputfile = f"data/edgelists/edgelists_{dataset}.csv"
    writer = open(outputfile, "w+")

    header = ["source", "target", "weight"]

    writer.write(",".join(header) + "\n")
    for synset_pair, weight in tqdm.tqdm(colex_freq_dict.items()):
        synset1, synset2 = synset_pair.split("_")
        writer.write(f"{synset1},{synset2},{weight}\n")

    writer.close()


def get_node2id(inputfile, dataset="bn"):
    # data/edgelists/clics_features_all.csv
    df = pd.read_csv(inputfile)
    print(len(df))
    nodes = list(set(df["source"].tolist()) | set(df["target"].tolist()))
    print("nr of synsets ->", len(nodes))
    node2id = {node: idx for idx, node in enumerate(nodes)}
    basefolder = os.path.dirname(inputfile)
    with open(os.path.join(basefolder, f"{dataset}_node2id.json"), "w") as f:
        json.dump(node2id, f)


def edgelist2id(dataset):
    inputfile = f"data/edgelists/edgelists_{dataset}.csv"
    print("loading dataframe")
    df = pd.read_csv(inputfile)
    node2id_file = f"data/edgelists/{dataset}_node2id.json"
    print("loading the node2id")
    with open(node2id_file) as f:
        node2id = json.load(f)

    df["target_id"] = df["target"].apply(lambda x: node2id[str(x)])
    df["source_id"] = df["source"].apply(lambda x: node2id[str(x)])
    print(df.head(5))

    df.to_csv(inputfile, index=False)


def main(task,  dataset, inputfile=None):
    if task == "edgelist":
        get_edgelists_from_freq_dict(inputfile, dataset)

    elif task == "node2id":
        get_node2id(inputfile, dataset)

    elif task == "edge2id":
        edgelist2id(dataset)


if __name__ == '__main__':
    plac.call(main)

