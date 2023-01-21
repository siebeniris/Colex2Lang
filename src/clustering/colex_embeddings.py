import json
import os

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nodevectors import ProNE, Glove, GGVec


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


def get_colex_embeddings(dataset, model_name, inputfile, outputfolder="data/colex_embeddings/"):
    # clics- colex_clics3_all.csv
    df = pd.read_csv(inputfile)
    model = load_model(model_name, dataset)
    with open(f"data/colex_node2id/{dataset}_node2id.json") as f:
        node2id = json.load(f)

    df = df[["Concepticon_ID_x", "Concepticon_ID_y", "Colex_ID"]].drop_duplicates()
    colex_len = len(df)
    print(f"There are {colex_len} colex patterns!")
    writer = open(os.path.join(outputfolder, f"{dataset}_{model_name}_embeddings"), "w")
    writer.write(f"{colex_len} 200\n")

    for node1, node2, colex_id in zip(df["Concepticon_ID_x"], df["Concepticon_ID_y"], df["Colex_ID"]):
        if model_name == "node2vec":
            node1_emb = model.wv.get_vector(node2id[str(node1)])
            node2_emb = model.wv.get_vector(node2id[str(node2)])
            colex_emb = np.concatenate((node1_emb, node2_emb), axis=0)
            emb_dim = colex_emb.shape[0]
            print("colex dim:", emb_dim)
            vec = list(colex_emb)
            writer.write(str(colex_id) + ' ')
            vec_str = ['%.9f' % val for val in vec]
            vec_str = " ".join(vec_str)
            writer.write(vec_str + '\n')
        else:
            node1_emb = model[node2id[str(node1)]]
            node2_emb = model[node2id[str(node1)]]
            colex_emb = np.concatenate((node1_emb, node2_emb), axis=0)
            emb_dim = colex_emb.shape[0]
            print("colex dim:", emb_dim)
            vec = list(colex_emb)
            writer.write(str(colex_id) + ' ')
            vec_str = ['%.9f' % val for val in vec]
            vec_str = " ".join(vec_str)
            writer.write(vec_str + '\n')
    writer.close()


def main(dataset, inputfile):
    for model_name in ["node2vec", "prone", "glove", "ggvc"]:
        get_colex_embeddings(dataset, model_name, inputfile)


if __name__ == '__main__':
    import plac

    plac.call(main)
