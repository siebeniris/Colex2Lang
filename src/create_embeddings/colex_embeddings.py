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


def get_colex_embeddings(dataset, connector, model_name, inputfile, outputfolder="data/colex_embeddings/"):
    # clics- colex_clics3_all.csv
    # wn
    # SENSE_LEMMA,LANG3,COLEX,COLEX_ID
    # hylotelephium_telephium,swg,bn:00051613n_bn:00054891n,0
    outputfolder = os.path.join(outputfolder, connector)

    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)

    def get_dim(connector):
        if connector == "concat":
            return 200
        elif connector == "add":
            return 100

    def get_vec_string(colex_emb):
        vec = list(colex_emb)
        vec_str = ['%.9f' % val for val in vec]
        vec_str = " ".join(vec_str)
        return vec_str

    def creating_colex_vector(connector, node1_emb, node2_emb):
        if connector == "concat":
            colex_emb = np.concatenate((node1_emb, node2_emb), axis=0)
            return colex_emb
        elif connector == "add":
            colex_emb = node1_emb + node2_emb
            return colex_emb

    # read the colexification file
    df = pd.read_csv(inputfile)
    # loading the node embeddings model
    model = load_model(model_name, dataset)
    # read the node2id file
    with open(f"data/colex_node2id/{dataset}_node2id.json") as f:
        node2id = json.load(f)

    print(f"workling on {dataset}, {model_name} , {connector}")

    # open the file to write the emebddings in word vectors format.
    writer = open(os.path.join(outputfolder, f"{dataset}_{model_name}_embeddings"), "w")

    if dataset == "clics":
        df = df[["Concepticon_ID_x", "Concepticon_ID_y", "Colex_ID"]].drop_duplicates()
        colex_len = len(df)
        print(f"There are {colex_len} colex patterns!")

        # embedding header
        colex_dim = get_dim(connector)
        print(f"colex dim {colex_dim}")
        writer.write(f"{colex_len} {colex_dim}\n")

        for node1, node2, colex_id in zip(df["Concepticon_ID_x"], df["Concepticon_ID_y"], df["Colex_ID"]):
            if model_name == "node2vec":
                node1_emb = model.wv.get_vector(node2id[str(node1)])
                node2_emb = model.wv.get_vector(node2id[str(node2)])
            else:
                node1_emb = model[node2id[str(node1)]]
                node2_emb = model[node2id[str(node1)]]

            colex_emb = creating_colex_vector(connector, node1_emb, node2_emb)
            assert colex_dim == colex_emb.shape[0]

            writer.write(str(colex_id) + ' ')
            vec_str = get_vec_string(colex_emb)
            writer.write(vec_str + '\n')

    elif dataset == "wn" or dataset == "wn_concept":
        # SENSE_LEMMA,LANG3,COLEX,COLEX_ID
        # hylotelephium_telephium,swg,bn:00051613n_bn:00054891n,0
        df = df[["COLEX", "COLEX_ID"]].drop_duplicates()
        colex_len = len(df)
        print(f"There are {colex_len} colex patterns!")

        # embedding header
        colex_dim = get_dim(connector)
        print(f"colex dim {colex_dim}")
        writer.write(f"{colex_len} {colex_dim}\n")

        for colex, colex_id in zip(df["COLEX"], df["COLEX_ID"]):
            t = colex.split("_")
            if len(t) == 2:
                node1, node2 = t
                if model_name == "node2vec":
                    node1_emb = model.wv.get_vector(node2id[str(node1)])
                    node2_emb = model.wv.get_vector(node2id[str(node2)])
                else:
                    node1_emb = model[node2id[str(node1)]]
                    node2_emb = model[node2id[str(node1)]]

                colex_emb = creating_colex_vector(connector, node1_emb, node2_emb)

                assert colex_dim == colex_emb.shape[0]

                writer.write(str(colex_id) + ' ')
                vec_str = get_vec_string(colex_emb)

                writer.write(vec_str + '\n')

    writer.close()


def main(dataset, connector, inputfile):
    # glove
    # for model_name in ["node2vec", "prone", "ggvc", "glove"]:
    for model_name in ["glove", "node2vec", "prone", "ggvc"]:
        get_colex_embeddings(dataset, connector, model_name, inputfile)


if __name__ == '__main__':
    import plac

    plac.call(main)
