# use logarithmic smoothing and bar chat.
import os
import json

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_csi(colex_only=False):
    colex_names = ["wn", "clics", "bn", "bn_concept", "wn_concept"]
    if colex_only:
        features = ["pmi", "chi_sq", "phi_sq", "jaccard"]
    else:
        features = ["pmi", "similarity", "featural", "phonological", "syntactic", "genetic", "geographic", "inventory"]
    # features = ["pmi"]

    modelnames = ["node2vec", "glove", "ggvc", "prone"]

    inputfolder = "data/eval_nodeEmb/"

    rows = []
    for colex in colex_names:
        for feature in features:
            csi_file = os.path.join(inputfolder, colex, f"{feature}_CSI.json")
            with open(csi_file) as f:
                csi_score = f.read().replace("\n", "")
                rows.append((colex, feature, csi_score))

    df_csi = pd.DataFrame.from_records(rows, columns=["Dataset", "Feature", "CSI"])
    df_csi["CSI"] = df_csi["CSI"].astype("float")

    df_csi = df_csi.round(4)

    d = {'pmi': "Colexification",
         'similarity': "Lexical",
         'featural': "Featural",
         'phonological': "Phonological",
         'syntactic': "Syntactic",
         'genetic': "Genetic",
         'geographic': "Geographic",
         'inventory': "Inventory"}

    df_csi["Feature"].replace(d, inplace=True)

    sns.set_theme(style="darkgrid", font_scale=0.8)
    g = sns.scatterplot(data=df_csi, x="Feature", y="CSI", hue="Dataset", style="Dataset")
    g.figure.set_size_inches(6.5, 4.5)

    g.set_title("Community Strength of Linguistic Features")

    fig = g.get_figure()
    if colex_only:
        fig.savefig(os.path.join("data/plots/csi", "csi_colex.png"))
    else:
        fig.savefig(os.path.join("data/plots/csi", "csi.png"))


if __name__ == '__main__':
    import plac

    plac.call(plot_csi)
