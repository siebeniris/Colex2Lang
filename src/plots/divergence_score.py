# use logarithmic smoothing and bar chat.
import os
import json

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_divergence_by_metrics(colex_only, inputfile="data/eval_nodeEmb/divergence.csv", outputfolder="data/plots/"):
    sns.set_theme(style="darkgrid", font_scale=0.8)

    df = pd.read_csv(inputfile, sep="\t")


    datasets = {"wn": "WordNet", "wn_concept": "WordNet (Concept)",
                "clics": "CLICS",}

    df["Dataset"].replace(datasets, inplace=True)

    sns.set_theme(style="darkgrid", font_scale=0.8)
    df_groups_feature = df.groupby(by=["metric"])

    for feature, group in df_groups_feature:
        plt.figure(figsize=(12, 12))
        plt.title(f"Divergence Score of Feature {feature}", fontsize=18)
        plt.yscale("log")
        plt.xticks(rotation=45)

        # kg.set_title(f"Divergence Score of Feature {feature}")
        kg = sns.scatterplot(data=group, x="Dataset", y="Divergence", hue="Model")

        fig = kg.get_figure()

        if colex_only:
            fig.savefig(os.path.join(outputfolder, "divergence", "comparing_colex_metrics", f"{feature}_div.png"))
        else:
            fig.savefig(os.path.join(outputfolder, "divergence", f"{feature}_div.png"))
        plt.clf()


def plot_divergence_by_dataset(
                               inputfile="data/eval_nodeEmb/divergence.csv", outputfolder="data/plots/",
                               ):
    sns.set_theme(style="darkgrid", font_scale=0.7)

    df = pd.read_csv(inputfile, sep="\t")

    datasets = {"wn": "WordNet", "wn_concept": "WordNet (Concept)",
                "clics": "CLICS"}

    df["Dataset"].replace(datasets, inplace=True)

    df_datasets_groups = df.groupby(by=["Dataset"])

    for dataset, group in df_datasets_groups:
        print(dataset)

        plt.figure(figsize=(6, 8))

        plt.title(f"Divergence Score of Dataset {dataset}", fontsize=12)

        plt.yscale("log") # smoothing
        plt.xticks(rotation=45)
        # plt.legend(loc="upper right")

        kg = sns.scatterplot(data=group, x="Metrics", y="Divergence", hue="Model", style="Model")
        # kg.set_title(f"Divergence Score of Dataset {dataset}")
        fig = kg.get_figure()

        fig.savefig(os.path.join(outputfolder, "divergence", f"{dataset}_div.png"))
        plt.clf()


if __name__ == '__main__':
    import plac

    # plac.call(plot_divergence_by_features)
    plac.call(plot_divergence_by_dataset)
    # plac.call(plot_divergence_colex)
