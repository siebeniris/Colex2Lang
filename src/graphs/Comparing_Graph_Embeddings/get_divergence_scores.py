#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import os

import plac
from termcolor import cprint


def main(outputfile="divergence.csv", inputfolder="/Users/yiyichen/Documents/experiments/ColexGraph/data/eval_nodeEmb"):
    result_writer = open(outputfile, "w+")
    # result_writer.write("Dataset\tModel\tDivergence\n")
    # datasets = ["wn", "clics", "bn", "b,/n_concept", "wn_concept"]
    datasets = ["wn", "wn_concept"]
    # datasets = ["clics"]
    modelnames = ["node2vec", "glove", "ggvc", "prone"]

    for dataset in datasets:
        for modelname in modelnames:
            edgelist_file = os.path.join(inputfolder, dataset, f"edgelists_{dataset}")
            ecg_file = os.path.join(inputfolder, dataset, f"{dataset}.ecg")
            model_file = os.path.join(inputfolder, dataset, f"{modelname}_embeddings")
            print(edgelist_file)
            print(ecg_file)
            print(model_file)
            if os.path.exists(edgelist_file) and os.path.exists(ecg_file) and os.path.exists(model_file):
                print("files exist")
                try:
                    div_score = subprocess.run(["./GED", "-g", edgelist_file, "-c", ecg_file, "-e", model_file],
                                               capture_output=True)
                    print(div_score)
                    div_score_ = str(div_score.stdout.decode("utf-8"))
                    div_score_ = div_score_.replace("\n", "").replace("Divergence: ", "")
                    div_score_ = float(div_score_)
                    cprint(f"{dataset}-{modelname}: {div_score_}", color="magenta")
                    result_writer.write(f"{dataset}\t{modelname}\t{div_score_}\n")
                except Exception as msg:
                    cprint(f"{dataset}-{modelname}: {msg}", color="red")

    result_writer.close()


if __name__ == '__main__':
    plac.call(main)
