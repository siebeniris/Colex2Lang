import pandas as pd
import json
import plac
import os


def get_synset_pairs_wn(inputfile, dataset):
    # langsim/data/linguistic_features/wn_all.csv
    df = pd.read_csv(inputfile)

    print("language nr.", len(df["LANG3"].value_counts()))
    print("len ->", len(df))
    df = df[["SENSE_LEMMA", "LANG3", "COLEX"]].drop_duplicates()

    print("lexicalizations, ->", len(df[["SENSE_LEMMA", "LANG3"]].drop_duplicates()))
    print("len ->", len(df))
    counter = df["COLEX"].value_counts()
    print(counter)

    # with open(f"data/synset_pairs/{dataset}_colexfreq.json", "w") as f:
    #     json.dump(counter.to_dict(), f)


def get_concept_pairs_clics(inputfile, dataset):
    df = pd.read_csv(inputfile)
    print("language nr.", len(df["Language_ID"].value_counts()))
    print("len ->", len(df))
    df["Concepticon_ID_x"] = df["Concepticon_ID_x"].astype(str)
    df["Concepticon_ID_y"] = df["Concepticon_ID_y"].astype(str)

    df["COLEX"] = df[["Concepticon_ID_x", "Concepticon_ID_y"]].apply(lambda x: "_".join(x), axis=1)
    df = df[["Concepticon_ID_x", "Concepticon_ID_y", "Language_ID", "COLEX", "Form", "Value"]].drop_duplicates()
    print("len->", len(df))
    print("lexicalizations, ->", len(df[["Form", "Language_ID"]].drop_duplicates()))

    counter = df["COLEX"].value_counts()
    print(counter)

    with open(f"data/synset_pairs/{dataset}_colexfreq.json", "w") as f:
        json.dump(counter.to_dict(), f)


if __name__ == '__main__':
    plac.call(get_concept_pairs_clics)
