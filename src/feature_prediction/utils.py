import pandas as pd
import json
import os


def lang_clics_wn():
    lang_embeddings_dir = "data/language_embeddings"
    with open(os.path.join(lang_embeddings_dir, "clics_langs.json")) as f:
        clics_langs = json.load(f)
    with open(os.path.join(lang_embeddings_dir, "wn_concept_langs.json")) as f:
        wn_concept_langs = json.load(f)
    with open(os.path.join(lang_embeddings_dir, "wn_langs.json")) as f:
        wn_langs = json.load(f)

    clics_langs = set(clics_langs)
    wn_concept_langs = set(wn_concept_langs)
    wn_langs = set(wn_langs)

    common = list(clics_langs.intersection(wn_langs, wn_concept_langs))
    print(len(common), common[:10])
    with open(os.path.join(lang_embeddings_dir, "common_langs.json"), "w") as f:
        json.dump(common, f)


def get_langs_inter_typpred(file, sep, dataset):
    lang_embeddings_dir = "data/language_embeddings"
    print(f"file -> {file}")
    with open(os.path.join(lang_embeddings_dir, f"{dataset}_langs.json")) as f:
        langs = json.load(f)
    df = pd.read_csv(file, sep=sep)
    print(f"length: {len(df)}")
    df_filter = df[df["wals_code"].isin(langs)]
    print(f'length: {len(df_filter)}')


def load_train_data():
    df = pd.read_csv("data/TypPred/preprocessed/train.csv")



if __name__ == '__main__':
    import plac
    # plac.call(lang_clics_wn)
    plac.call(get_langs_inter_typpred)