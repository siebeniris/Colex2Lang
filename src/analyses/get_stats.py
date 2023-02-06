import pandas as pd
from collections import defaultdict

def get_stats_wn(filepath, dataset="wn_concept"):
    df = pd.read_csv(filepath)
    lang_stats_dict = defaultdict(dict)
    print(f"loading the {filepath}...")

    for lemma, lang, colex, colex_id in zip(df["SENSE_LEMMA"], df["LANG3"], df["COLEX"], df["COLEX_ID"]):
        if lang not in lang_stats_dict:
            lang_stats_dict[lang] = defaultdict(list)
        lang_stats_dict[lang]["LEMMA"].append(lemma)
        lang_stats_dict[lang]["SYNSET"]+= colex.split("_")
        lang_stats_dict[lang]["COLEX_ID"].append(colex_id)

    lang_dict = defaultdict[list]
    for lang, d in lang_stats_dict:
        for k,v in d.items():
            # lang_dict[lang]
            pass

