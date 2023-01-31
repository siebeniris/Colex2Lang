import pandas as pd
import json
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import yaml
from termcolor import cprint

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from collections import defaultdict


def compile_results(filepath, feature_ids_file=None):
    with open(filepath) as f:
        results = json.load(f)

    feature_area_result_dict = defaultdict(dict)
    feature_area_result_dict_zs = defaultdict(dict)

    feature_area_labels = dict()
    model_acc_dict = defaultdict(list)
    model_acc_dict_zs = defaultdict(list)
    #
    # metric_dict = defaultdict(list)
    # metric_dict_zs = defaultdict(list)

    if feature_ids_file is not None:
        with open(feature_ids_file) as f:
            feature_ids = json.load(f)

    feature_ids_tested = []
    for feature_area, feature_result in results.items():
        # model:[scores]
        print(feature_area)
        feature_result_dict = defaultdict(list)
        feature_result_dict_zs = defaultdict(list)
        dev_dict = defaultdict(list)
        feature_labels = []
        # print(feature_result)
        for feature_id in feature_result:
            if feature_ids_file is not None:
                if feature_id in feature_ids:
                    if "test_lang_embeds" in feature_result[feature_id]:
                        feature_ids_tested.append(feature_id)
                        feature_labels.append(feature_result[feature_id]["label_dim"])
                        result_models = feature_result[feature_id]["results"]
                        test_lang_embeds = feature_result[feature_id]["test_lang_embeds"]

                        for model, r in result_models.items():
                            feature_result_dict[model].append(r["test_acc"])
                            model_acc_dict[model].append(r["test_acc"])
                            # if model != "random":
                            #     t1, mo, me = model.split("_")
                            #     metric_dict[me].append(r["test_acc"])

                            if test_lang_embeds == 0:
                                feature_result_dict_zs[model].append(r["test_acc"])
                                model_acc_dict_zs[model].append(r["test_acc"])
                                # if model != "random":
                                #     t1, mo, me = model.split("_")
                                #     metric_dict_zs[me].append(r["test_acc"])
            else:
                if "test_lang_embeds" in feature_result[feature_id]:
                    feature_ids_tested.append(feature_id)
                    feature_labels.append(feature_result[feature_id]["label_dim"])
                    result_models = feature_result[feature_id]["results"]
                    test_lang_embeds = feature_result[feature_id]["test_lang_embeds"]

                    for model, r in result_models.items():
                        feature_result_dict[model].append(r["test_acc"])
                        model_acc_dict[model].append(r["test_acc"])
                        # if model != "random":
                        #     t1, mo, me = model.split("_")
                        #     metric_dict[me].append(r["test_acc"])

                        if test_lang_embeds == 0:
                            feature_result_dict_zs[model].append(r["test_acc"])
                            model_acc_dict_zs[model].append(r["test_acc"])
                            # if model != "random":
                            #     t1, mo, me = model.split("_")
                            #     metric_dict_zs[me].append(r["test_acc"])

        feature_area_labels[feature_area] = round(np.average(feature_labels), 1)



        for model, scores in feature_result_dict.items():
            feature_area_result_dict[feature_area][model] = (round(np.average(scores), 4), len(scores))

        for model, scores in feature_result_dict_zs.items():
            feature_area_result_dict_zs[feature_area][model] = (round(np.average(scores), 4), len(scores))

    basename = os.path.basename(filepath).split("_")[1]

    if feature_ids_file is None:

        with open(f"output/results/features_ids_{basename}", "w") as f:
            json.dump(feature_ids_tested, f)


    return feature_area_result_dict, feature_area_result_dict_zs, model_acc_dict, model_acc_dict_zs, feature_area_labels


def restructure_dict(d):
    # {model:[acc1, acc2, ...., accn]}
    new_dict = dict()
    for m, acc in d.items():
        new_dict[m] = (round(np.average(acc), 3), len(acc))
    return sorted(new_dict.items(), key=lambda x: x[1][0], reverse=True)


def show_results(filepath, feature_ids_file=None):
    feature_area_result_dict, feature_area_result_dict_zs, model_acc_dict, model_acc_dict_zs, feature_area_labels = compile_results(
        filepath, feature_ids_file=feature_ids_file)

    print("TEST- AVERAGE")

    acc_ = []

    acc_dict= defaultdict(list)


    for feature_area, model_results in feature_area_result_dict.items():
        print(f"{feature_area} : label_dim {feature_area_labels[feature_area]}")
        model_restructed = sorted(model_results.items(), key=lambda x: x[1][0], reverse=True)
        for m, acc in model_results.items():
            acc_dict[m].append(acc)
        print(model_restructed[:5])
        # print(model_results["random"])
        # acc_.append(model_results["random"][0])

    print("average")
    acc_dict_ = {k:np.average(v) for k,v in acc_dict.items()}
    print(sorted(acc_dict_.items(), key=lambda x: x[1], reverse=True))

    print("*" * 30)
    # print(np.average(acc_))
    #
    #
    print("ZS- TEST- AVERAGE")
    # acc_ = []
    acc_dict=defaultdict(list)
    for feature_area, model_results in feature_area_result_dict_zs.items():
        print(f"{feature_area} : label_dim {feature_area_labels[feature_area]}")
        # acc_.append(model_results["random"][0])
        # print(f"{feature_area} : label_dim {feature_area_labels[feature_area]}")
        # print(model_results["random"])
        for m, acc in model_results.items():
            acc_dict[m].append(acc)
        model_restructed= sorted(model_results.items(), key=lambda x: x[1][0], reverse=True)
        print(model_restructed[:5])

    print("average")
    acc_dict_ = {k:np.average(v) for k,v in acc_dict.items()}

    print(sorted(acc_dict_.items(), key=lambda x: x[1], reverse=True))


    print("*" * 30)
    #
    # print(np.average(acc_))


def create_dfs(filepath, feature_ids_file=None):
    """
    results for clics to compare different embeddings and different metrics.

    :param filepath:
    :param feature_ids_file:
    :return:
    """
    feature_area_result_dict, feature_area_result_dict_zs, model_acc_dict, model_acc_dict_zs, feature_area_labels = compile_results(
        filepath, feature_ids_file=feature_ids_file)

    print("TEST- AVERAGE")

    acc_ = []

    columns =["NodeEmb", "Metric", "Complex_Sentences", "Lexicon", "Morphology",
              "Nominal_Categories", "Nominal_Syntax", "Other", "Phonology", "Sign_Languages",
              "Simple_Cluases", "Verbal_Categories", "Word_Order" ]

    acc_dict = defaultdict(list)

    df_dict = defaultdict(dict)

    for feature_area, model_results in feature_area_result_dict.items():
        print(f"{feature_area} : label_dim {feature_area_labels[feature_area]}")

        model_restructed = sorted(model_results.items(), key=lambda x: x[1][0], reverse=True)

        for m, acc in model_results.items():
            if m not in df_dict:
                df_dict[m] = defaultdict(dict)
            if "clics" in m:
                # model, nodeEmb, metric
                t1, t2, t3 = m.split("_")
                df_dict[m]["NodeEmb"]= t2
                df_dict[m]["Metric"] = t3
                df_dict[m][feature_area]= acc[0]
            else:
                if m=="random":
                    df_dict[m]["NodeEmb"] = None
                    df_dict[m]["Metric"] = None
                    df_dict[m][feature_area] = acc[0]


            acc_dict[m].append(acc)
        print(model_restructed[:5])
        # print(model_results["random"])
        # acc_.append(model_results["random"][0])

    print("average")
    acc_dict_ = {k: np.average(v) for k, v in acc_dict.items()}
    print(sorted(acc_dict_.items(), key=lambda x: x, reverse=True))

    df = pd.DataFrame.from_dict(df_dict, columns=columns, orient="index")
    df = df.dropna(axis=1, how="all")
    print(df)

    df.to_csv("output/results/clics_metrics_nodeEmb_test.csv")

    print("*" * 30)
    # print(np.average(acc_))
    #
    #
    print("ZS- TEST- AVERAGE")
    # acc_ = []
    df_dict_zs = defaultdict(dict)

    acc_dict = defaultdict(list)
    for feature_area, model_results in feature_area_result_dict_zs.items():
        print(f"{feature_area} : label_dim {feature_area_labels[feature_area]}")
        # acc_.append(model_results["random"][0])
        # print(f"{feature_area} : label_dim {feature_area_labels[feature_area]}")
        # print(model_results["random"])
        for m, acc in model_results.items():
            if m not in df_dict_zs:
                df_dict_zs[m] = defaultdict(dict)
            if "clics" in m:
                # model, nodeEmb, metric
                t1, t2, t3 = m.split("_")
                df_dict_zs[m]["NodeEmb"]= t2
                df_dict_zs[m]["Metric"] = t3
                df_dict_zs[m][feature_area]= acc[0]
            else:
                if m == "random":
                    df_dict[m]["NodeEmb"] = None
                    df_dict[m]["Metric"] = None
                    df_dict[m][feature_area] = acc[0]

            acc_dict[m].append(acc)
        model_restructed = sorted(model_results.items(), key=lambda x: x[1][0], reverse=True)
        print(model_restructed[:5])


    print("average")
    acc_dict_ = {k: np.average(v) for k, v in acc_dict.items()}

    print(sorted(acc_dict_.items(), key=lambda x: x[1], reverse=True))

    print("*" * 30)

    df_zs = pd.DataFrame.from_dict(df_dict_zs, columns=columns, orient="index")
    df_zs = df_zs.dropna(axis=1, how="all")
    print(df_zs)

    df_zs.to_csv("output/results/clics_metrics_nodeEmb_test_zs.csv")



if __name__ == '__main__':
    import plac

    plac.call(create_dfs)
