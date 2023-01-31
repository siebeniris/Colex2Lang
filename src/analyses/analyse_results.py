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

    print("ZS- TEST- AVERAGE")
    acc_ = []

    for feature_area, model_results in feature_area_result_dict.items():
        print(f"{feature_area} : label_dim {feature_area_labels[feature_area]}")
        print(model_results["random"])
        acc_.append(model_results["random"][0])

    print("*" * 30)
    print(np.average(acc_))
    #
    #
    # print("ZS- TEST- AVERAGE")
    # acc_ = []
    # for feature_area, model_results in feature_area_result_dict_zs.items():
    #     acc_.append(model_results["random"][0])
    #     print(f"{feature_area} : label_dim {feature_area_labels[feature_area]}")
    #     print(model_results["random"])
    # print("*" * 30)
    #
    # print(np.average(acc_))


if __name__ == '__main__':
    import plac

    plac.call(show_results)
