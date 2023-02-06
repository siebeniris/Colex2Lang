import os
import json

import numpy as np

from sklearn.metrics import classification_report


def train_model_splits(model_name, train_dev_splits, train_dev,
                       test_data, feature_name, feature_id, lang_dict,
                       langs_list,
                       output_folder,
                       ):
    best_acc = [0]

    for i, (train_index, dev_index) in enumerate(train_dev_splits):
        # print(f"{epoch}-{i} ")
        train_data = train_dev.iloc[train_index]
        dev_data = train_dev.iloc[dev_index]

        majority = train_data[feature_name].value_counts().idxmax()
        gold = train_data[feature_name].tolist()
        pred = [majority for x in range(len(gold))]

        train_report = classification_report(gold, pred, output_dict=True)

        # dev evaluation
        gold_dev = dev_data[feature_name].tolist()
        pred_dev = [majority for x in range(len(gold_dev))]
        dev_report = classification_report(gold, pred, output_dict=True)
        acc = sum([gold[i] == pred[i] for i in range(len(gold))]) / len(gold)

        if acc > np.max(best_acc):
            best_acc.append(acc)

            # test evaluation
            gold_test = test_data[feature_name].tolist()
            pred_test = [majority for _ in range(len(gold_test))]
            test_report = classification_report(gold_test, pred_test, output_dict=True)

            result = {
                "feature_name": feature_name,
                "feature_id": feature_id,
                "k": i,
                "train": {
                    "report": train_report,

                },
                "dev": {
                    "report": dev_report,

                },
                "test": {
                    "report": test_report,

                }

            }

            file_name = f"{model_name}_{feature_id}"

            print(f"output model {file_name}")
            with open(os.path.join(output_folder, f"{file_name}.json"), "w") as f:
                json.dump(result, f)
