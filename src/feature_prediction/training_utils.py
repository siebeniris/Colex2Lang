import os
import json

import numpy as np
import torch
from torch import nn

from sklearn.metrics import classification_report, accuracy_score


def evaluate_dataset(model, data, feature_name, lang_dict, langs_list, mode="dev", language_vectors=None):
    """

    :param model: MODEL
    :param data: Test or Dev
    :param feature_name: feature from typology prediction
    :param lang_dict:
    :param mode: dev or test
    :return: results
    """
    model.eval()
    creterion = nn.CrossEntropyLoss()

    gold = []
    pred = []
    losses = []
    langs_embeddings = []
    total_langs = []

    with torch.no_grad():
        for lang, feature_value in zip(data["ISO"], data[feature_name]):
            lang_idx = lang_dict[lang]
            total_langs.append(lang)
            if langs_list is not None:
                # if language_vectors is not None:
                if lang in language_vectors.index_to_key and lang in langs_list:
                    langs_embeddings.append(lang)
                    language_vector = language_vectors[lang]
                    output = model(lang_idx, language_vector)
                else:
                    output = model(lang_idx, None)
            else:
                output = model(lang_idx, None)

            pred_label = np.argmax(output.detach().numpy())
            feature_value = torch.tensor(feature_value)

            if mode == "dev":
                loss = creterion(output, feature_value)
                loss = loss.detach().numpy()
                losses.append(loss)
            gold.append(feature_value)
            pred.append(pred_label)

    report = classification_report(gold, pred, output_dict=True)
    # acc = accuracy_score(gold, pred)
    acc = sum([gold[i] == pred[i] for i in range(len(gold))]) / len(gold)
    acc = float(acc.detach())

    if mode == "dev":
        return total_langs, langs_embeddings, acc, report, np.average(losses)
    else:
        return total_langs, langs_embeddings, acc, report, None


def train_model(model, model_name, optimizer, train_data, dev_data, test_data, feature_name, feature_id, lang_dict,
                langs_list,
                output_folder,
                max_epochs=10, language_vectors=None):
    model.train()
    creterion = nn.CrossEntropyLoss()
    best_acc = [0]
    train_langs_embeddings = []
    for epoch in range(max_epochs):
        gold = []
        pred = []
        losses = []
        train_total_langs = []
        for lang, feature_value in zip(train_data["ISO"], train_data[feature_name]):
            lang_idx = lang_dict[lang]
            train_total_langs.append(lang)
            # zero the parameter gradients
            optimizer.zero_grad()
            if langs_list is not None:
                # if language_vectors is not None:
                if lang in language_vectors.index_to_key and lang in langs_list:
                    train_langs_embeddings.append(lang)
                    language_vector = language_vectors[lang]
                    output = model(lang_idx, language_vector)
                else:
                    output = model(lang_idx, None)
            else:
                output = model(lang_idx, None)

            feature_value = torch.tensor(feature_value)
            # print(output)
            # print(feature_value)

            loss = creterion(output, feature_value)

            loss.backward()
            optimizer.step()

            output = output.detach().numpy()
            loss = loss.detach().numpy()

            gold.append(feature_value)
            output = np.argmax(output)
            pred.append(output)
            losses.append(loss)

        train_report = classification_report(gold, pred, output_dict=True)
        train_loss = np.average(losses)

        train_langs_embeddings = list(set(train_langs_embeddings))
        train_total_langs = list(set(train_total_langs))

        dev_langs, dev_langs_embed, dev_acc, dev_report, dev_loss = evaluate_dataset(model, dev_data, feature_name,
                                                                                     lang_dict, langs_list,
                                                                                     mode="dev",
                                                                                     language_vectors=language_vectors)
        dev_langs = list(set(dev_langs))
        dev_langs_embed = list(set(dev_langs_embed))

        if dev_acc > np.max(best_acc):
            best_acc.append(dev_acc)

            test_langs, test_lang_embeds, test_acc, test_report, _ = evaluate_dataset(model, test_data, feature_name,
                                                                                      lang_dict, langs_list,
                                                                                      mode="test",
                                                                                      language_vectors=language_vectors)

            test_langs_zs, test_lang_embeds_zs, test_acc_zs, test_report_zs, _ = evaluate_dataset(model, test_data,
                                                                                                  feature_name,
                                                                                                  lang_dict, None,
                                                                                                  mode="test")

            test_lang_embeds = list(set(test_lang_embeds))
            test_lang_embeds_zs = list(set(test_lang_embeds_zs))

            result = {
                "feature_name": feature_name,
                "feature_id": feature_id,
                "epoch": epoch,
                "train": {
                    "report": train_report,
                    "loss": str(train_loss),
                    "lang_embeds": train_langs_embeddings,
                    "lang_embeds_length": len(train_langs_embeddings)
                },
                "dev": {
                    "report": dev_report,
                    "loss": str(dev_loss),
                    "lang_embeds": dev_langs_embed,
                    "lang_embeds_length": len(dev_langs_embed)
                },
                "test": {
                    "report": test_report,
                    "acc": test_acc,
                    "lang_embeds": test_lang_embeds,
                    "lang_embeds_length": len(test_lang_embeds)
                },
                "test_zs": {
                    "report": test_report_zs,
                    "acc": test_acc_zs,
                    "lang_embeds": test_lang_embeds_zs,
                    "lang_embeds_length": len(test_lang_embeds_zs)
                }

            }
            print(
                f"total langs: train {len(list(set(train_total_langs)))} dev {len(list(set(dev_langs)))} test {len(list(set(test_langs)))}")
            print(
                f"langs with embeddings :train {len(list(set(train_langs_embeddings)))} dev {len(list(set(dev_langs_embed)))} test {len(list(set(test_lang_embeds)))} ")

            file_name = f"{model_name}_{feature_id}"

            print(f"output model {file_name}")
            with open(os.path.join(output_folder, f"{file_name}.json"), "w") as f:
                json.dump(result, f)

            torch.save(model.state_dict(), os.path.join(output_folder, file_name))


def load_dataset(df, feature_name):
    df_feature = df[["wals_code", feature_name]].dropna()
    df_feature[feature_name] = df_feature[feature_name].astype("int")
    return df_feature, df_feature["wals_code"].tolist()
