import os
import json

import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings

warnings.filterwarnings("ignore")

torch.manual_seed(42)


class OneFF(nn.Module):
    def __init__(self, num_langs, input_dim, hidden_dim, label_dim, dropout):
        super(OneFF, self).__init__()
        self.label_dim = label_dim
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, self.label_dim)
        self.num_langs = num_langs
        self.input_dim = input_dim

    def forward(self, input_idx):
        emb1 = nn.Embedding(self.num_langs, self.input_dim)
        emb1 = nn.init.uniform_(emb1.weight, -1, 1.0)
        input_embeddings = emb1[input_idx]
        fc_output = self.fc(input_embeddings)
        droput_layer = self.dropout_layer(fc_output)
        output = self.classifier(droput_layer)

        return output


def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        m.weight.data.uniform_(-1.0, 1.0)
        print(m.weight)


def evaluate_dataset(model, data, feature_name, lang_dict, mode="dev"):
    creterion = nn.CrossEntropyLoss()
    gold = []
    pred = []
    losses = []
    with torch.no_grad():
        for lang, feature_value in zip(data["wals_code"], data[feature_name]):
            lang_idx = lang_dict[lang]
            output = model(lang_idx)
            pred_label = np.argmax(output.detach().numpy())
            feature_value = torch.tensor(feature_value)
            # print(output)
            # print(feature_value)
            if mode == "dev":
                loss = creterion(output, feature_value)
                loss = loss.detach().numpy()
                losses.append(loss)
            gold.append(feature_value)
            pred.append(pred_label)

    report = classification_report(gold, pred, output_dict=True)
    acc = accuracy_score(gold, pred)
    if mode == "dev":
        return acc, report, np.average(losses)
    else:
        return acc, report, None


def train_model(model, optimizer, train_data, dev_data, test_data, feature_name, lang_dict, output_folder,
                max_epochs=10):
    creterion = nn.CrossEntropyLoss()
    best_acc = [0]
    for epoch in range(max_epochs):
        gold = []
        pred = []
        losses = []
        for lang, feature_value in zip(train_data["wals_code"], train_data[feature_name]):
            lang_idx = lang_dict[lang]

            # zero the parameter gradients
            optimizer.zero_grad()
            output = model(lang_idx)
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
        acc = accuracy_score(gold, pred)

        dev_acc, dev_report, dev_loss = evaluate_dataset(model, dev_data, feature_name, lang_dict, mode="dev")

        if dev_acc > np.max(best_acc):
            # print(f"Epoch {epoch} ")
            # print("training ")
            # print(f"classification report {train_report}")
            # print(f"accuracy {acc}")
            # print(f"avg loss {train_loss}")
            #
            # print("dev...")
            # print(f"accuracy {dev_acc}")
            # print(f"report {dev_report}")
            # print(f"avg loss{dev_loss}")
            best_acc.append(dev_acc)

            test_acc, test_report, _ = evaluate_dataset(model, test_data, feature_name, lang_dict, mode="test")
            # print(f"testing")
            # print(f"acc {test_acc}")
            # print(f"report {test_report}")

            result = {
                "epoch": epoch,
                "train": {
                    "report": train_report,
                    "loss": str(train_loss)
                },
                "dev": {
                    "report": dev_report,
                    "loss": str(dev_loss)
                },
                "test": {
                    "report": test_report
                }
            }

            model_name = f"oneff_{feature_name}"
            print(f"output model {model_name}")
            with open(os.path.join(output_folder, f"{model_name}.json"), "w") as f:
                json.dump(result, f)

            torch.save(model.state_dict(), os.path.join(output_folder, model_name))


def load_dataset(df, feature_name):
    df_feature = df[["wals_code", feature_name]].dropna()
    df_feature[feature_name] = df_feature[feature_name].astype("int")
    return df_feature, df["wals_code"].tolist()


def run(device="cpu"):
    # device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = device
    print(f"Using {device} device")
    feature_name = "Number_of_Non-Derived_Basic_Colour_Categories"


    with open("data/TypPred/preprocessed/feature_maps.json") as f:
        feature_maps = json.load(f)

    train_file = os.path.join("data/TypPred/preprocessed", "train.csv")
    dev_file = os.path.join("data/TypPred/preprocessed", "dev.csv")
    test_file = os.path.join("data/TypPred/preprocessed", "test.csv")
    train_df = pd.read_csv(train_file)
    dev_df = pd.read_csv(dev_file)
    test_df = pd.read_csv(test_file)

    for feature_name in list(feature_maps.keys())[:10]:

        label_dim = len(feature_maps[feature_name])

        train_data, langs_train = load_dataset(train_df, feature_name)
        dev_data, langs_dev = load_dataset(dev_df, feature_name)
        test_data, langs_test = load_dataset(test_df, feature_name)

        all_langs = list(set(langs_dev) | set(langs_train) | set(langs_test))
        num_langs = len(all_langs)
        print(f"feature {feature_name} nr of langs {num_langs}, {all_langs[:10]}")
        lang_dict = {lang: idx for idx, lang in enumerate(all_langs)}

        model = OneFF(num_langs=num_langs, input_dim=100, hidden_dim=200, label_dim=label_dim, dropout=0.5)
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        output_folder = "output/models/oneff/"
        train_model(model, optimizer, train_data, dev_data, test_data, feature_name, lang_dict, output_folder,
                    max_epochs=100)


if __name__ == '__main__':
    import plac
    plac.call(run)
